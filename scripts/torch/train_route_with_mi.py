#!/usr/bin/env python

import os
import random
import argparse
import time
import numpy as np
import torch
import sys
import itertools
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F



# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-list-1', required=True, help='line-seperated list of training files')
parser.add_argument('--img-list-2', required=True, help='line-seperated list of training files')

parser.add_argument('--train-img-prefix', help='optional input image file prefix')
parser.add_argument('--train-img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='1', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=300,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mi',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')

# val
parser.add_argument('--pairs', required=True, help='path to list of image pairs to register')
parser.add_argument('--img-suffix', help='input image file suffix')
parser.add_argument('--seg-suffix', help='input seg file suffix')
parser.add_argument('--img-prefix', help='input image file prefix')
parser.add_argument('--seg-prefix', help='input seg file prefix')
parser.add_argument('--labels', help='optional label list to compute dice for (in npy format)')
args = parser.parse_args()



def freeze(item):
    for param in item.parameters():
        param.requires_grad = False
def unfreeze(item):
    for param in item.parameters():
        param.requires_grad = True

bidir = args.bidir


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)


# load val data
if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
    print('Error: Must provide a differing file suffix and/or prefix for images and segs.')
    exit(1)
img_pairs = vxm.py.utils.read_pair_list_modify(args.pairs, prefix=args.img_prefix, suffix=args.img_suffix)
seg_pairs = vxm.py.utils.read_pair_list_modify(args.pairs, prefix=args.seg_prefix, suffix=args.seg_suffix)
# load seg labels if provided
labels = np.load(args.labels) if args.labels else None
# check if multi-channel data
add_feat_axis = not args.multichannel



# load and prepare training data
# note that train_files is a dictionary
train_files_1 = vxm.py.utils.read_file_list(args.img_list_1)
assert len(train_files_1) > 0, 'Could not find any training data.'
train_files_2 = vxm.py.utils.read_file_list(args.img_list_2)
assert len(train_files_2) > 0, 'Could not find any training data.'


print('train_files_1:',len(train_files_1))
print('train_files_2:',len(train_files_2))

# sys.exit()

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    # generator = vxm.generators.scan_to_atlas(train_files, atlas,
    #                                          batch_size=args.batch_size, bidir=args.bidir,
    #                                          add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(
        train_files_1, train_files_2, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
 

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# # prepare model folder
# model_dir = args.model_dir
# os.makedirs(model_dir, exist_ok=True)


# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

model_0 = vxm.networks.VxmDense_joint_level0.load('/scripts/torch/model_joint_R_G_level0/reg_best.pt', device)
model_1 = vxm.networks.VxmDense_joint_level1.load('/scripts/torch/model_joint_R_G_level1/reg_best.pt', device)
model_2 = vxm.networks.VxmDense_joint_level2.load('/scripts/torch/model_joint_R_G_level2/reg_best.pt', device)
model_3 = vxm.networks.VxmDense_joint_level3.load('/scripts/torch/model_joint_R_G_level3/reg_best.pt', device)

# if nb_gpus > 1:
#     # use multiple GPUs via DataParallel
#     model = torch.nn.DataParallel(model)
#     model.save = model.module.save

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
elif args.image_loss == 'mi':
    image_loss_func = vxm.losses.MI(2,normalized = True).mi 
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]



# prepare the model for training and send to device
model_0.to(device)
model_1.to(device)
model_2.to(device)
model_3.to(device)
model_0.eval()
model_1.eval()
model_2.eval()
model_3.eval()

model = [model_0,model_1,model_2,model_3]

# route
model_route = vxm.networks.networks_route.Route(4)
model_route.to(device)
model_route.train()

# set optimizer
optimizer = torch.optim.Adam(model_route.parameters(), lr=args.lr)




# best_dice
best_dice = 0

def every_epoch_val(best_dice,epoch):
    with open('/route_with_label_choice.txt', 'a', encoding = 'utf-8') as f:
        f.write('\n'+'epoch_'+ str(epoch) + ': ')
    # make sure to val
    # model.eval()
    # keep track of all dice scores
    model_route.eval()
    reg_times = []
    dice_means = []
    
    for i in range(len(img_pairs)):

        # load moving image and seg, note that seg is no need normalized
        moving_vol = vxm.py.utils.load_volfile_normalization(
            img_pairs[i][0], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg = vxm.py.utils.load_volfile(
            seg_pairs[i][0], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)

        # load fixed image and seg
        fixed_vol = vxm.py.utils.load_volfile_normalization(
            img_pairs[i][1], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed_seg = vxm.py.utils.load_volfile(
            seg_pairs[i][1], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        
    
        moving_vol = torch.from_numpy(moving_vol).to(device).float().permute(0, 3, 1, 2)
        moving_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 3, 1, 2)
        fixed_vol = torch.from_numpy(fixed_vol).to(device).float().permute(0, 3, 1, 2)
        fixed_seg = torch.from_numpy(fixed_seg).to(device).float().permute(0, 3, 1, 2)

        # predict warp and time
        start = time.time()
        
        logits = model_route(moving_vol,fixed_vol)
        logits = logits.squeeze()
        logits_max = torch.argmax(logits)
        logits_max = logits_max.cpu() 
        # logits_max = logits_max.item
        logits_max = int(logits_max)
        with open('/route_with_label_choice.txt', 'a', encoding = 'utf-8') as f:
            f.write(str(logits_max))
        
        
        pos_flow = model[1].get_pos_flow(moving_vol,fixed_vol)
        
        reg_time = time.time() - start
        if i != 0:
            # first keras prediction is generally rather slow
            reg_times.append(reg_time)

        # apply transform
        warped_seg = model[1].predict_label(moving_seg, pos_flow)
        
        cpu_warped_seg = warped_seg.cpu()
        cpu_fixed_seg = fixed_seg.cpu()
        

        numpy_array_warped_seg = cpu_warped_seg.detach().numpy()
        numpy_array_fixed_seg = cpu_fixed_seg.detach().numpy()



        # compute volume overlap (dice)
        overlap = vxm.py.utils.dice(numpy_array_warped_seg, numpy_array_fixed_seg, labels = labels)
        
        # overlap = vxm.py.utils.my_dsc(numpy_array_fixed_seg, numpy_array_warped_seg)
        if len(overlap) == 0:
        # dice_mean.append(0)
            pass
        else:
            dice_means.append(np.mean(overlap))       
    print('Avg Reg Time: %.4f +/- %.4f  (skipping first prediction)' % (np.mean(reg_times),
                                                                        np.std(reg_times)))
    accurent_dice = np.mean(dice_means)
    print('Avg Dice: %.4f +/- %.4f' % (accurent_dice, np.std(dice_means)))
    print('len(dice_means)',len(dice_means))
    if best_dice < accurent_dice:
        best_dice = accurent_dice
        torch.save(model_route.state_dict(), 'model_route_with_label/best_model_route_with_mi.pth')

        
    print('best_dice: %.4f' % (best_dice))
    return best_dice


# Loss plot
logger = vxm.py.utils.Logger(args.epochs, args.steps_per_epoch)


loss_entropy = torch.nn.CrossEntropyLoss()

# training loops
for epoch in range(args.initial_epoch, args.epochs):
    # make sure to train
    model_route.train()

    # # save model checkpoint
    # if epoch % 10 == 0:
    #     model.save(os.path.join(model_dir, '%04d.pt' % epoch))
    #     print('model.save success')

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)
        # for item in inputs:
        #     print('item.shape',item.shape)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]
        # print('y_true',len(y_true))
        
        # route
        logits = model_route(inputs[0], inputs[1]) 
        logits = logits.squeeze() 
        # print('logits',logits)
        
        
        # mk labels
        
        y_pred = []
        for i in range(4):
            y_pred.append(model[i](inputs[0], inputs[1]))
            
                

        # calculate total loss
        sample_mi = [0,0,0,0]
        
        for i in range(4):
            curr_loss = losses[0](y_true[0], y_pred[i][0]) * weights[0]  
            sample_mi[i] = -curr_loss
        
        # print('sample_mi',sample_mi)
        # def to_one_hot(lst):
        #     max_value = max(lst)
        #     one_hot_vector = [0] * len(lst)
        #     index = lst.index(max_value)
        #     one_hot_vector[index] = 1
        #     return one_hot_vector
       
        
        # sample_mi_hard = to_one_hot(sample_mi)          
        # sample_mi_hard = torch.tensor(sample_mi_hard,dtype=float)    
        # sample_mi_hard = sample_mi_hard.detach()
        # sample_mi_hard = sample_mi_hard.cuda()
        
        sample_mi = torch.tensor(sample_mi,dtype=float)
        sample_mi = sample_mi.detach()
        sample_mi = sample_mi.cuda()
        
        sample_mi_prob = F.softmax(sample_mi,dim=0)

       
        # print('sample_mi_prob',sample_mi_prob) 
        # print('sample_mi_hard',sample_mi_hard)
        # print('logits',logits)
        # print('type(sample_mi_hard)',type(sample_mi_hard))
        # print('type(logits)',type(logits))
        
    
        loss = loss_entropy(logits, sample_mi_prob)

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  

# epoch = 1

    best_dice = every_epoch_val(best_dice, epoch)
