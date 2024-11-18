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

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense_joint_level2.load(args.load_model, device)
else:
#     # otherwise configure new model
    model = vxm.networks.VxmDense_joint_level2(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )


if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()
# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

netG_A2B = vxm.networks_cyclegan.Generator(32, 32)
netG_B2A = vxm.networks_cyclegan.Generator(32, 32)
netD_A = vxm.networks_cyclegan.Discriminator(32)
netD_B = vxm.networks_cyclegan.Discriminator(32)

netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)
netG_A2B.train()
netG_B2A.train()
netD_A.train()
netD_B.train()

netG_A2B.apply(vxm.py.utils.weights_init_normal)
netG_B2A.apply(vxm.py.utils.weights_init_normal)
netD_A.apply(vxm.py.utils.weights_init_normal)
netD_B.apply(vxm.py.utils.weights_init_normal)

torch.save(netG_A2B.state_dict(), 'model_joint_R_G_level2/netG_A2B.pth')
torch.save(netG_B2A.state_dict(), 'model_joint_R_G_level2/netG_B2A.pth')
torch.save(netD_A.state_dict(), 'model_joint_R_G_level2/netD_A.pth')
torch.save(netD_B.state_dict(), 'model_joint_R_G_level2/netD_B.pth')

# registration items
# set optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    losses_registration = [image_loss_func, image_loss_func]
    weights_registration = [0.5, 0.5]
else:
    losses_registration = [image_loss_func]
    weights_registration = [1]

# prepare deformation loss
losses_registration += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights_registration += [args.weight]



# cyclegan items

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=0.0002, betas=(0.5, 0.999))

optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
# all starting decay
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=vxm.py.utils.LambdaLR(args.epochs, 0, 100).step)
# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=vxm.py.utils.LambdaLR(args.epochs, 0, 100).step)

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=vxm.py.utils.LambdaLR(args.epochs, 0, 100).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=vxm.py.utils.LambdaLR(args.epochs, 0, 100).step)


# best_dice
best_dice = 0
# val fuction

def every_epoch_val(best_dice):
    # make sure to val
    model.eval()
    # keep track of all dice scores
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
        pos_flow = model.get_pos_flow(moving_vol,fixed_vol)
        
        reg_time = time.time() - start
        if i != 0:
            # first keras prediction is generally rather slow
            reg_times.append(reg_time)

        # apply transform
        warped_seg = model.predict_label(moving_seg, pos_flow)
        
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
        # save registration network
        model.save(os.path.join('model_joint_R_G_level2', '%04d_best.pt' % epoch))
        # save cyclegan
        torch.save(netG_A2B.state_dict(), 'model_joint_R_G_level2/best_netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'model_joint_R_G_level2/best_netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'model_joint_R_G_level2/best_netD_A.pth')
        torch.save(netD_B.state_dict(), 'model_joint_R_G_level2/best_netD_B.pth')

        
        print('cyclegan_model.save success')    
    print('best_dice: %.4f' % (best_dice))
    return best_dice


# Loss plot
logger = vxm.py.utils.Logger(args.epochs, args.steps_per_epoch)

target_real = torch.ones((args.batch_size,), device='cuda', requires_grad=False)
target_fake = torch.zeros((args.batch_size,), device='cuda', requires_grad=False)

fake_A_buffer = vxm.py.utils.ReplayBuffer()
fake_B_buffer = vxm.py.utils.ReplayBuffer()



# training loops
for epoch in range(args.initial_epoch, args.epochs):
    # make sure to train

    for step in range(args.steps_per_epoch):


        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)
        # for item in inputs:
        #     print('item.shape',item.shape)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]
      
        y_pred = model(*inputs)
        # y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]
        
        # real_A, real_B = model.get_fea_for_cyclegan(*inputs)
        real_A = y_pred[-2]
        real_B = y_pred[-1]
        # real_A = real_A.detach()
        # real_B = real_B.detach()
        # print('len(y_pred)',len(y_pred))
        
       


        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # print('aaaa')
        freeze(model)
        unfreeze(netG_A2B)
        unfreeze(netG_B2A)
        unfreeze(netD_A)
        unfreeze(netD_B)
        
        # optimizer.zero_grad()
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        # print('real_B.shape',real_B.shape)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        # loss_joint = loss_G + loss
        
        # for param in model.parameters():
        #     param.requires_grad = True
        
        # optimizer.zero_grad()
    
        # loss_joint.backward(retain_graph=True)
        loss_G.backward(retain_graph=True)
        # loss.backward(retain_graph=True)
        # print('aaaa')
        
        # optimizer.step()
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()
        
        # for param in model.parameters():
        #     param.requires_grad = False

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward(retain_graph=True)

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward(retain_graph=True)

        optimizer_D_B.step()
        ###################################
        # print('ohhhhhhhhhhhhhhh')
        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
        # print('hohhhhhhhhhhhhhhh')
        
        
        unfreeze(model)
        freeze(netG_A2B)
        freeze(netG_B2A)
        freeze(netD_A)
        freeze(netD_B)
         # calculate loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses_registration):
            # print('y_true[n].shape',y_true[n].shape)
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights_registration[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('bbbbb')
        
        
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'model_joint_R_G_level2/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'model_joint_R_G_level2/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'model_joint_R_G_level2/netD_A.pth')
    torch.save(netD_B.state_dict(), 'model_joint_R_G_level2/netD_B.pth')
    
        


    # val
    best_dice = every_epoch_val(best_dice)
        

