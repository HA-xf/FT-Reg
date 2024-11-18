#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import time
import numpy as np
import torch
import sys





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
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=4, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=150,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=4e-4, help='learning rate (default: 1e-4)')
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

bidir = args.bidir

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

# print('train_files_1:',train_files_1)
# print('train_files_2:',train_files_2)
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
    
# for item in generator:
#     print('item',item)
    

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)


# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense_fea_trans.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense_fea_trans(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )
    # model = vxm.networks.VxmDense(
    #     inshape=inshape,
    #     nb_unet_features=[enc_nf, dec_nf],
    #     bidir=bidir,
    #     int_steps=args.int_steps,
    #     int_downsize=args.int_downsize
    # )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
        overlap = vxm.py.utils.dice(numpy_array_warped_seg, numpy_array_fixed_seg)
        
        # overlap = vxm.py.utils.my_dsc(numpy_array_fixed_seg, numpy_array_warped_seg)
        if len(overlap) == 0:
        # dice_mean.append(0)
            pass
        else:
            dice_means.append(np.mean(overlap))
        
        
        # # calculate total loss
        # loss = 0
        # loss_list = []
        # for n, loss_function in enumerate(losses):
        #     # print('y_true[n].shape',y_true[n].shape)
        #     curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
        #     loss_list.append(curr_loss.item())
        #     loss += curr_loss

        # epoch_loss.append(loss_list)
        # epoch_total_loss.append(loss.item())
        
        # # print epoch info
        # epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        # time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        # losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        # loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        # print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        
    
        # print('Pair %d    Reg Time: %.4f    Dice: %.4f +/- %.4f' % (i + 1, reg_time,
        #                                                             np.mean(overlap),
        #                                                             np.std(overlap)))


    print('Avg Reg Time: %.4f +/- %.4f  (skipping first prediction)' % (np.mean(reg_times),
                                                                        np.std(reg_times)))
    accurent_dice = np.mean(dice_means)
    print('Avg Dice: %.4f +/- %.4f' % (accurent_dice, np.std(dice_means)))
    print('len(dice_means)',len(dice_means))
    
    if best_dice < accurent_dice:
        best_dice = accurent_dice
        # save model checkpoint  
        model.save(os.path.join(model_dir, '%04d_best.pt' % epoch))
        print('model.save success')
        
    print('best_dice: %.4f' % (best_dice))
    
    return best_dice





# for param in model.parameters():
#     param.requires_grad = False


# training loops
for epoch in range(args.initial_epoch, args.epochs):
    # make sure to train
    model.train()

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

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs)
        
        # print('y_pred[0]',y_pred[0].shape)
        # print('y_pred[1]',y_pred[1].shape)
        # print('y_true[0]',y_true[0].shape)
        # print('y_true[1]',y_true[1].shape)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            # print('y_true[n].shape',y_true[n].shape)
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
    
    # val
    best_dice = every_epoch_val(best_dice)
    

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
