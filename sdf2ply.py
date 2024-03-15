from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import random
import torch
import numpy as np
import gc

import data_util
import scene_dataloader
import model
import loss as loss_util
from math import ceil
from math import floor
import struct

import plyfile

import marching_cubes.marching_cubes as mc
# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=5, help='which gpu to use')
#parser.add_argument('--data_path', default='/mnt/data2/wzq/room_train_data_input_target1',required=False, help='path to data')
#parser.add_argument('--train_file_list',default='../filelists/room_train_data_target_input_block_files.txt', required=False, help='path to file list of train data')
#parser.add_argument('--target_data_path', default='../../../../zwx/mp_sdf_vox_2cm_target',required=False, help='path to target data')
# parser.add_argument('--target_data_path', default='/mnt/data3/wzq-sgnn/mp_sdf_vox_2cm_scanned0.3all/',required=False, help='path to target data')
parser.add_argument('--target_data_path', default="/mnt/data4/wzq/output_spsg_rebuttal0.97/mp_sdf_vox_2cm_scanned/",required=False, help='path to target data')
#parser.add_argument('--test_file_list',default='../filelists/mp-rooms_test-scenes.txt', required=False, help='path to file list of test data')
# parser.add_argument('--test_file_list',default='../filelists/proportion_0.3all.txt', required=False, help='path to file list of test data')
parser.add_argument('--test_file_list',default='../filelists/rebuttal_scannet.txt', required=False, help='path to file list of test data')
# parser.add_argument('--output_path', default='./test_target_ply_ours',required=False, help='path to target data')
parser.add_argument('--output_path', default="/mnt/data4/wzq/output_spsg_rebuttal0.97/target/",required=False, help='path to target data')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
args = parser.parse_args()
UNK_THRESH = 2
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
#get file name
id=0
target_files, _ = data_util.get_train_files(args.target_data_path, args.test_file_list, '')
target_files=target_files[id:]
names = open(args.test_file_list).read().splitlines()

# names = [name[0:-5] for name in names]
names = [name+'__0_' for name in names]
names=names[id:]
print(target_files,names)
print('#train files = ', len(target_files),len(names))
test_num=len(target_files)

for i in range(test_num):
    [target_locs, target_sdf], [dimz, dimy, dimx], world2grid= data_util.load_scene(target_files[i])
    target_sdfs1 = data_util.sparse_to_dense_np(target_locs, target_sdf[:, np.newaxis], dimx, dimy, dimz, 1000)
    print(target_sdfs1.shape)
    target_sdfs1=target_sdfs1[:min(128,target_sdfs1.shape[0]),:,:]
    print(target_sdfs1.shape)
    isovalue = 0
    trunc = args.truncation - 0.1
    ext = '.ply'
    target = target_sdfs1
    mc.marching_cubes(torch.from_numpy(target), None, isovalue=isovalue, truncation=trunc, thresh=10,
                     output_filename=os.path.join(args.output_path, names[i] + '_target-mesh' + ext))
    print(id+i,'done')