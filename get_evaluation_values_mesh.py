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
import marching_cubes.marching_cubes as mc
from math import ceil
from math import floor
import loss as loss_util
import csv
from pytorch3d.structures import Meshes
#,Pointclouds,join_meshes_as_batch
from pytorch3d.loss import (
   chamfer_distance,
   mesh_edge_loss,
   mesh_laplacian_smoothing,
   mesh_normal_consistency,
)
UNK_THRESH=2
# python test_scene.py --gpu 0 --input_data_path ./data/mp_sdf_vox_2cm_input --target_data_path ./data/mp_sdf_vox_2cm_target --test_file_list filelists/mp_rooms_1.txt --output output/mp
#python test_scene.py --gpu 0 --input_data_path ../../../zwx/mp_sdf_vox_2cm_input --target_data_path ../../../zwx/mp_sdf_vox_2cm_target --test_file_list ../filelists/mp-rooms_val-scenes.txt --model_path ../sgnn.pth --output ./output  --max_to_vis 20
#python test_scene.py --gpu 0 --input_data_path ../../../zwx/mp_sdf_vox_2cm_input --target_data_path ../../../zwx/mp_sdf_vox_2cm_target --test_file_list ../filelists/mp-rooms_val-scenes.txt --model_path ../../sgnn-master/torch/logs/mp-test/model-epoch-4.pth --output ./output_overfit  --max_to_vis 10


# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=5, help='which gpu to use')
parser.add_argument('--input_data_path', default='../../../../zwx/mp_sdf_vox_2cm_input',required=False, help='path to input data')
parser.add_argument('--target_data_path', default='../../../../zwx/mp_sdf_vox_2cm_target',required=False, help='path to target data')
parser.add_argument('--test_file_list',default='../filelists/mp-rooms_test-scenes.txt', required=False, help='path to file list of test data')
parser.add_argument('--model_path',default="/mnt/data2/wzq/pycharm/stage1-connect2layer-transformer-nomask1-6-821x8-normal3D2-IOU2/torch/logs/stage1-connect2layer-transformer-nomask1-6-kvrec-821x8-normal3D2-iou2-test/model-epoch-4.pth", required=False, help='path to model to test')
#parser.add_argument('--model_path',default="/mnt/data2/wzq/pycharm/stage1-connect2layer-transformer-nomask1-6-821x8-normal3D/torch/logs/stage1-connect2layer-transformer-nomask1-6-kvrec-821x8-normal3D/model-epoch-4.pth", required=False, help='path to model to test')
parser.add_argument('--output', default='./mp-test-select', help='folder to output predictions')
parser.add_argument('--csv_path',default='./logs/evaluation-mesh2.csv', required=False, help='path to model to test')
# parser.add_argument('--input_data_path', default='/mnt/data2/wzq/pycharm/sgnn-master1/',required=False, help='path to input data')
# parser.add_argument('--target_data_path', default='/mnt/data2/wzq/pycharm/sgnn-master1/',required=False, help='path to target data')
# parser.add_argument('--test_file_list',default='../filelists/scannet_test.txt', required=False, help='path to file list of test data')
# parser.add_argument('--model_path',default="/mnt/data2/wzq/pycharm/sgnn-master1/sgnn.pth", required=False, help='path to model to test')
# #parser.add_argument('--model_path',default="/mnt/data2/wzq/pycharm/stage1-connect2layer-transformer-nomask1-6-821x8-normal3D2-IOU2/torch/logs/stage1-connect2layer-transformer-nomask1-6-kvrec-821x8-normal3D2-iou2/model-epoch-4.pth", required=False, help='path to model to test')
# parser.add_argument('--output', default='./scannet_test', help='folder to output predictions')
# model params
parser.add_argument('--num_hierarchy_levels', type=int, default=4, help='#hierarchy levels.')
parser.add_argument('--max_input_height', type=int, default=128, help='max height in voxels')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--input_dim', type=int, default=128, help='voxel dim.')
parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
# test params
parser.add_argument('--max_to_vis', type=int, default=400, help='max num to vis')
parser.add_argument('--cpu', dest='cpu', action='store_true')


parser.set_defaults(no_pass_occ=False, no_pass_feats=False, cpu=False)
args = parser.parse_args()
assert( not (args.no_pass_feats and args.no_pass_occ) )
assert( args.num_hierarchy_levels > 1 )
args.input_nf = 1
print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
UP_AXIS = 0 # z is 0th
#csv recording
header_list=["test_num","entire","unobserved","target","prediction"]

# create model
model = model.GenModel(args.encoder_dim, args.input_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, not args.no_pass_occ, not args.no_pass_feats, args.use_skip_sparse, args.use_skip_dense)
if not args.cpu:
    model = model.cuda()
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)

def  cut_pro_block_scence(inputs,orig_dims,loss_weights,sample):
    a1=orig_dims
    Z,Y,X=orig_dims[0]
    start=np.zeros(128*Y*X)
    start = start[:, np.newaxis]
    point_num=np.zeros(128*Y*X)
    if False:#args.compute_pred_occs:
        start_occ=[None]*args.num_hierarchy_levels
        point_num_occ=[None]*args.num_hierarchy_levels
        for h in range(args.num_hierarchy_levels):
            factor1 = 2 ** (args.num_hierarchy_levels - h - 1)
            start_occ[h]=np.zeros((128//factor1)*(Y//factor1)*(X//factor1))
            # start_occ[h]=start_occ[h][:, np.newaxis]
            point_num_occ[h] = np.zeros((128//factor1)*(Y//factor1)*(X//factor1))
    stride_x=56
    stride_y=56
    wide_x = 64
    wide_y = 64
    # print('Z,Y,X',Z,Y,X)
    for y in range(int(ceil(float(Y-wide_y)/stride_y))+1):
        for x in range(int(ceil(float(X-wide_x)/stride_x))+1):
            mask_y=np.array(inputs[0][:,1]>=y*stride_y)  & np.array(inputs[0][:,1]<(y*stride_y+wide_y))
            mask_x = np.array(inputs[0][:, 2] >= x * stride_x) & np.array(inputs[0][:, 2] < (x * stride_x + wide_x))
            mask=mask_y & mask_x
            sum_point = np.sum(mask)
            print('sum',sum_point)
            print(y*(int(ceil(float(X-wide_x)/stride_x))+1)+x+1,'--',(int(ceil(float(Y-wide_y)/stride_y))+1)*(int(ceil(float(X-wide_x)/stride_x))+1))
            if(sum_point>100):
                block_input_locs=inputs[0][mask]
                block_input_feats=inputs[1][mask]
                block_input_locs=np.array(block_input_locs[:])-[0,y*stride_y,x*stride_x,0]
                block_input_locs=torch.from_numpy(block_input_locs)
                input=[block_input_locs, block_input_feats]
                # print('max',torch.max(input[0][:,0]),torch.max(input[0][:,1]),torch.max(input[0][:,2]),torch.max(input[0][:,3]))
                # try:
                #     if not args.cpu:
                input[1] = input[1].cuda()
                output_sdf, output_occs = model(input, loss_weights)
                # print('output_sdf', output_sdf)
                # print('output_occs', output_occs)
                if False:#args.compute_pred_occs:
                    pred_occs = [None] * args.num_hierarchy_levels
                    for h in range(args.num_hierarchy_levels):
                        if len(output_occs[h][0]) == 0:
                            continue
                        output_occs[h][1] = torch.nn.Sigmoid()(output_occs[h][1][:, 0].detach()) > 0.5
                        locs=output_occs[h][0][:,:-1]
                        vals=output_occs[h][1]
                        pred_occs[h]=locs[vals.view(-1)]
                        #if pred_occs[h] is not None:
                            #pred_occs[h] = pred_occs[h].cpu().numpy()
                #print('pred_occs',pred_occs)
                # print('output_sdf',torch.max(output_sdf[0][:,0]),torch.max(output_sdf[0][:,1]),torch.max(output_sdf[0][:,2]))
                # print('pred_occs1',torch.max(pred_occs[0][:,0]),torch.max(pred_occs[0][:,1]),torch.max(pred_occs[0][:,2]))
                # print('pred_occs2',torch.max(pred_occs[1][:,0]),torch.max(pred_occs[1][:,1]),torch.max(pred_occs[1][:,2]))
                # print('pred_occs3',torch.max(pred_occs[2][:,0]),torch.max(pred_occs[2][:,1]),torch.max(pred_occs[2][:,2]))
                # print('pred_occs4',torch.max(pred_occs[3][:,0]),torch.max(pred_occs[3][:,1]),torch.max(pred_occs[3][:,2]))
                # print('pred_occs4-all',pred_occs[3])
                        #print('output_sdf inner',output_sdf,output_sdf[1].type())
                # except:
                #     print('exception at %s' % sample['name'])
                #     gc.collect()
                #     continue
                #local coordinate to global coordinate
                #print('output_sdf[0]',output_sdf[0][1:100])
                # print('max1', torch.max(output_sdf[0][:, 0]), torch.max(output_sdf[0][:, 1]), torch.max(output_sdf[0][:, 2]),
                #       torch.max(output_sdf[0][:, 3]))
                if False:#args.compute_pred_occs:
                    for h in range(args.num_hierarchy_levels):
                        factor=2**(args.num_hierarchy_levels-1-h)
                        pred_occs[h]=pred_occs[h][:]+torch.IntTensor([0,y*stride_y//factor,x*stride_x//factor])
                        #remove padding
                        mask_occ=(pred_occs[h][:,0]<(Z//factor))&(pred_occs[h][:,1]<(Y//factor))&(pred_occs[h][:,2]<(X//factor))
                        pred_occs[h]=pred_occs[h][mask_occ]
                        max_z_occ = Z//factor
                        max_y_occ = Y//factor
                        max_x_occ = X//factor
                        # flatten locs
                        pred_occs[h] = pred_occs[h][:, 0] * max_y_occ * max_x_occ + pred_occs[h][:, 1] * max_x_occ + pred_occs[h][:, 2]
                        # end append to start
                        locs = pred_occs[h].view(-1).cpu().numpy()
                        start_occ[h][locs] = locs
                        point_num_occ[h][locs] = 1
                        # end = np.zeros((128//factor)*(Y//factor)*(X//factor))
                        # end[locs] = output_sdf[1].view(-1).cpu().numpy()
                        # # point_mean.append(end)
                        # end = end[:, np.newaxis]
                        # print('ready for append')
                        # start = np.append(start, end, axis=1)
                        # start = start.sum(axis=1)[:, np.newaxis]

                output_sdf[0]=output_sdf[0][:]+torch.IntTensor([0,y*stride_y,x*stride_x,0])
                #remove padding
                #dims = sample['orig_dims'][0]
                #orig_dims = sample['orig_dims']
                mask = (output_sdf[0][:, 0] <Z) & (output_sdf[0][:, 1] < Y) & (
                            output_sdf[0][:, 2] < X)
                output_sdf[0] = output_sdf[0][mask]
                output_sdf[1] = output_sdf[1][mask]
                # print('model out')
                # print('max', torch.max(output_sdf[0][:, 0]), torch.max(output_sdf[0][:, 1]), torch.max(output_sdf[0][:, 2]),
                #       torch.max(output_sdf[0][:, 3]))
                # print('max11occ', torch.max(output_occs[0][0][:, 0]), torch.max(output_occs[0][0][:, 1]), torch.max(output_occs[0][0][:, 2]),
                #       torch.max(output_occs[0][0][:, 3]))
                # print('len',output_sdf[0].shape,output_occs[0].shape)
                max_z=Z
                max_y=Y
                max_x=X
                #flatten locs
                output_sdf[0]=output_sdf[0][:,0]*max_y*max_x+output_sdf[0][:,1]*max_x+output_sdf[0][:,2]
                # end append to start
                locs= output_sdf[0].view(-1).cpu().numpy()

                point_num[locs]+=1
                end = np.zeros(128 * Y * X)
                end[locs]=output_sdf[1].view(-1).cpu().numpy()
                #point_mean.append(end)
                end = end[:, np.newaxis]
                # print('ready for append')
                start=np.append(start,end,axis=1)
                start = start.sum(axis=1)[:,np.newaxis]

                #print('---')
    start=np.squeeze(start)
    #non_zero_mean
    #start1=np.array(point_mean)
    #start_tensor=torch.Tensor(point_mean)
    #start=start1.transpose()
    #start_tensor=start_tensor.T
    #sum1=np.sum(start1)
    #exist=(start!=0)
    #exist=(start_tensor!=0)
    # print('before sum')
    #start_sum=start.sum(axis=1)
    #non_zero=exist.sum(axis=1)
    #start_sum=(torch.sum(start_tensor,1)).numpy()
    #non_zero=(torch.sum(exist,1)).numpy()
    # print('sum')
    #mask_loc=(non_zero!=0)
    mask_loc=(point_num!=0)
    # print('divide')
    mean_feature=start[mask_loc]/point_num[mask_loc]
    #get locs
    mean_loc=np.zeros(128 * Y * X)
    mean_loc[mask_loc]=1
    mean_loc=np.nonzero(mean_loc)
    mean_loc=np.array(mean_loc,dtype=int)
    mean_loc=np.squeeze(mean_loc)
    if False:#args.compute_pred_occs:
        mean_loc_occ=[None] * args.num_hierarchy_levels
        for h in range(args.num_hierarchy_levels):
            factor = 2 ** (args.num_hierarchy_levels - 1 - h)
            #mask_loc=(start_occ[h]!=0)
            mean_loc_occ[h]=np.nonzero(start_occ[h])
            mean_loc_occ[h] = np.array(mean_loc_occ[h], dtype=int)
            mean_loc_occ[h] = np.squeeze(mean_loc_occ[h])
            #print('mean_loc_occ',mean_loc_occ)
            z_loc_occ = mean_loc_occ[h] // ((max_y//factor) * (max_x//factor)).numpy()
            print('z_loc_occ',z_loc_occ)
            y_loc_occ = mean_loc_occ[h] % ((max_y//factor) * (max_x//factor)).numpy() // (max_x//factor).numpy()
            x_loc_occ = mean_loc_occ[h] % (max_x//factor).numpy()
            zyx = np.array([z_loc_occ, y_loc_occ, x_loc_occ])
            zyx=zyx.transpose()
            mean_loc_occ[h]=zyx
            mean_loc_occ[h]=mean_loc_occ[h][np.newaxis,:]
            print('mean_loc_occ[h]',mean_loc_occ[h].shape,mean_loc_occ[h])
    # print('mean_loc',mean_loc)
    # print('(max_y*max_x).numpy()',(max_y*max_x).numpy())
    # print('divide1')
    z_loc=mean_loc//(max_y*max_x).numpy()
    #z_loc=z_loc[:,np.newaxis]
    #print('z_loc',z_loc)
    y_loc=mean_loc%(max_y*max_x).numpy()//max_x.numpy()#[:,np.newaxis]
    #print('y_loc', y_loc)
    x_loc=mean_loc%max_x.numpy()#[:,np.newaxis]
    #print('x_loc', x_loc)
    b_loc=np.zeros(len(x_loc),dtype=np.int)#[:,np.newaxis]
    #print('b_loc', b_loc)
    ##zy = np.append(z_loc, y_loc, axis=1)
    #print('zy', zy)
    ##zyx = np.append(zy, x_loc, axis=1)
    ##zyxb = np.append(zyx, b_loc, axis=1)
    #zyxb=torch.from_numpy(zyxb)
    #mean_feature=torch.from_numpy(mean_feature)
    zyxb=np.array([z_loc,y_loc,x_loc,b_loc])
    zyxb = zyxb.transpose()
    # print('--')
    # print([zyxb,mean_feature],output_sdf)
    #return[zyxb,mean_feature],mean_loc_occ
    return[zyxb,mean_feature]
def test(loss_weights, dataloader, output_vis, num_to_vis,id):
    f = open(args.csv_path, "w")
    model.eval()
    # a=torch.Tensor([[[1,2,3],[1,2,3],[4,5,6],[4,5,6],[7,8,9]]])
    # b=torch.Tensor([[[1,2,3],[1,2,3],[1,2,3]]])
    # print('shape',a.shape,b.shape)
    # loss, loss_normals = chamfer_distance(a, b, point_reduction='mean')
    # print('losstest',loss)
    num_vis = 0
    num_batches = len(dataloader)
    #print('0',num_batches)
    with torch.no_grad():
        #print('1')
        for t, sample in enumerate(dataloader):
            print(t+id)
            #print('num_to_vis',num_to_vis)
            #print('len',len(dataloader))
            #print('2')
            inputs = sample['input']
            #get target
            target=sample['sdf']
            input_dim = np.array([128, 64, 64])
            print(target.shape)
            print('target max min',target.max(),target.min(),target.shape)
            target=data_util.preprocess_sdf_np(target, 3)
            #target=target.squeeze()
            print('target max min', target.max(), target.min(), target.shape)

            #input_dim = np.array(sample['sdf'].shape[2:])
            sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d)    ' % (num_vis, num_to_vis, sample['name'], input_dim[0], input_dim[1], input_dim[2]))
            sys.stdout.flush()
            hierarchy_factor = pow(2, args.num_hierarchy_levels-1)
            model.update_sizes(input_dim, input_dim // hierarchy_factor)
            orig_dims = sample['orig_dims']
            inputs[1] = inputs[1].cuda()
            output_sdf = cut_pro_block_scence(inputs, orig_dims, loss_weights, sample)
            output_sdf = [torch.from_numpy(output_sdf[0]), (torch.from_numpy(output_sdf[1])).float()]
            # try:
            #     if not args.cpu:
            #         inputs[1] = inputs[1].cuda()
            #         print('in model')
            #         output_sdf, output_occs = model(inputs, loss_weights)
            #         print('out model')
            # except:
            #     print('exception at %s' % sample['name'])
            #     gc.collect()
            #     continue

            # remove padding
            dims = sample['orig_dims'][0]
            # print('dims',dims)
            #mask = (output_sdf[0][:,0] < dims[0]) & (output_sdf[0][:,1] < dims[1]) & (output_sdf[0][:,2] < dims[2])
            mask = (output_sdf[0][:,0] < target.shape[2]) & (output_sdf[0][:,1] < target.shape[3]) & (output_sdf[0][:,2] < target.shape[4])
            output_sdf[0] = output_sdf[0][mask]
            output_sdf[1] = output_sdf[1][mask]
            # print('output_sdf[0] ',output_sdf[0] )
            mask = (inputs[0][:,0] < dims[0]) & (inputs[0][:,1] < dims[1]) & (inputs[0][:,2] < dims[2])
            inputs[0] = inputs[0][mask]
            inputs[1] = inputs[1][mask]
            vis_pred_sdf = [None]
            if len(output_sdf[0]) > 0:
                vis_pred_sdf[0] = [output_sdf[0].cpu().numpy(), output_sdf[1].squeeze().cpu().numpy()]
                print('beforeshape', vis_pred_sdf[0][0].shape, vis_pred_sdf[0][1].shape)
            # print(vis_pred_sdf)
            # print(sample['name'],target.shape)
            #target=target[:,:,0:min(128,target.shape[2]),:,:]
            #unobserved space
            known = sample['known'].cpu().numpy()
            mask2 = known < UNK_THRESH
            mask3=known>=UNK_THRESH
            vis_pred_sdf_dense = data_util.sparse_to_dense_np(vis_pred_sdf[0][0], vis_pred_sdf[0][1][:, np.newaxis],
                                                              target.shape[4], target.shape[3], target.shape[2],
                                                            1000)
            mask4=mask3.squeeze()

            vis_pred_sdf_dense[mask4]=1000
            locs_new, sdf_new = data_util.dense_to_sparse_np(vis_pred_sdf_dense, 100)
            locs_new=locs_new.T
            locs_new1=np.zeros([locs_new.shape[0],locs_new.shape[1]+1],dtype=np.int16)
            locs_new1[:,:3]=locs_new
            vis_pred_sdf[0]=[locs_new1, sdf_new]

            mask3=torch.from_numpy(mask3)
            target[mask3]=3.1
            # print('aftershape', vis_pred_sdf[0][0].shape, vis_pred_sdf[0][1].shape)

            #target=target[mask_target]
            pred_mesh, target_mesh = data_util.sdf2mesh(
                 sample['name'],
                inputs, target.cpu(),
                vis_pred_sdf, sample['world2grid'].numpy(), args.truncation, 10)
            # vec color face
            pred_mesh[0][0] = pred_mesh[0][0][np.newaxis, :]
            target_mesh[0][0] = target_mesh[0][0][np.newaxis, :]

            # print('shapes',pred_mesh[0][0].shape,target_mesh[0][0].shape,min(30000,np.array(target_mesh[0][0].shape[1])))
            #sample
            linespace1=np.random.randint(0,pred_mesh[0][0].shape[1]-1,min(30000,np.array(target_mesh[0][0].shape[1])))

            pred_mesh[0][0]=pred_mesh[0][0][:,linespace1,:]


            linespace2 = np.random.randint(0, target_mesh[0][0].shape[1] - 1, min(30000, np.array(target_mesh[0][0].shape[1])))
            target_mesh = np.array(target_mesh)
            target_mesh[0][0]=target_mesh[0][0][:,linespace2,:]
            # print('pred_mesh', len(pred_mesh), pred_mesh[0][0].shape)
            # print('pred_mesh', pred_mesh[0][0][0][:,0].max(),pred_mesh[0][0][0][:,0].min(),pred_mesh[0][0][0][:,1].max(),pred_mesh[0][0][0][:,1].min(),pred_mesh[0][0][0][:,2].max(),pred_mesh[0][0][0][:,2].min())
            # print('target_mesh', len(target_mesh), target_mesh[0][0].shape)
            # print('target_mesh', target_mesh[0][0][0][:, 0].max(), target_mesh[0][0][0][:, 0].min(),
            #       target_mesh[0][0][0][:, 1].max(), target_mesh[0][0][0][:, 1].min(), target_mesh[0][0][0][:, 2].max(),
            #       target_mesh[0][0][0][:, 2].min())
            # print(pred_mesh[0][0])
            # print(target_mesh[0][0])

            #mc.save_mesh(pred_mesh[0][0], pred_mesh[0][1], pred_mesh[0][2], 'pred.obj')
            #mc.save_mesh(target_mesh[0][0],target_mesh[0][1], target_mesh[0][2], 'target.obj')
            #2cm to 1m
            pred_mesh[0][0]=pred_mesh[0][0]/50.0
            target_mesh[0][0]=target_mesh[0][0]/50.0
            # print(pred_mesh[0][0])
            # print(target_mesh[0][0])
            loss, loss_normals = chamfer_distance(pred_mesh[0][0], target_mesh[0][0],point_reduction = 'mean')
            #loss, loss_normals = chamfer_distance(pred_mesh[0][0], target_mesh[0][0],x_lengths=pred_mesh[0][0].shape[1],y_lengths=target_mesh[0][0].shape[1],point_reduction = 'mean')
            print('loss',loss, loss_normals)

            # # get dense prediction
            # vis_pred_sdf_dense=data_util.sparse_to_dense_np(vis_pred_sdf[0][0], vis_pred_sdf[0][1][:, np.newaxis], target.shape[2], target.shape[1],target.shape[0], -float('inf'))
            # print('vis_pred_sdf_dense max min', vis_pred_sdf_dense.max(), vis_pred_sdf_dense.min(), vis_pred_sdf_dense.shape)
            # vis_pred_sdf_dense = data_util.preprocess_sdf_np(vis_pred_sdf_dense, 3)
            # #compute entire volume
            # loss_entire_volume=np.abs(np.abs(target)-np.abs(vis_pred_sdf_dense)).mean()
            # print('loss_entire_volume',loss_entire_volume)
            #
            # #conpute unobserved space
            # known = sample['known']
            # known=known.squeeze()
            # print('known.shape',known.shape)
            # mask1 = known >= UNK_THRESH
            # loss_unobserved=np.abs(np.abs(target)-np.abs(vis_pred_sdf_dense))[mask1].mean()
            # print('loss_unobserved', loss_unobserved)
            #
            # #near target space
            # mask2 = known < UNK_THRESH
            # mask_neartarget=np.abs(target)<=1
            # loss_near_target = np.abs(np.abs(target) - np.abs(vis_pred_sdf_dense))[mask2 & mask_neartarget].mean()
            # print('loss_near_target', loss_near_target)
            #
            # # near prediction
            # mask_nearpred=np.abs(vis_pred_sdf_dense)<=1
            # loss_near_pred=np.abs(np.abs(target) - np.abs(vis_pred_sdf_dense))[mask_nearpred].mean()
            # print('loss_near_pred', loss_near_pred)

            value=''
            value+=str(np.array(t+id))
            value+=','
            value+=str(np.array(loss))
            # value += ','
            # value+=str(np.array(loss_unobserved))
            # value += ','
            # value+=str(np.array(loss_near_target))
            # value += ','
            # value+=str(np.array(loss_near_pred))
            print('value',value)
            print(value,file=f)
            # with open(args.csv_path,mode="w",encoding="utf-8-sig",newline="") as f:
            #     writer=csv.DictWriter(f, header_list)

            inputs = [inputs[0].numpy(), inputs[1].cpu().numpy()]
            #data_util.save_predictions(output_vis, sample['name'], inputs, None, None, vis_pred_sdf, None, sample['world2grid'], args.truncation)
            num_vis += 1
            print(num_vis,num_to_vis)
            if num_vis >= num_to_vis:
                break
            #print('---')
    sys.stdout.write('\n')


def main():
    # data files
    id=358
    test_files, _ = data_util.get_train_files(args.input_data_path, args.test_file_list, '')
    test_files=test_files[id:]
    # if len(test_files) > args.max_to_vis:
    #     test_files = test_files[0:73]
    # else:
    #     args.max_to_vis = len(test_files)
    # random.seed(42)
    # random.shuffle(test_files)
    #print('#test files = ', len(test_files),test_files)
    test_dataset = scene_dataloader.SceneDataset(test_files, args.input_dim, args.truncation, args.num_hierarchy_levels, args.max_input_height, 0, args.target_data_path)
    print('len(test_dataset)',len(test_dataset))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=scene_dataloader.collate)

    if os.path.exists(args.output):
        input('warning: output dir %s exists, press key to overwrite and continue' % args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # start testing
    print('starting evaluating...')
    loss_weights = np.ones(args.num_hierarchy_levels+1, dtype=np.float32)
    test(loss_weights, test_dataloader, args.output, args.max_to_vis,id)


if __name__ == '__main__':
    main()


