import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from os import path
import scipy.io as sio
import cv2
import shutil
import sys

val_list_txt = open('.../val_list.txt','r')
val_list_txt_readlines = val_list_txt.readlines() 
long_side = 473
result_without_depth = open('.../result_RGB.txt','r')
result_without_depth_readlines = result_without_depth.readlines()
result_with_depth = open('.../result_RGB_and_depth.txt','r')
result_with_depth_readlines = result_with_depth.readlines()
fuse_accuracy_array = np.zeros((1817),dtype=float)

for val_item in val_list_txt_readlines:
    cmap_item = val_item.split(' ')[0].split('/')[-1].split('\n')[0] 
    mat = sio.loadmat('.../dump_result_RGB_depth/' + cmap_item[0:-4] + '_blob_0.mat')
    mat = mat['data']
    mat = mat[:,:,0,0]
    ref_img = cv2.imread('.../JPEGImages/'+cmap_item)
    single_ch_input_img = mat
    single_ch_input_img = single_ch_input_img[0:ref_img.shape[1], 0:ref_img.shape[0]]
    single_ch_input_img_height_width_inverse = cv2.rotate(single_ch_input_img,rotateCode=cv2.ROTATE_90_CLOCKWISE)
    single_ch_input_img_height_width_inverse = cv2.flip(single_ch_input_img_height_width_inverse, flipCode=1)
    single_ch_input_img_height_width_inverse[np.where(single_ch_input_img_height_width_inverse > 0)] = 1
    depth_result_mask = single_ch_input_img_height_width_inverse

    mat = sio.loadmat('.../dump_result_RGB/' + cmap_item[0:-4] + '_blob_0.mat')
    mat = mat['data']
    mat = mat[:,:,0,0]
    ref_img = cv2.imread('.../JPEGImages/'+cmap_item)
    single_ch_input_img = mat
    single_ch_input_img = single_ch_input_img[0:ref_img.shape[1], 0:ref_img.shape[0]]
    single_ch_input_img_height_width_inverse = cv2.rotate(single_ch_input_img,rotateCode=cv2.ROTATE_90_CLOCKWISE)
    single_ch_input_img_height_width_inverse = cv2.flip(single_ch_input_img_height_width_inverse, flipCode=1)
    single_ch_input_img_height_width_inverse[np.where(single_ch_input_img_height_width_inverse > 0)] = 1
    ori_result_mask = single_ch_input_img_height_width_inverse

    temporary_gt = cv2.imread('.../1class_segmentation_result/'+cmap_item[0:-4] + '.png')
    temporary_gt = temporary_gt[:,:,0]
    temporary_gt[np.where(temporary_gt > 0)] = 1 

    depth_gt_joint_array = depth_result_mask*temporary_gt
    depth_gt_union_array = depth_result_mask+temporary_gt
    depth_gt_union_array[np.where(depth_gt_union_array>0)] = 1
    if float(np.sum(depth_gt_union_array)) > 0:
       depth_miou = float(np.sum(depth_gt_joint_array))/float(np.sum(depth_gt_union_array))
    else:
       depth_miou = 0.0 

    ori_gt_joint_array = ori_result_mask*temporary_gt
    ori_gt_union_array = ori_result_mask+temporary_gt
    ori_gt_union_array[np.where(ori_gt_union_array>0)] = 1
    if float(np.sum(ori_gt_union_array)) > 0:
       ori_miou = float(np.sum(ori_gt_joint_array))/float(np.sum(ori_gt_union_array))
    else:
       ori_miou = 0.0 

    if ori_miou < depth_miou: 
       fuse_accuracy_array[val_list_txt_readlines.index(val_item)] = float(result_with_depth_readlines[val_list_txt_readlines.index(val_item)*3+2].split(' ')[-1].split('\n')[0])
    else: 
       fuse_accuracy_array[val_list_txt_readlines.index(val_item)] = float(result_without_depth_readlines[val_list_txt_readlines.index(val_item)*3+2].split(' ')[-1].split('\n')[0])

print(str(np.average(fuse_accuracy_array)))






