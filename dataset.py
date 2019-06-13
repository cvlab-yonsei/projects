import os
import scipy
import scipy.misc
#import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from python_flo import readFLO
from read_depth import depth_read

class Dataset:
    
    def __init__(self, video_split, batch_size):

        self.vs = video_split
        self.bs = batch_size
        
        #< ----------------------- Base ----------------------- >#
        self.KITTI_path = os.path.abspath('/mnt/sdb/datasets/KITTI_raw_data/')
        
        #< ----------------------- Detail ----------------------- >#
        self.rgb_path = os.path.join(self.KITTI_path, 'RGB')
        self.opt_path = os.path.join(self.KITTI_path, 'optflw_dis_inv')
        self.dep_path = os.path.join(self.KITTI_path, 'depth')

    def get_list(self, train_list_path=None, val_list_path=None, test_list_path=None):

        train_list = []
        val_list   = []
        test_list  = []
        
        if train_list_path != None:
            with open(train_list_path, 'r') as f:
                train_list =  f.read().splitlines()
            
        if val_list_path != None:
            with open(val_list_path, 'r') as f:
                val_list =  f.read().splitlines()
            
        if test_list_path != None:
            with open(test_list_path, 'r') as f:
                test_list =  f.read().splitlines()
            
        return train_list, val_list, test_list

    def get_batch(self, folder_list, fr_num):
        
        rgb_set = []
        opt_set = []
        dep_set = []
     
        for b in range( self.bs ):
            
            video = folder_list[b].split()[0].split('image')
            video_num = int( folder_list[b].split()[1] )
            
            rgb_path = os.path.join( self.rgb_path, video[0], 'image'+video[1], 'data' )
            opt_path = os.path.join( self.opt_path, video[0], 'image'+video[1], 'data' )
            dep_path = os.path.join( self.dep_path, video[0], 'proj_depth/groundtruth/image'+video[1] )
                        
            rgb = os.path.join( rgb_path, '%010d.png'%( self.vs*video_num + fr_num[b] + 5 ) )
            rgb_data = scipy.misc.imread( rgb )
            rgb_set.append( rgb_data )

            opt = os.path.join( opt_path, '%010d.flo'%( self.vs*video_num + fr_num[b] + 5 ) )
            opt_data = readFLO( opt )
            opt_set.append( opt_data[:,:,0:2] )

            dep = os.path.join(dep_path, '%010d.png'%( self.vs*video_num + fr_num[b] + 5 ) )
            dep_data = depth_read( dep )
            dep_set.append(dep_data)

        return rgb_set, opt_set, dep_set
    
    def data_prepro(self, rgb_set, opt_set, dep_set, mode, height, width, crop_h=None, crop_w=None):
        
        new_rgb_set = np.zeros( [self.bs, height, width, 3] )
        new_opt_set = np.zeros( [self.bs, height, width, 2] )
        new_dep_set = np.zeros( [self.bs, height, width] )

        for b in range( self.bs ):

            in_h = rgb_set[b].shape[0]
            in_w = rgb_set[b].shape[1]
            
            # Read the data
            rgb_data = rgb_set[b]
            opt_data = opt_set[b]
            dep_data = dep_set[b]
            
            if mode == 'training':
                # Random Cropping
                rgb_data = rgb_data[ int(crop_h[b]):int(crop_h[b])+height, int(crop_w[b]):int(crop_w[b])+width ]
                opt_data = opt_data[ int(crop_h[b]):int(crop_h[b])+height, int(crop_w[b]):int(crop_w[b])+width ]
                dep_data = dep_data[ int(crop_h[b]):int(crop_h[b])+height, int(crop_w[b]):int(crop_w[b])+width ]

            
            if mode == 'validation':
                # Bottom Right Cropping
                rgb_data = rgb_data[ in_h-height:in_h, in_w-width:in_w]
                opt_data = opt_data[ in_h-height:in_h, in_w-width:in_w]
                dep_data = dep_data[ in_h-height:in_h, in_w-width:in_w]
            
            new_rgb_set[b] = rgb_data
            new_opt_set[b] = opt_data
            new_dep_set[b] = dep_data
            
        return new_rgb_set, new_opt_set, new_dep_set

        