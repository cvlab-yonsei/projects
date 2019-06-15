import os
import scipy
import scipy.misc
import cv2
import numpy as np
import matplotlib.pyplot as plt

from python_flo import readFLO
from read_depth import depth_read

from opt import opt

class Dataset:
    def __init__(self):
        self.rgb_path = opt.data_path + '/RGB'
        self.opf_path = opt.data_path + '/optflw_dis_inv'
        self.dep_path = opt.data_path + '/depth'

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

    def get_batch(self, folder_list, fr_num,):
        
        rgb_set = []
        opf_set = []
        dep_set = []
     
        for b in range(opt.bs):
            
            video = folder_list[b].split()[0].split('image')
            video_num = int(folder_list[b].split()[1])
            
            rgb_path = os.path.join(self.rgb_path, video[0], 'image'+video[1], 'data')
            opf_path = os.path.join(self.opf_path, video[0], 'image'+video[1], 'data')
            dep_path = os.path.join(self.dep_path, video[0], 'proj_depth/groundtruth/image'+video[1])
                        
            rgb = os.path.join(rgb_path, '%010d.png'%(opt.video_split*video_num + fr_num[b] + 5))
            rgb_data = scipy.misc.imread(rgb)
            rgb_set.append(rgb_data)

            opf = os.path.join(opf_path, '%010d.flo'%(opt.video_split*video_num + fr_num[b] + 5))
            opf_data = readFLO(opf)
            opf_set.append(opf_data[:,:,0:2])

            dep = os.path.join(dep_path, '%010d.png'%(opt.video_split*video_num + fr_num[b] + 5))
            dep_data = depth_read(dep)
            dep_set.append(dep_data)
        
        return rgb_set, opf_set, dep_set
    
    def get_batch_test(self, folder_list, frame_num):
        
        rgb_set = []
        opf_set = []
        dsp_set = []
        
        for i in range(opt.bs_test):

            video = folder_list[i].split()[0]
            video_num = len(next(os.walk(os.path.join(self.rgb_path, folder_list[i], 'image_02/data')))[2])
                
            rgb_path  = os.path.join(self.rgb_path , video, 'image_02/data')
            opf_path = os.path.join(self.opf_path, video, 'image_02/data')
                        
            rgb = os.path.join(rgb_path, '%010d.png'%(frame_num))
            rgb_data = scipy.misc.imread(rgb)
            rgb_set.append(rgb_data)

            opf = os.path.join(opf_path, '%010d.flo'%(frame_num))
            opf_data = readFLO(opf)
            opf_set.append(opf_data[:,:,0:2])

        return rgb_set, opf_set
    
    def data_prepro(self, rgb_set, opf_set, dep_set, mode, crop_h=None, crop_w=None):
        
        new_rgb_set = np.zeros([opt.bs, opt.h, opt.w, 3])
        new_opf_set = np.zeros([opt.bs, opt.h, opt.w, 2])
        new_dep_set = np.zeros([opt.bs, opt.h, opt.w])

        for b in range(opt.bs):

            in_h = rgb_set[b].shape[0]
            in_w = rgb_set[b].shape[1]
            
            # Read the data
            rgb_data = rgb_set[b]
            opf_data = opf_set[b]
            dep_data = dep_set[b]
            
            if mode == 'training':
                # Random Cropping
                rgb_data = rgb_data[ int(crop_h[b]):int(crop_h[b])+opt.h, int(crop_w[b]):int(crop_w[b])+opt.w ]
                opf_data = opf_data[ int(crop_h[b]):int(crop_h[b])+opt.h, int(crop_w[b]):int(crop_w[b])+opt.w ]
                dep_data = dep_data[ int(crop_h[b]):int(crop_h[b])+opt.h, int(crop_w[b]):int(crop_w[b])+opt.w ]

            
            if mode == 'validation':
                # Bottom Right Cropping
                rgb_data = rgb_data[ in_h-opt.h:in_h, in_w-opt.w:in_w]
                opf_data = opf_data[ in_h-opt.h:in_h, in_w-opt.w:in_w]
                dep_data = dep_data[ in_h-opt.h:in_h, in_w-opt.w:in_w]
            
            new_rgb_set[b] = rgb_data
            new_opf_set[b] = opf_data
            new_dep_set[b] = dep_data
            
        return new_rgb_set, new_opf_set, new_dep_set
    
    def data_resize(self, rgb_set, opf_set):
    
        new_rgb_set = np.zeros([len(rgb_set), opt.h, opt.w, 3])
        new_opf_set = np.zeros([len(rgb_set), opt.h, opt.w, 2])

        for i in range(len(rgb_set)):
            
            rgb_h = len(rgb_set[i])
            rgb_w = len(rgb_set[i][0])

            # Read the data
            rgb_data = rgb_set[i]
            opf_data = opf_set[i]

            rgb_data = cv2.resize(rgb_data, (opt.w, opt.h), interpolation=cv2.INTER_AREA)
            opf_data = cv2.resize(opf_data, (opt.w, opt.h), interpolation=cv2.INTER_AREA)
            opf_data[:,:,0] = opf_data[:,:,0]*(opt.w/rgb_w)
            opf_data[:,:,1] = opf_data[:,:,1]*(opt.h/rgb_h)
                
            new_rgb_set[i] = (rgb_data)
            new_opf_set[i] = (opf_data)
        
        return new_rgb_set, new_opf_set, rgb_w, rgb_h

        