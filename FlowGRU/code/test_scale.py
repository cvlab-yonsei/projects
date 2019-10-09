import os
import time
import cv2
import png
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import dataset
from read_depth import depth_read
from opt import opt

def write_depth_png(save_file, depth):
    f = open(save_file, 'wb')
    h, w = depth.shape
    w = png.Writer(width=w, height=h, greyscale=True, bitdepth=16)
    
    I = depth*256
    I[depth==0] = 0
    I[depth<0] = 0
    I[depth>65535] = 0
    
    w.write(f, I)
    f.close()
    
if __name__ == '__main__':
    
    ds_class = dataset.Dataset()
    test_path = './folder/eigen_test_first_frame.txt'
    test_list = ds_class.get_list(test_list_path = test_path)[2]
    print('Total test fol : ', len(test_list))

    DEP_path = opt.data_path + '/eigen_test_gt_from_lidar'
    load_path = './results/flonexwGRU/'
    save_path = './results/flowGRU_median/'
    
    for i in range(len(test_list)):
        print(i)

        #< ----------------------- Data Path ----------------------- >#
        full = test_list[i].split()[0]
        split = test_list[i].split()[0].split('/')
        pred_path = os.path.join(load_path, split[0], split[1], '%s.png'%split[4].split('.')[0])
        dep_path  = os.path.join(DEP_path,  split[0], split[1], '%s.png'%split[4].split('.')[0])

        #< ----------------------- Data Load & Normalization ----------------------- >#
        out = depth_read(pred_path)
        dep_data = depth_read(dep_path)

        #< ----------------------- Median Scaling ----------------------- >#
        n = np.sum(dep_data != -1)
        out[dep_data<0] = np.nan
        dep_data[dep_data<0] = np.nan
        net_out = depth_read(pred_path)

        scale  = np.nanmedian(dep_data)/np.nanmedian(out)
        scaled = net_out * scale

        scaled[ scaled <= 1e-3 ] = 1e-3
        scaled[ scaled >= 80   ] = 80

        #< ----------------------- Save ----------------------- >#
        if not os.path.exists(os.path.join(save_path, split[0])) :
            os.makedirs(os.path.join(save_path, split[0]))
        if not os.path.exists(os.path.join(save_path, split[0], split[1])) :
            os.makedirs(os.path.join(save_path, split[0], split[1]))
        if not os.path.exists(os.path.join(save_path, split[0], split[1], 'image_02')) :
            os.makedirs(os.path.join(save_path, split[0], split[1], 'image_02'))
        if not os.path.exists(os.path.join(save_path, split[0], split[1], 'image_02', 'data')) :
            os.makedirs(os.path.join(save_path, split[0], split[1], 'image_02', 'data'))

        write_path = os.path.join(save_path, split[0], split[1], 'image_02/data', '%s.png'%split[4].split('.')[0])
        write_depth_png(write_path, scaled)


    #     #< ----------------------- Visualization Save ----------------------- >#
    #     if not os.path.exists(os.path.join(viz_path, split[0])) :
    #         os.makedirs(os.path.join(viz_path, split[0]))
    #     if not os.path.exists(os.path.join(viz_path, split[0], split[1])) :
    #         os.makedirs(os.path.join(viz_path, split[0], split[1]))
    #     write_viz_path = os.path.join(viz_path, split[0], split[1], '%s.png'%split[4].split('.')[0])
    #     plt.imsave(write_viz_path, scaled, vmin=0, vmax=80, cmap='plasma')

    