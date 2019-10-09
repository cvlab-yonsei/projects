import os
import time
import cv2
import png
import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import dataset
from read_depth import depth_read
from python_flo import readFLO
from opt import opt

sess = tf.InteractiveSession()
ds_class = dataset.Dataset( )
test_path = './folder/eigen_test_tmpr.txt'
_, _, test_list = ds_class.get_list( test_list_path=test_path )
print( 'Total test fol : ', len(test_list) )

GT_path = opt.data_path + '/eigen_test_gt_from_lidar'
RGB_path = opt.data_path + '/RGB'
OPT_path = opt.data_path + '/optflw_dis_inv'
save_path = './results/flowGRU_tmpr/'

def scale(pred_path, gt_path):
    
    pred = depth_read(pred_path)
    gt   = depth_read(gt_path)
    
    mask = (gt != -1)
    gt[np.invert(mask)] = np.nan

    gt_median   = np.nanmedian(gt)
    pred_median = np.nanmedian(pred)
    
    pred *= gt_median / pred_median

    return pred
    
def bound(pred, min_dep=1e-3, max_dep=50):
    
    pred[pred<min_dep] = min_dep
    pred[pred>max_dep] = max_dep
    
    return pred

def idx(height, width):

    idx_x = [ [ i for i in range(width)] for j in range(height) ]
    idx_y = []

    for j in range(height):
        idx_y.append([ j for i in range(width) ])

    idx = np.zeros((height, width, 2))
    idx[:, :, 0] = idx_x
    idx[:, :, 1] = idx_y

    idx_tensor = tf.constant(idx, dtype=tf.float32)

    return idx_tensor

def warp(pst, back_flw):

    h, w, _ = back_flw.get_shape().as_list()
    index = idx(h, w)

    a_1 = tf.zeros_like(pst)
    a_2 = tf.zeros_like(pst)
    a_3 = tf.zeros_like(pst)
    a_4 = tf.zeros_like(pst)

    x_n = tf.clip_by_value(index[:,:,0] + back_flw[:,:,0], 0, w-1)
    y_n = tf.clip_by_value(index[:,:,1] + back_flw[:,:,1], 0, h-1)

    x = tf.clip_by_value(tf.floor(index[:,:,0] + back_flw[:,:,0]), 0, w-1)
    y = tf.clip_by_value(tf.floor(index[:,:,1] + back_flw[:,:,1]), 0, h-1)

    x_1 = tf.clip_by_value(x+1, 0, w-1)
    y_1 = tf.clip_by_value(y+1, 0, h-1)

    a_1 = tf.multiply(tf.expand_dims(tf.multiply(1.0-tf.abs(x_n - x), 1.0-tf.abs(y_n - y)), axis=2),
                      tf.gather_nd(pst, tf.to_int32(tf.stack([y,x],axis=2))))
    a_2 = tf.multiply(tf.expand_dims(tf.multiply(1.0-tf.abs(x_n - (x+1)), 1.0-tf.abs(y_n - y)), axis=2),
                      tf.gather_nd(pst, tf.to_int32(tf.stack([y,x_1],axis=2))))
    a_3 = tf.multiply(tf.expand_dims(tf.multiply(1.0-tf.abs(x_n - x), 1.0-tf.abs(y_n - (y+1))), axis=2),
                      tf.gather_nd(pst, tf.to_int32(tf.stack([y_1,x],axis=2))))
    a_4 = tf.multiply(tf.expand_dims(tf.multiply(1.0-tf.abs(x_n - (x+1)), 1.0-tf.abs(y_n - (y+1))), axis=2),
                      tf.gather_nd(pst, tf.to_int32(tf.stack([y_1,x_1],axis=2))))

    return a_1 + a_2 + a_3 + a_4

def pred_warp(prev_pred, backflw):
    
    return tf.map_fn(lambda inputs : warp(inputs[0],inputs[1]), elems=[prev_pred, backflw], dtype=tf.float32)

def no_mask(appr_X, p_appr_X, backflw, alpha=0.5):

    wrpd_p_appr_X = tf.map_fn(lambda inputs : warp(inputs[0],inputs[1]), elems=[p_appr_X, backflw], dtype=tf.float32)

    diff = tf.reduce_mean(tf.abs(appr_X - wrpd_p_appr_X), 3) 
    msk  = tf.exp((-1) * alpha * diff)

    return msk
    
def readFLO(file_path):
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return
        
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            #print('Reading %d x %d flo file' % (h, w))
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
            return data2D
        
def first_order(PRED_path, SAVE_path):

    total_error = 0.
    total_error1 = 0.
    total_error2 = 0.
    total_error3 = 0.
    cnt = 0
    
    height = 376
    width  = 1242
    
    p_appr_X = tf.placeholder(tf.float32, [1, height, width, 3])
    appr_X   = tf.placeholder(tf.float32, [1, height, width, 3])
    tmpr_X   = tf.placeholder(tf.float32, [1, height, width, 2])
    p_dep    = tf.placeholder(tf.float32, [1, height, width])

    mask = tf.squeeze(no_mask(appr_X, p_appr_X, tmpr_X, alpha=0.5)) > 0.05
    wrpd_prev_pred = tf.squeeze(pred_warp(tf.expand_dims(p_dep, 3), tmpr_X))

    for fol in range(len(test_list)):

        video_split = len(next(os.walk(opt.data_path+'/RGB/%s%s/'%(test_list[fol], 'image_02/data')))[2])

        for frame_num in range(2, video_split-1):
            
            #< ----------------------- Data Path ----------------------- >#
            full = test_list[fol].split()[0]
            split = test_list[fol].split()[0].split('/')

            prev_pred_path = os.path.join(PRED_path, full, '%010d.png'%(frame_num-1))
            pred_path      = os.path.join(PRED_path, full, '%010d.png'%frame_num)
                
            prev_gt_path  = os.path.join(GT_path, full, 'image_02', '%010d.png'%(frame_num-1))
            gt_path       = os.path.join(GT_path, full, 'image_02', '%010d.png'%frame_num)
            prev_rgb_path = os.path.join(RGB_path, full, 'image_02', 'data', '%010d.png'%(frame_num-1))
            rgb_path      = os.path.join(RGB_path, full, 'image_02', 'data', '%010d.png'%frame_num)
            flw_path      = os.path.join(OPT_path, full, 'image_02', 'data', '%010d.flo'%frame_num)

            #< ----------------------- Data Load  ----------------------- >#
            prev_pred = scale(prev_pred_path, prev_gt_path)
            pred = scale(pred_path, gt_path)
            prev_rgb = scipy.misc.imread(prev_rgb_path)
            rgb = scipy.misc.imread(rgb_path)
            flw = readFLO(flw_path)

            prev_pred = cv2.resize(np.squeeze(prev_pred), (width, height), interpolation=cv2.INTER_AREA)
            pred = cv2.resize(np.squeeze(pred), (width, height), interpolation=cv2.INTER_AREA)
            prev_rgb = cv2.resize(np.squeeze(prev_rgb), (width, height), interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(np.squeeze(rgb), (width, height), interpolation=cv2.INTER_AREA)
            flw = cv2.resize(np.squeeze(flw), (width, height), interpolation=cv2.INTER_AREA)

            M = mask.eval(feed_dict={p_appr_X: [prev_rgb], appr_X: [rgb], tmpr_X: [flw]})
            
            Wrpd = wrpd_prev_pred.eval(feed_dict={p_dep: [prev_pred], tmpr_X: [flw]})
            pred_diff = M * np.abs(Wrpd - pred)
            
            masked_mean_error = np.sum(pred_diff) / np.sum(M)
            total_error += masked_mean_error
            
            error1 = pred_diff < 1
            masked_mean_error1 = np.mean(error1)
            total_error1 += masked_mean_error1
            
            error2 = pred_diff < 2
            masked_mean_error2 = np.mean(error2)
            total_error2 += masked_mean_error2
            
            error3 = pred_diff < 3
            masked_mean_error3 = np.mean(error3)
            total_error3 += masked_mean_error3
            
            cnt += 1

            if not os.path.exists(os.path.join(SAVE_path, split[0])):
                os.makedirs(os.path.join(SAVE_path, split[0]))
            if not os.path.exists(os.path.join(SAVE_path, split[0], split[1])):
                os.makedirs(os.path.join(SAVE_path, split[0], split[1]))

            with open(os.path.join(SAVE_path, split[0], split[1], 'log.txt'), 'a') as f:
                f.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(masked_mean_error, masked_mean_error1, masked_mean_error2, masked_mean_error3))
                
    return total_error / cnt, total_error1/cnt , total_error2/cnt , total_error3/cnt

if __name__ == '__main__':
    
    error, error1, error2, error3 = first_order('./results/flowGRU', save_path)

    with open(os.path.join(save_path, 'total.txt'), 'a') as f:
        f.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(error, error1, error2, error3))
