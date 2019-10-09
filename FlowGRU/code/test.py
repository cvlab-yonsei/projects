import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import time
import cv2
import png
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import dataset
import model
from opt import opt

def normal(img):
    mean, var = tf.nn.moments(img, axes=[0,1,2])
    return (img - mean) / tf.sqrt(var + 0.0001)

def normalize(imgs):
    return tf.map_fn(lambda inputs : normal(inputs[0]), elems=[imgs], dtype=tf.float32)

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
    
    #< ----------------------- Placeholders ----------------------- >#
    appr_X   = tf.placeholder(tf.float32, [1, opt.h, opt.w, 3])
    tmpr_X   = tf.placeholder(tf.float32, [1, opt.h, opt.w, 2])
    p_appr_X = tf.placeholder(tf.float32, [1, opt.h, opt.w, 3])
    p_tmpr_X = tf.placeholder(tf.float32, [1, opt.h, opt.w, 2])
    h_prev   = tf.placeholder(tf.float32, [1, int(np.ceil(opt.h/4)), int(np.ceil(opt.w/4)), 64])

    #< ----------------------- Model ----------------------- >#
    md = model.TwoStreamDepthNet(opt.h, opt.w, 1)
    net_out0, frn_out0, frn_out1, frn_out2, gru_out, p_gru_out = \
        md.forward_pass(normalize(appr_X), tmpr_X, normalize(p_appr_X), p_tmpr_X, h_prev)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(opt.weight_path))

    save_path = './results/flowGRU/'
    save_viz_path = './results/flowGRU_viz/'

    ds_class = dataset.Dataset()
    test_path = './folder/eigen_test.txt'
    _, _, test_list = ds_class.get_list(test_list_path=test_path)
    print('Total test fol : ', len(test_list))
    
    for start, end in zip(range(0, len(test_list)+1, 1), range(1, len(test_list)+1, 1)):

        hidden_state = np.zeros((1, int(np.ceil(opt.h/4)), int(np.ceil(opt.w/4)), 64))
        video_split = len(next(os.walk(opt.data_path+'/RGB/%s%s/'%(test_list[start], 'image_02/data')))[2])

        for frame_num in range(1, video_split): 

            p_rgb_set, p_opf_set = ds_class.get_batch_test(test_list[start:end], frame_num)
            p_rgb_set, p_opf_set, _, _ = ds_class.data_resize(p_rgb_set, p_opf_set)

            rgb_set, opf_set = ds_class.get_batch_test(test_list[start:end], frame_num)
            rgb_set, opf_set, rgb_w, rgb_h = ds_class.data_resize(rgb_set, opf_set)

            pred_depth, hidden_state = sess.run([net_out0, p_gru_out], \
                                                feed_dict = {appr_X: rgb_set, tmpr_X: opf_set, 
                                                             p_appr_X: p_rgb_set, p_tmpr_X: p_opf_set,
                                                             h_prev: hidden_state})

            pred_depth_resized = cv2.resize(np.squeeze(pred_depth), (rgb_w, rgb_h), interpolation=cv2.INTER_AREA)

            if not os.path.exists(save_path + test_list[start]):
                os.makedirs(save_path + test_list[start])
            write_depth_png(save_path + test_list[start]+'%010d.png' %(frame_num), pred_depth_resized)

    #         if not os.path.exists(viz_path + test_list[start]):
    #             os.makedirs(viz_path + test_list[start])
    #         plt.imsave(viz_path + test_list[start]+'%010d.png' %(frame_num), pred_depth_resized, vmin=0, vmax=80, cmap='plasma')

        print(test_list[start:end])
