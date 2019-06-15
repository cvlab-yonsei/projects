import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import time
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

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def cal_loss(fol_list):
    cnt  = 0
    Loss = []
    
    for start,end in zip(range(0,len(fol_list)+1,opt.bs),range(opt.bs,len(fol_list)+1, opt.bs)):
        
        sel = np.ones([opt.bs])
        states = np.zeros((opt.bs, int(np.ceil(opt.h/4)), int(np.ceil(opt.w/4)), 64))
        
        rgb_set, opf_set, dep_set = ds_class.get_batch(fol_list[start:end], sel)
        rgb_set, opf_set, dep_set = ds_class.data_prepro(rgb_set, opf_set, dep_set, 'validation')

        p_rgb_set, p_opf_set, p_dep_set = ds_class.get_batch(fol_list[start:end], sel-1)
        p_rgb_set, p_opf_set, p_dep_set = ds_class.data_prepro(p_rgb_set, p_opf_set, p_dep_set, 'validation')

        Loss.append(sess.run(recon_loss, feed_dict={appr_X: rgb_set, tmpr_X: opf_set,
                                                    p_appr_X: p_rgb_set, p_tmpr_X: p_opf_set, 
                                                    h_prev: states, gt_dep: dep_set}))
    return np.mean(Loss)

def sample_ex(epoch):
    sel = np.ones([opt.bs])
    states = np.zeros((opt.bs, int(np.ceil(opt.h/4)), int(np.ceil(opt.w/4)), 64))

    rgb_set, opf_set, dep_set = ds_class.get_batch(val_list, sel)
    rgb_set, opf_set, dep_set = ds_class.data_prepro(rgb_set, opf_set, dep_set, 'validation')
            
    p_rgb_set, p_opf_set, p_dep_set = ds_class.get_batch(val_list, sel-1)
    p_rgb_set, p_opf_set, p_dep_set = ds_class.data_prepro(p_rgb_set, p_opf_set, p_dep_set, 'validation')
    
    ex_train = tf.squeeze(nout).eval(feed_dict={appr_X:rgb_set, tmpr_X:opf_set,
                                                p_appr_X:p_rgb_set, p_tmpr_X:p_opf_set, h_prev: states })
    
    ex_train_path = save_path + 'val_epoch%03d.png'%(epoch)
    output = np.concatenate((ex_train[0], dep_set[0]), axis = 0)
    plt.imsave(ex_train_path, output, cmap='plasma', vmin = 0, vmax = 80)
    
    
    
if __name__ == '__main__':
    
    save_path = './weights/flowGRU/'

    appr_X = tf.placeholder(tf.float32, [opt.bs, opt.h, opt.w, 3])
    tmpr_X = tf.placeholder(tf.float32, [opt.bs, opt.h, opt.w, 2])
    p_appr_X = tf.placeholder(tf.float32, [opt.bs, opt.h, opt.w, 3])
    p_tmpr_X = tf.placeholder(tf.float32, [opt.bs, opt.h, opt.w, 2])
    gt_dep = tf.placeholder(tf.float32, [opt.bs, opt.h, opt.w])
    h_prev = tf.placeholder(tf.float32, [opt.bs, int(np.ceil(opt.h/4)), int(np.ceil(opt.w/4)), 64])

    lr = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(lr, name='Adam')

    with tf.device('/cpu:0'):
        md = model.TwoStreamDepthNet(opt.h, opt.w, opt.bs//2)

    appr_X_rs2   = md.resize(appr_X, 1/2)
    p_appr_X_rs2 = md.resize(p_appr_X, 1/2)
    appr_X_rs4   = md.resize(appr_X, 1/4)
    p_appr_X_rs4 = md.resize(p_appr_X, 1/4)
    gt = tf.expand_dims(gt_dep, axis=3)
    
    s = 0
    num_gpu = 2
    per_gpu = [ opt.bs//2, opt.bs//2 ]

    state_stack = []
    loss_stack = []
    grad_stack = []
    nout_stack = []

    with tf.variable_scope(tf.get_variable_scope()):
        for g in range(num_gpu):
            with tf.device('/gpu:' + str(g)):

                #< ----------------------- Forward Pass ----------------------- >#

                net_out0, frn_out0, frn_out1, frn_out2, gru_out, p_gru_out = md.forward_pass\
                (normalize(appr_X[s:s+per_gpu[g]]), tmpr_X[s:s+per_gpu[g]], normalize(p_appr_X[s:s+per_gpu[g]]), p_tmpr_X[s:s+per_gpu[g]], h_prev[s:s+per_gpu[g]])

                #< ----------------------- Loss ----------------------- >#

                recon_loss   = md.sci_log(net_out0, gt[s:s+per_gpu[g]])
                recon_smooth = tf.reduce_mean(md.smoothness_2nd(net_out0, appr_X[s:s+per_gpu[g]]/255., 10))

                photo_loss0   = md.phot_loss(appr_X[s:s+per_gpu[g]]/255., p_appr_X[s:s+per_gpu[g]]/255., frn_out0)
                photo_smooth0 = tf.reduce_mean(md.smoothness_2nd(frn_out0, appr_X[s:s+per_gpu[g]]/255., 10))
                photo_loss1   = md.phot_loss(appr_X_rs2[s:s+per_gpu[g]]/255., p_appr_X_rs2[s:s+per_gpu[g]]/255., frn_out1)
                photo_smooth1 = tf.reduce_mean(md.smoothness_2nd(frn_out1, appr_X_rs2[s:s+per_gpu[g]]/255., 10))
                photo_loss2   = md.phot_loss(appr_X_rs4[s:s+per_gpu[g]]/255., p_appr_X_rs4[s:s+per_gpu[g]]/255., frn_out2)
                photo_smooth2 = tf.reduce_mean(md.smoothness_2nd(frn_out2, appr_X_rs4[s:s+per_gpu[g]]/255., 10))

                loss = (recon_loss + 0.1*recon_smooth) + 0.05*(photo_loss0 + 0.1*photo_smooth0 + photo_loss1 + 0.1*photo_smooth1 + photo_loss2 + 0.1*photo_smooth2)

                tf.get_variable_scope().reuse_variables()

                grad = optimizer.compute_gradients(loss)

                state_stack.append(p_gru_out)
                loss_stack.append(loss)
                grad_stack.append(grad)
                nout_stack.append(net_out0)

                s += per_gpu[g]

    state = tf.concat(values=state_stack, axis=0)
    nout  = tf.concat(values=nout_stack, axis =0)
    grad = average_gradients(grad_stack)
    train = optimizer.apply_gradients(grads_and_vars= grad)
    
    ds_class = dataset.Dataset()

    train_path = './folder/eigen_train_video.txt'
    val_path = './folder/eigen_val_video.txt'

    train_list, val_list, _ = ds_class.get_list(train_list_path=train_path, val_list_path=val_path)

    print('Total train fol : ', len(train_list))
    print('Total val fol   : ', len(val_list))
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    sess = tf.InteractiveSession(config=config)
    tf.global_variables_initializer().run()

    current_step = 1  
    iteration    = (len(train_list)//opt.bs) * (current_step - 1) + 1

    print(current_step)
    print(iteration)
    
    
    print("Start Training...")

    if not os.path.exists(save_path): os.makedirs(save_path)

    train_epochs = 200
    for epoch in range(current_step, train_epochs+1):

        if epoch < 100: learning_rate = 0.0001
        elif epoch < 140: learning_rate = 0.00005
        elif epoch < 170: learning_rate = 0.000025
        else: learning_rate = 0.0000125

        tic = time.time()

        np.random.shuffle(train_list)
        for start, end in zip(range(0, len(train_list)+1, opt.bs), range(opt.bs, len(train_list)+1, opt.bs)):
            i_tic = time.time()

            sel = np.random.randint(opt.video_split - opt.frame_split - 1, size=opt.bs) + 1
            crop_h   = np.zeros([opt.bs])
            crop_w   = np.zeros([opt.bs])
            hidden_state = np.zeros((opt.bs, int(np.ceil(opt.h/4)), int(np.ceil(opt.w/4)), 64))

            rgb_set, opf_set, dep_set = ds_class.get_batch(train_list[start:end], sel)
            for b in range(opt.bs):
                rgb_h, rgb_w, _ = np.shape(rgb_set[b])
                crop_h[b] = np.random.randint(rgb_h - opt.h)
                crop_w[b] = np.random.randint(rgb_w - opt.w)


            for f in range(opt.frame_split):
                rgb_set, opf_set, dep_set = ds_class.get_batch(train_list[start:end], sel)
                rgb_set, opf_set, dep_set = ds_class.data_prepro(rgb_set, opf_set, dep_set, 'training', crop_h, crop_w)

                p_rgb_set, p_opf_set, p_dep_set = ds_class.get_batch(train_list[start:end], sel-1)
                p_rgb_set, p_opf_set, p_dep_set = ds_class.data_prepro(p_rgb_set, p_opf_set, p_dep_set, 'training', crop_h, crop_w)

                _, hidden_state = sess.run([train, state], feed_dict={appr_X: rgb_set, tmpr_X: opf_set,
                                                                      p_appr_X: p_rgb_set, p_tmpr_X: p_opf_set,
                                                                      gt_dep: dep_set, h_prev: hidden_state, lr: learning_rate})

                sel += 1

            i_toc = time.time()

            if epoch < 10:
                print('epoch%03d'%(epoch),
                      'iteration%07d'%(iteration),
                      '%03.1f%%'%((iteration - ((len(train_list)//opt.bs)*(epoch-1)))/(len(train_list)//opt.bs)*100),
                      'time ', '%02d min' %((i_toc-i_tic)//60),
                      'loss : ', cal_loss(train_list[start:end]))

            else:
                print('epoch%03d'%(epoch),
                      'iteration%07d'%(iteration),
                      '%03.1f%%'%((iteration - ((len(train_list)//opt.bs)*(epoch-1)))/(len(train_list)//opt.bs)*100),
                      'time ', '%02d min' % ((i_toc-i_tic)//60))

            iteration += 1

        toc = time.time()
        train_time = toc - tic

        #< ----------------------- Weight Save ----------------------- >#
        if not os.path.exists(save_path + 'epoch%03d'%(epoch)):
            os.makedirs(save_path + 'epoch%03d'%(epoch))
        saver = tf.train.Saver()
        saver.save(sess, save_path + 'epoch%03d/TwoStreamDepth'%(epoch))

        #< ----------------------- Example Output ----------------------- >#
        sample_ex(epoch)

        #< ----------------------- Loss ----------------------- >#
        val_loss = cal_loss(val_list)

        #< ----------------------- Print ----------------------- >#
        print("Epoch : ", '%03d' % (epoch),
              "val_loss = ", "{:.5f}".format(val_loss),
              "train_time = ", '%d min' % (train_time//60))

        with open(save_path + 'log.txt', 'a') as f:
            f.write('%03d %.5f %d \n'%(epoch, val_loss, train_time//60))