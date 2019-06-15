import tensorflow as tf
import numpy as np
import net

class TwoStreamNet():
    
    def __init__(self, height, width, batch_size):
        
        self.h = height
        self.w = width
        self.bs = batch_size
        
        self.appr_W1_1   = net.Variable([3, 3, 3, 32], name = 'appr_W1_1')
        self.appr_W1_2   = net.Variable([3, 3, 32, 32], name = 'appr_W1_2')
        self.appr_W2_1   = net.Variable([3, 3, 32, 64], name = 'appr_W2_1')
        self.appr_W2_2   = net.Variable([3, 3, 64, 64], name = 'appr_W2_2')
        self.appr_W3_1   = net.Variable([3, 3, 64, 64], name = 'appr_W3_1')
        self.appr_W3_2   = net.Variable([3, 3, 64, 64], name = 'appr_W3_2')
        self.appr_W4_1   = net.Variable([3, 3, 64, 64], name = 'appr_W4_1')
        self.appr_W4_2   = net.Variable([3, 3, 64, 64], name = 'appr_W4_2')
        self.appr_W5_1   = net.Variable([3, 3, 64, 128], name = 'appr_W5_1')
        self.appr_W5_2   = net.Variable([3, 3, 128, 128], name = 'appr_W5_2')
        self.appr_W6_1   = net.Variable([3, 3, 128, 128], name = 'appr_W6_1')
        self.appr_W6_2   = net.Variable([3, 3, 128, 128], name = 'appr_W6_2')
        self.appr_W7_1   = net.Variable([3, 3, 128, 256], name = 'appr_W7_1')
        self.appr_W7_2   = net.Variable([3, 3, 256, 256], name = 'appr_W7_2')
        self.appr_W8     = net.Variable([3, 3, 256, 256], name = 'appr_W8')
        self.appr_Wout   = net.Variable([3, 3, 256, 64], name = 'appr_Wout')
        self.a_b1_1 = net.Variable(name='a_b1_1',  init=tf.zeros([32]))
        self.a_b1_2 = net.Variable(name='a_b1_2',  init=tf.zeros([32]))
        self.a_b2_1 = net.Variable(name='a_b2_1',  init=tf.zeros([64]))
        self.a_b2_2 = net.Variable(name='a_b2_2',  init=tf.zeros([64]))
        self.a_b3_1 = net.Variable(name='a_b3_1',  init=tf.zeros([64]))
        self.a_b3_2 = net.Variable(name='a_b3_2',  init=tf.zeros([64]))
        self.a_b4_1 = net.Variable(name='a_b4_1',  init=tf.zeros([64]))
        self.a_b4_2 = net.Variable(name='a_b4_2',  init=tf.zeros([64]))
        self.a_b5_1 = net.Variable(name='a_b5_1',  init=tf.zeros([128]))
        self.a_b5_2 = net.Variable(name='a_b5_2',  init=tf.zeros([128]))
        self.a_b6_1 = net.Variable(name='a_b6_1',  init=tf.zeros([128]))
        self.a_b6_2 = net.Variable(name='a_b6_2',  init=tf.zeros([128]))
        self.a_b7_1 = net.Variable(name='a_b7_1',  init=tf.zeros([256]))
        self.a_b7_2 = net.Variable(name='a_b7_2',  init=tf.zeros([256]))
        self.a_b8_1 = net.Variable(name='a_b8_1',  init=tf.zeros([256]))
        self.a_bout = net.Variable(name='a_bout',  init=tf.zeros([64]))
        
        self.tmpr_W1_1   = net.Variable([3, 3, 2, 32], name = 'tmpr_W1_1')
        self.tmpr_W1_2   = net.Variable([3, 3, 32, 32], name = 'tmpr_W1_2')
        self.tmpr_W2_1   = net.Variable([3, 3, 32, 64], name = 'tmpr_W2_1')
        self.tmpr_W2_2   = net.Variable([3, 3, 64, 64], name = 'tmpr_W2_2')
        self.tmpr_W3_1   = net.Variable([3, 3, 64, 64], name = 'tmpr_W3_1')
        self.tmpr_W3_2   = net.Variable([3, 3, 64, 64], name = 'tmpr_W3_2')
        self.tmpr_W4_1   = net.Variable([3, 3, 64, 64], name = 'tmpr_W4_1')
        self.tmpr_W4_2   = net.Variable([3, 3, 64, 64], name = 'tmpr_W4_2')
        self.tmpr_W5_1   = net.Variable([3, 3, 64, 128], name = 'tmpr_W5_1')
        self.tmpr_W5_2   = net.Variable([3, 3, 128, 128], name = 'tmpr_W5_2')
        self.tmpr_W6_1   = net.Variable([3, 3, 128, 128], name = 'tmpr_W6_1')
        self.tmpr_W6_2   = net.Variable([3, 3, 128, 128], name = 'tmpr_W6_2')
        self.tmpr_W7_1   = net.Variable([3, 3, 128, 256], name = 'tmpr_W7_1')
        self.tmpr_W7_2   = net.Variable([3, 3, 256, 256], name = 'tmpr_W7_2')
        self.tmpr_W8     = net.Variable([3, 3, 256, 256], name = 'tmpr_W8')
        self.tmpr_Wout   = net.Variable([3, 3, 256, 64], name = 'tmpr_Wout')
        self.t_b1_1 = net.Variable(name='t_b1_1',  init=tf.zeros([32]))
        self.t_b1_2 = net.Variable(name='t_b1_2',  init=tf.zeros([32]))
        self.t_b2_1 = net.Variable(name='t_b2_1',  init=tf.zeros([64]))
        self.t_b2_2 = net.Variable(name='t_b2_2',  init=tf.zeros([64]))
        self.t_b3_1 = net.Variable(name='t_b3_1',  init=tf.zeros([64]))
        self.t_b3_2 = net.Variable(name='t_b3_2',  init=tf.zeros([64]))
        self.t_b4_1 = net.Variable(name='t_b4_1',  init=tf.zeros([64]))
        self.t_b4_2 = net.Variable(name='t_b4_2',  init=tf.zeros([64]))
        self.t_b5_1 = net.Variable(name='t_b5_1',  init=tf.zeros([128]))
        self.t_b5_2 = net.Variable(name='t_b5_2',  init=tf.zeros([128]))
        self.t_b6_1 = net.Variable(name='t_b6_1',  init=tf.zeros([128]))
        self.t_b6_2 = net.Variable(name='t_b6_2',  init=tf.zeros([128]))
        self.t_b7_1 = net.Variable(name='t_b7_1',  init=tf.zeros([256]))
        self.t_b7_2 = net.Variable(name='t_b7_2',  init=tf.zeros([256]))
        self.t_b8_1 = net.Variable(name='t_b8_1',  init=tf.zeros([256]))
        self.t_bout = net.Variable(name='t_bout',  init=tf.zeros([64]))
        
        self.frn_W0  = net.Variable([3, 3, 11, 32], name = 'frn_W0')
        self.frn_C0  = net.Variable([3, 3, 32, 2], name = 'frn_C0')
        self.frn_F0  = net.Variable([3, 3, 4, 2], name = 'frn_F0')
        self.frn_W1  = net.Variable([3, 3, 32, 32], name = 'frn_W1')
        self.frn_C1  = net.Variable([3, 3, 34, 2], name = 'frn_C1')
        self.frn_F1  = net.Variable([3, 3, 4, 2], name = 'frn_F1')
        self.frn_W2  = net.Variable([3, 3, 32, 32], name = 'frn_W2')
        self.frn_C2  = net.Variable([3, 3, 34, 2], name = 'frn_C2')
        self.frn_F2  = net.Variable([3, 3, 4, 2], name = 'frn_F2')
        self.frn_Wb0 = net.Variable(name='frn_Wb0',  init=tf.zeros([32]))
        self.frn_Cb0 = net.Variable(name='frn_Cb0',  init=tf.zeros([2]))
        self.frn_Fb0 = net.Variable(name='frn_Fb0',  init=tf.zeros([2]))
        self.frn_Wb1 = net.Variable(name='frn_Wb1',  init=tf.zeros([32]))
        self.frn_Cb1 = net.Variable(name='frn_Cb1',  init=tf.zeros([2]))
        self.frn_Fb1 = net.Variable(name='frn_Fb1',  init=tf.zeros([2]))
        self.frn_Wb2 = net.Variable(name='frn_Wb2',  init=tf.zeros([32]))
        self.frn_Cb2 = net.Variable(name='frn_Cb2',  init=tf.zeros([2]))
        self.frn_Fb2 = net.Variable(name='frn_Fb2',  init=tf.zeros([2]))
        
        self.dec_W2_1  = net.Variable([3, 3, 32, 64], name = 'dec_W2_1')
        self.dec_W2_2  = net.Variable([3, 3, 96, 32], name = 'dec_W2_2')
        self.dec_W1_1  = net.Variable([3, 3, 16, 32], name = 'dec_W1_1')
        self.dec_W1_2  = net.Variable([3, 3, 16, 16], name = 'dec_W1_2')
        self.dec_Wout  = net.Variable([1, 1, 16, 1], name = 'dec_Wout')
        self.d_b2_1 = net.Variable(name='d_b2_1',  init=tf.zeros([32]))
        self.d_b2_2 = net.Variable(name='d_b2_2',  init=tf.zeros([32]))
        self.d_b1_1 = net.Variable(name='d_b1_1',  init=tf.zeros([16]))
        self.d_b1_2 = net.Variable(name='d_b1_2',  init=tf.zeros([16]))
        
        self.k = tf.Variable(0.5, name='k')
        ConvGRU.__init__(self)
        
    def encoder(self, appr_X, tmpr_X):
        
        print('< ----------------------- AppearanceNet ----------------------- >\n')
        
        self.appr_conv1_1 = net.ReLU(net.Conv(appr_X, self.appr_W1_1, 2, 'appr_conv1_1')+ self.a_b1_1,'appr_relu1_1')
        self.appr_conv1_2 = net.ReLU(net.Conv(self.appr_conv1_1, self.appr_W1_2, 1, 'appr_conv1_2')+ self.a_b1_2,'appr_relu1_2')
        print('appr_conv1_1 : ', self.appr_conv1_1)
        print('appr_conv1_2 : ', self.appr_conv1_2)
        
        self.appr_conv2_1 = net.ReLU(net.Conv(self.appr_conv1_2, self.appr_W2_1, 2, 'appr_conv2_1')+ self.a_b2_1,'appr_relu2_1')
        self.appr_conv2_2 = net.ReLU(net.Conv(self.appr_conv2_1, self.appr_W2_2, 1, 'appr_conv2_2')+ self.a_b2_2,'appr_relu2_2')
        print('appr_conv2_1 : ', self.appr_conv2_1)
        print('appr_conv2_2 : ', self.appr_conv2_2)
        
        self.appr_conv3_1 = net.ReLU(net.AtConv(self.appr_conv2_2, self.appr_W3_1, 2,'appr_conv3_1')+ self.a_b3_1,'appr_relu3_1')
        self.appr_conv3_2 = net.ReLU(net.AtConv(self.appr_conv3_1, self.appr_W3_2, 1,'appr_conv3_2')+ self.a_b3_2,'appr_relu3_2')
        print('appr_conv3_1 : ', self.appr_conv3_1)
        print('appr_conv3_2 : ', self.appr_conv3_2)
        
        self.appr_conv4_1 = net.ReLU(net.AtConv(self.appr_conv3_2, self.appr_W4_1, 4,'appr_conv4_1')+ self.a_b4_1,'appr_relu4_1')
        self.appr_conv4_2 = net.ReLU(net.AtConv(self.appr_conv4_1, self.appr_W4_2, 1,'appr_conv4_2')+ self.a_b4_2,'appr_relu4_2')
        print('appr_conv4_1 : ', self.appr_conv4_1)
        print('appr_conv4_2 : ', self.appr_conv4_2)
        
        self.appr_conv5_1 = net.ReLU(net.AtConv(self.appr_conv4_2, self.appr_W5_1, 8,'appr_conv5_1')+ self.a_b5_1,'appr_relu5_1')
        self.appr_conv5_2 = net.ReLU(net.AtConv(self.appr_conv5_1, self.appr_W5_2, 1,'appr_conv5_2')+ self.a_b5_2,'appr_relu5_2')
        print('appr_conv5_1 : ', self.appr_conv5_1)
        print('appr_conv5_2 : ', self.appr_conv5_2)
        
        self.appr_conv6_1 = net.ReLU(net.AtConv(self.appr_conv5_2, self.appr_W6_1,16,'appr_conv6_1')+ self.a_b6_1,'appr_relu6_1')
        self.appr_conv6_2 = net.ReLU(net.AtConv(self.appr_conv6_1, self.appr_W6_2, 1,'appr_conv6_2')+ self.a_b6_2,'appr_relu6_2')
        print('appr_conv6_1 : ', self.appr_conv6_1)
        print('appr_conv6_2 : ', self.appr_conv6_2)
        
        self.appr_conv7_1 = net.ReLU(net.AtConv(self.appr_conv6_2, self.appr_W7_1,16,'appr_conv7_1')+ self.a_b7_1,'appr_relu7_1')
        self.appr_conv7_2 = net.ReLU(net.AtConv(self.appr_conv7_1, self.appr_W7_2, 1,'appr_conv7_2')+ self.a_b7_2,'appr_relu7_2')
        print('appr_conv7_1 : ', self.appr_conv7_1)
        print('appr_conv7_2 : ', self.appr_conv7_2)
        
        self.appr_conv8 = net.ReLU(net.Conv(self.appr_conv7_2, self.appr_W8, 1,'appr_conv8')+ self.a_b8_1,'appr_relu8')
        self.appr_out   = net.ReLU(net.Conv(self.appr_conv8, self.appr_Wout, 1,'appr_out')+ self.a_bout,'appr_relu')
        print('appr_conv8   : ', self.appr_conv8)
        print('appr_out     : ', self.appr_out)
        print('\n')
        
        print('< ----------------------- TemporanceNet ----------------------- >\n')
        
        self.tmpr_conv1_1 = net.ReLU(net.Conv(tmpr_X, self.tmpr_W1_1, 2, 'tmpr_conv1_1')+ self.t_b1_1,'tmpr_relu1_1')
        self.tmpr_conv1_2 = net.ReLU(net.Conv(self.tmpr_conv1_1, self.tmpr_W1_2, 1, 'tmpr_conv1_2')+ self.t_b1_2,'tmpr_relu1_2')
        print('tmpr_conv1_1 : ', self.tmpr_conv1_1)
        print('tmpr_conv1_2 : ', self.tmpr_conv1_2)
        
        self.tmpr_conv2_1 = net.ReLU(net.Conv(self.tmpr_conv1_2, self.tmpr_W2_1, 2, 'tmpr_conv2_1')+ self.t_b2_1,'tmpr_relu2_1')
        self.tmpr_conv2_2 = net.ReLU(net.Conv(self.tmpr_conv2_1, self.tmpr_W2_2, 1, 'tmpr_conv2_2')+ self.t_b2_2,'tmpr_relu2_2')
        print('tmpr_conv2_1 : ', self.tmpr_conv2_1)
        print('tmpr_conv2_2 : ', self.tmpr_conv2_2)
        
        self.tmpr_conv3_1 = net.ReLU(net.AtConv(self.tmpr_conv2_2, self.tmpr_W3_1, 2,'tmpr_conv3_1')+ self.t_b3_1,'tmpr_relu3_1')
        self.tmpr_conv3_2 = net.ReLU(net.AtConv(self.tmpr_conv3_1, self.tmpr_W3_2, 1,'tmpr_conv3_2')+ self.t_b3_2,'tmpr_relu3_2')
        print('tmpr_conv3_1 : ', self.tmpr_conv3_1)
        print('tmpr_conv3_2 : ', self.tmpr_conv3_2)
        
        self.tmpr_conv4_1 = net.ReLU(net.AtConv(self.tmpr_conv3_2, self.tmpr_W4_1, 4,'tmpr_conv4_1')+ self.t_b4_1,'tmpr_relu4_1')
        self.tmpr_conv4_2 = net.ReLU(net.AtConv(self.tmpr_conv4_1, self.tmpr_W4_2, 1,'tmpr_conv4_2')+ self.t_b4_2,'tmpr_relu4_2')
        print('tmpr_conv4_1 : ', self.tmpr_conv4_1)
        print('tmpr_conv4_2 : ', self.tmpr_conv4_2)
        
        self.tmpr_conv5_1 = net.ReLU(net.AtConv(self.tmpr_conv4_2, self.tmpr_W5_1, 8,'tmpr_conv5_1')+ self.t_b5_1,'tmpr_relu5_1')
        self.tmpr_conv5_2 = net.ReLU(net.AtConv(self.tmpr_conv5_1, self.tmpr_W5_2, 1,'tmpr_conv5_2')+ self.t_b5_2,'tmpr_relu5_2')
        print('tmpr_conv5_1 : ', self.tmpr_conv5_1)
        print('tmpr_conv5_2 : ', self.tmpr_conv5_2)
        
        self.tmpr_conv6_1 = net.ReLU(net.AtConv(self.tmpr_conv5_2, self.tmpr_W6_1,16,'tmpr_conv6_1')+ self.t_b6_1,'tmpr_relu6_1')
        self.tmpr_conv6_2 = net.ReLU(net.AtConv(self.tmpr_conv6_1, self.tmpr_W6_2, 1,'tmpr_conv6_2')+ self.t_b6_2,'tmpr_relu6_2')
        print('tmpr_conv6_1 : ', self.tmpr_conv6_1)
        print('tmpr_conv6_2 : ', self.tmpr_conv6_2)
        
        self.tmpr_conv7_1 = net.ReLU(net.AtConv(self.tmpr_conv6_2, self.tmpr_W7_1,16,'tmpr_conv7_1')+ self.t_b7_1,'tmpr_relu7_1')
        self.tmpr_conv7_2 = net.ReLU(net.AtConv(self.tmpr_conv7_1, self.tmpr_W7_2, 1,'tmpr_conv7_2')+ self.t_b7_2,'tmpr_relu7_2')
        print('tmpr_conv7_1 : ', self.tmpr_conv7_1)
        print('tmpr_conv7_2 : ', self.tmpr_conv7_2)
        
        self.tmpr_conv8 = net.ReLU(net.Conv(self.tmpr_conv7_2, self.tmpr_W8, 1,'tmpr_conv8')+ self.t_b8_1,'tmpr_relu8')
        self.tmpr_out   = net.ReLU(net.Conv(self.tmpr_conv8, self.tmpr_Wout, 1,'tmpr_out')+ self.t_bout,'tmpr_relu')
        print('tmpr_conv8   : ', self.tmpr_conv8)
        print('tmpr_out     : ', self.tmpr_out)
        print('\n')
        
        return self.appr_out, self.tmpr_out
        
    def flow_refine_net(self, appr_X, p_appr_X, tmpr_X ):
        
        print('< ----------------------- Flow Refine Net ----------------------- >\n')
        
        tmpr_X2  = self.resize(tmpr_X, 1/2) / 2
        tmpr_X4  = self.resize(tmpr_X, 1/4) / 4
        
        #1
        concat0_1      = tf.concat([tmpr_X, appr_X, p_appr_X, appr_X-p_appr_X], 3)
        self.frn_conv0 = net.ReLU(net.Conv(concat0_1, self.frn_W0, 1, 'frn_conv0')+self.frn_Wb0,'frn_relu0')
        self.frn_out0c = net.Conv(self.frn_conv0, self.frn_C0, 1, 'frn_out0c')+self.frn_Cb0
        concat0_2      = tf.concat([self.frn_out0c, tmpr_X], 3)
        self.frn_out0  = net.Conv(concat0_2, self.frn_F0, 1, 'frn_out0')+self.frn_Fb0
        print('frn_conv0   : ', self.frn_conv0)
        print('frn_out0    : ', self.frn_out0)
        
        #1/2
        self.frn_conv1 = net.ReLU(net.Conv(self.frn_conv0, self.frn_W1, 2, 'frn_conv1')+self.frn_Wb1,'frn_relu1')
        concat1_1      = tf.concat([self.frn_conv1, self.resize(self.frn_out0, 1/2)], 3)
        self.frn_out1c = net.Conv(concat1_1, self.frn_C1, 1, 'frn_out1c')+self.frn_Cb1
        concat1_2      = tf.concat([self.frn_out1c, tmpr_X2], 3)
        self.frn_out1  = net.Conv(concat1_2, self.frn_F1, 1, 'frn_out1')+self.frn_Fb1
        print('frn_conv1   : ', self.frn_conv1)
        print('frn_out1    : ', self.frn_out1)
        print('\n')
        
        #1/4
        self.frn_conv2 = net.ReLU(net.Conv(self.frn_conv1, self.frn_W2, 2, 'frn_conv2')+self.frn_Wb2,'frn_relu2')
        concat2_1      = tf.concat([self.frn_conv2, self.resize(self.frn_out1, 1/2)], 3)
        self.frn_out2c = net.Conv(concat2_1, self.frn_C2, 1, 'frn_out2c')+self.frn_Cb2
        concat2_2      = tf.concat([self.frn_out2c, tmpr_X4], 3)
        self.frn_out2  = net.Conv(concat2_2, self.frn_F2, 1, 'frn_out2')+self.frn_Fb2
        print('frn_conv2   : ', self.frn_conv2)
        print('frn_out2    : ', self.frn_out2)
        print('\n')
        
        return self.frn_out0, self.frn_out1, self.frn_out2
        
    def decoder(self, gru_out):
        
        print('< ----------------------- Decoder ----------------------- >\n')
        
        dec_sh1 = [ self.bs, int(np.ceil(self.h/2)), int(np.ceil(self.w/2)), 32 ]
        dec_sh0 = [ self.bs, self.h, self.w, 16 ]
          
        self.dec_conv2_1= net.ReLU(net.DeConv(self.gru_out, self.dec_W2_1, dec_sh1, 2,'dec_conv2_1')+self.d_b2_1,'dec_relu2_1')
        concat2         = tf.concat([self.dec_conv2_1, self.appr_conv1_2, self.tmpr_conv1_2], 3)
        self.dec_conv2_2= net.ReLU(net.Conv(concat2, self.dec_W2_2, 1, 'dec_conv2_2')+self.d_b2_2,'dec_relu2_2')
        print('dec_conv2_1 : ', self.dec_conv2_1)
        print('dec_conv2_2 : ', self.dec_conv2_2)
        
        self.dec_conv1_1= net.ReLU(net.DeConv(self.dec_conv2_2,self.dec_W1_1,dec_sh0, 2,'dec_conv1_1')+self.d_b1_1,'dec_relu1_1')
        self.dec_conv1_2= net.ReLU(net.Conv(self.dec_conv1_1, self.dec_W1_2, 1,'dec_conv1_2')+self.d_b1_2,'dec_relu1_2')
        print('dec_conv1_1 : ', self.dec_conv1_1)
        print('dec_conv1_2 : ', self.dec_conv1_2)
        
        net_out0        = net.Conv(self.dec_conv1_2, self.dec_Wout, 1, 'net_out')
        print('dec_out : ', net_out0)
        print('\n')
        
        return net_out0
        
    def forward_pass(self, appr_X, tmpr_X, p_appr_X, p_tmpr_X, h_prev):
        
        # previous
        p_appr_out, p_tmpr_out = self.encoder(p_appr_X, p_tmpr_X)
        p_appr_out, p_tmpr_out = tf.stop_gradient(p_appr_out), tf.stop_gradient(p_tmpr_out)
        p_apn1   = self.appr_conv1_2
        p_apn1   = tf.stop_gradient(p_apn1)
        p_merged = tf.concat([ self.k * p_appr_out, (1-self.k) * p_tmpr_out], 3)
        self.p_gru_out  = ConvGRU.forward_pass(self, p_merged, h_prev)
        
        # encoder
        appr_out, tmpr_out = self.encoder(appr_X, tmpr_X)
        
        # frn
        frn_out0, frn_out1, frn_out2 = self.flow_refine_net(appr_X, p_appr_X, tmpr_X)
        
        # gru
        self.wrpd_p_gru = tf.map_fn(
            lambda inputs : self.warp(inputs[0],inputs[1]), elems=[self.p_gru_out, frn_out2], dtype=tf.float32)
        self.no_mask_h  = self.no_mask(self.resize(appr_X, 1/4), self.resize(p_appr_X, 1/4), frn_out2)
        self.no_mask_h  = tf.expand_dims(self.no_mask_h, 3)
        
        merged = tf.concat([self.k*self.appr_out, (1-self.k)*self.tmpr_out], 3)
        self.gru_out = ConvGRU.forward_pass(self, merged, self.no_mask_h*self.wrpd_p_gru)
        
        # decoder
        net_out = self.decoder(self.gru_out)
        
        return net_out, frn_out0, frn_out1, frn_out2, self.gru_out, self.no_mask_h*self.wrpd_p_gru
    
    def resize(self, x, ratio):
        
        s = x.get_shape().as_list()
        h = s[1]
        w = s[2]
        
        return tf.image.resize_images(x, [int(np.ceil(h*ratio)), int(np.ceil(w*ratio))])
    
class ConvGRU():
    
    def __init__(self):
        
        self.W_xz  = net.Variable([5, 5, 128, 64],'W_xz')
        self.W_hz  = net.Variable([5, 5, 64, 64],'W_hz')
        self.W_xr  = net.Variable([5, 5, 128, 64],'W_xr')
        self.W_hr  = net.Variable([5, 5, 64, 64],'W_hr')
        self.W_xh_ = net.Variable([5, 5, 128, 64],'W_xh_')
        self.W_hh_ = net.Variable([5, 5, 64, 64],'W_hh_')
        self.b_z = net.Variable(name='b_z',  init=tf.zeros([64]))
        self.b_r = net.Variable(name='b_r',  init=tf.zeros([64]))
        self.b_h = net.Variable(name='b_h',  init=tf.zeros([64]))
        
    def forward_pass(self, merged, h_prev):
        
        print('< ----------------------- ConvGRU ----------------------- >\n')
        
        self.z  = net.Sigmoid(net.Conv(merged, self.W_xz, 1) + net.Conv(h_prev, self.W_hz, 1) + self.b_z)
        self.r  = net.Sigmoid(net.Conv(merged, self.W_xr, 1) + net.Conv(h_prev, self.W_hr, 1) + self.b_r)
        self.h_ = net.Tanh(net.Conv(merged, self.W_xh_, 1) + net.Conv(tf.multiply(self.r,h_prev), self.W_hh_, 1) + self.b_h)
        h       = tf.multiply((1-self.z), h_prev) + tf.multiply(self.z, self.h_)
        
        print('Hidden state :', h)
        print('\n')
        
        return h
    
class TwoStreamDepthNet(TwoStreamNet):
    
    def __init__(self, height, width, batch_size):
        TwoStreamNet.__init__(self, height, width, batch_size)
        
    def forward_pass(self, appr_X, tmpr_X, p_appr_X, p_tmpr_X, h_prev):
        return TwoStreamNet.forward_pass(self, appr_X, tmpr_X, p_appr_X, p_tmpr_X, h_prev)

    def gradient_x(self, img):
        return img[:,:,:-1,:] - img[:,:,1:,:]

    def gradient_y(self, img):
        return img[:,:-1,:,:] - img[:,1:,:,:]
        
    def idx(self, height, width):
        
        idx_x = [ [ i for i in range(width)] for j in range(height) ]
        idx_y = []
        
        for j in range(height):
            idx_y.append([ j for i in range(width) ])

        idx = np.zeros((height, width, 2))
        idx[:, :, 0] = idx_x
        idx[:, :, 1] = idx_y
        
        idx_tensor = tf.constant(idx, dtype=tf.float32)
        
        return idx_tensor
    
    def warp(self, pst, back_flw):
        
        h, w, _ = back_flw.get_shape().as_list()
        index = self.idx(h, w)
        
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
    
    def SSIM(self, x, y):
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = net.AvgP(x, 3, 1, 'VALID')
        mu_y = net.AvgP(y, 3, 1, 'VALID')

        sigma_x  = net.AvgP(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = net.AvgP(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = net.AvgP(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
                 
    def no_mask(self, appr_X, p_appr_X, backflw, alpha=1):
    
        wrpd_p_appr_X = tf.map_fn(lambda inputs : self.warp(inputs[0],inputs[1]), elems=[p_appr_X, backflw], dtype=tf.float32)

        diff = tf.reduce_mean(tf.abs(appr_X - wrpd_p_appr_X), 3)
        msk  = tf.exp((-1) * alpha * diff)

        return msk
    
    def gt_mask(self, gt):
        
        m = tf.cast(tf.not_equal(gt, -tf.ones_like(gt)), tf.float32)
        n = tf.reduce_sum(m, axis=(1,2))
        
        return m, n
    
    def l1(self, pred, gt):

        d = m * tf.abs(pred - gt)
        l = tf.reduce_sum(d, axis=(1,2)) / n

        return l
    
    def smooth_l1(self, pred, gt):
        
        HUBER_DELTA = 1
        
        m,n = self.gt_mask(gt)
        d = m * tf.abs(pred - gt)
        l = tf.where(d < HUBER_DELTA, 0.5 * d ** 2, HUBER_DELTA * (d - 0.5 * HUBER_DELTA))
        
        return  tf.reduce_sum(l, axis=(1,2)) / n
    
    def sci(self, pred, gt):

        m,n = self.gt_mask(gt)
        d = m * (pred - gt)
        l = (tf.reduce_sum(tf.square(d), axis=(1,2)) / n) - 0.5*(tf.square((tf.reduce_sum(d, axis=(1,2)))) / tf.square(n))

        return l
    
    def sci_log(self, pred, gt, alpha=0.5):
        
        pred = net.ReLU(pred)
        
        #< ----------------------- Modify ----------------------- >#
        cmp  = tf.equal(pred, tf.constant(0.))    
        out  = tf.where(cmp, 1e-3*tf.ones_like(pred), pred)

        #< ----------------------- Loss ----------------------- >#
        m,n = self.gt_mask(gt)
        d = tf.log(out) - tf.log(gt)
        d = tf.where(tf.is_nan(d), tf.zeros_like(d), d)
        l = (tf.reduce_sum(tf.square(d), axis=(1,2)) / n) - alpha*(tf.square((tf.reduce_sum(d, axis=(1,2)))) / tf.square(n))

        return tf.reduce_mean(l)

    def smoothness(self, disp, img, alpha=10):
        
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)
        
        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)
        
        weights_x = tf.exp((-1) * alpha * tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp((-1) * alpha * tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))
        
        smoothness_x = tf.abs(disp_gradients_x) * weights_x
        smoothness_y = tf.abs(disp_gradients_y) * weights_y
        
        return 0.5 * tf.reduce_mean(smoothness_x, axis=(1,2)) + 0.5 * tf.reduce_mean(smoothness_y, axis=(1,2))
    
    
    def smoothness_2nd(self, pred, imgs, a):

        gradient_x = self.gradient_x(pred)
        gradient_y = self.gradient_y(pred)
        
        img_x = self.gradient_x(imgs)
        img_y = self.gradient_y(imgs)

        smoothness_2nd_x = self.smoothness(disp = gradient_x, img = img_x, alpha=a)
        smoothness_2nd_y = self.smoothness(disp = gradient_y, img = img_y, alpha=a)

        smoothness_loss = 0.5*(smoothness_2nd_x + smoothness_2nd_y)

        return smoothness_loss

    def temp_loss(self, prs, pst, back_flw, gt, appr_X, p_appr_X, thrs=0.5, alpha=0.5):
        
        no_m = self.no_mask(appr_X, p_appr_X, back_flw) > thrs
        no_m = tf.cast(tf.expand_dims(no_m, 3), tf.float32)
        no_n = tf.reduce_sum(no_m, axis=(1,2))
        
        pst_warped = tf.map_fn(lambda inputs : self.warp(inputs[0],inputs[1]), elems=[pst, back_flw], dtype=tf.float32)
        
        prs = net.ReLU(prs)
        pst_warped = net.ReLU(pst_warped)
        
        prs = tf.where(tf.equal(prs, tf.constant(0.)), 1e-3*tf.ones_like(prs), prs)
        pst_warped = tf.where(tf.equal(pst_warped, tf.constant(0.)), 1e-3*tf.ones_like(pst_warped), pst_warped)
        
        d = tf.log(prs) - tf.log(pst_warped)
        d = no_m * tf.where(tf.is_nan(d), tf.zeros_like(d), d)
        l = (tf.reduce_sum(tf.square(d), axis=(1,2))/no_n) - alpha*(tf.square((tf.reduce_sum(d, axis=(1,2))))/tf.square(no_n))

        return l
    
    def phot_loss(self, prs, pst, back_flw, alpha=0.85):
        
        _, height, width, _ = back_flw.get_shape().as_list()

        pst_warped = tf.map_fn(lambda inputs : self.warp(inputs[0],inputs[1]), elems=[pst, back_flw], dtype=tf.float32)

        l1_loss = tf.abs(prs - pst_warped)
        SSIM_loss = self.SSIM(prs, pst_warped)

        return (1-alpha) * tf.reduce_mean(l1_loss) + alpha * tf.reduce_mean(SSIM_loss)