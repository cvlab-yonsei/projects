import tensorflow as tf
    
def Variable(shape=None, name=None, init=tf.contrib.layers.xavier_initializer(uniform=False)):
    return tf.get_variable(name, shape=shape, initializer=init)
#    return tf.get_variable(name, shape=shape, initializer=tf.keras.initializers.he_normal()) 

# def Conv( x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu ):
#         p = np.floor((kernel_size - 1) / 2).astype(np.int32)
#         p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
#         return slim.conv2d( p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn )

def Conv( fmap, weight, stride, name=None, zeropd='SAME' ):
    return tf.nn.conv2d(fmap, weight, [1,stride,stride,1], padding=zeropd, name=name)

def DeConv( fmap, filtdim, filtsize, stride, name=None ):
    return tf.nn.conv2d_transpose(fmap, filtdim, filtsize, [1,stride,stride,1], name=name)

def AtConv( fmap, weight, rate, name=None, zeropd='SAME' ):
    return tf.nn.atrous_conv2d(fmap, weight, rate, zeropd, name=name)

# def BN(fmap, reuse=tf.AUTO_REUSE):
#     return tf.contrib.layers.batch_norm(fmap, center=True, scale=True, is_training=True, scope=name, reuse=reuse)
def BN( fmap ):
    return tf.contrib.layers.batch_norm(fmap, center=True, scale=True, is_training=True, scope=name)

def ReLU(fmap, name=None):
    return tf.nn.relu(fmap, name=name)

def LReLu(fmap, name=None):
    return tf.nn.leaky_relu(fmap, name=name)

def ELU(fmap, name=None):
    return tf.nn.elu(fmap)

def Sigmoid(fmap, name=None):
    return tf.nn.sigmoid(fmap, name=name)

def Tanh(fmap, name=None):
    return tf.nn.tanh(fmap, name=name)

def MaxP(fmap, fsize, stride, zeropd='SAME'):
    return tf.nn.max_pool(fmap, [1,fsize,fsize,1], [1,stride,stride,1], padding=zeropd)

def AvgP(fmap, fsize, stride, zeropd='SAME'):
    return tf.nn.avg_pool(fmap, [1,fsize,fsize,1], [1,stride,stride,1], padding=zeropd)
