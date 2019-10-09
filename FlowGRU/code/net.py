import tensorflow as tf
    
def Variable(shape=None, name=None, init=tf.contrib.layers.xavier_initializer(uniform=False)):
    return tf.get_variable(name, shape=shape, initializer=init)

def Conv(fmap, weight, stride, name=None, zeropd='SAME'):
    return tf.nn.conv2d(fmap, weight, [1,stride,stride,1], padding=zeropd, name=name)

def DeConv(fmap, filtdim, filtsize, stride, name=None):
    return tf.nn.conv2d_transpose(fmap, filtdim, filtsize, [1,stride,stride,1], name=name)

def AtConv(fmap, weight, rate, name=None, zeropd='SAME'):
    return tf.nn.atrous_conv2d(fmap, weight, rate, zeropd, name=name)

def BN(fmap):
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
