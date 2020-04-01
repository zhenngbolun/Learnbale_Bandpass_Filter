import tensorflow as tf
from keras import backend as k
from keras.utils import conv_utils
import numpy as np
from keras import layers, models, activations, initializers, constraints
from math import cos, pi, sqrt
from keras.regularizers import l2

class Space2Depth(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(Space2Depth, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs, **kwargs):
        return tf.space_to_depth(inputs, self.scale)

    def compute_output_shape(self, input_shape):
        if input_shape[1] != None and input_shape[2] != None:
            return (None, int(input_shape[1]/self.scale), int(input_shape[2]/self.scale), input_shape[3]*self.scale**2)
        else:
            return (None, None, None, input_shape[3]*self.scale**2)

class Depth2Space(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(Depth2Space, self).__init__(**kwargs)
        self.scale = scale
    def call(self, inputs, **kwargs):
        return tf.depth_to_space(inputs, self.scale)

    def compute_output_shape(self, input_shape):
        if input_shape[1] != None and input_shape[2] != None:
            return (None, input_shape[1]*self.scale, input_shape[2]*self.scale, int(input_shape[3]/self.scale**2))
        else:
            return (None, None, None, int(input_shape[3]/self.scale**2))

class adaptive_implicit_trans(layers.Layer):
    def __init__(self, **kwargs):
        super(adaptive_implicit_trans, self).__init__(**kwargs)

    def build(self, input_shape):
        conv_shape = (1,1,64,64)
        self.it_weights = self.add_weight(
            shape = (1,1,1,64),
            initializer = initializers.get('ones'),
            constraint = constraints.NonNeg(),
            name = 'ait_conv')
        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0/8)
        r2 = sqrt(2.0/8)
        for i in range(8):
            _u = 2*i+1
            for j in range(8):
                _v = 2*j+1
                index = i*8+j
                for u in range(8):
                    for v in range(8):
                        index2 = u*8+v
                        t = cos(_u*u*pi/16)*cos(_v*v*pi/16)
                        t = t*r1 if u==0 else t*r2
                        t = t*r1 if v==0 else t*r2
                        kernel[0,0,index2,index] = t
        self.kernel = k.variable(value = kernel, dtype = 'float32')

    def call(self, inputs):
        #it_weights = k.softmax(self.it_weights)
        #self.kernel = self.kernel*it_weights
        self.kernel = self.kernel*self.it_weights
        y = k.conv2d(inputs,
                        self.kernel,
                        padding = 'same',
                        data_format='channels_last')
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class ScaleLayer(layers.Layer):
    def __init__(self, s, **kwargs):
        self.s = s
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape = (1,),
            name = 'scale',
            initializer=initializers.Constant(value=self.s))
    def call(self, inputs):
        return inputs*self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape