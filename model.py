from keras import layers
from keras.models import Model
from keras import backend as K
from core_layers import *

def conv_relu(x, filters, kernel, padding='same', use_bias = True, dilation_rate=1, strides=(1,1)):
    if dilation_rate == 0:
        y = layers.Conv2D(filters,1,padding=padding,use_bias=use_bias,
            activation='relu')(x)
    else:
        y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
            dilation_rate=dilation_rate,
            strides=strides,
            activation='relu')(x)
    return y

def conv(x, filters, kernel, padding='same', use_bias=True, dilation_rate=1, strides = (1,1)):
    y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
        dilation_rate=dilation_rate, strides=strides)(x)
    return y

def conv_bn_relu(x, filters, kernel, padding='same', use_bias = True, dilation_rate=1):
    y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
        dilation_rate=dilation_rate)(x)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Activation('relu')(y)
    return y

def conv_prelu(x, filters, kernel, padding='same', use_bias=False, dilation_rate=1, strides = (1,1)):
    y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
        dilation_rate=dilation_rate, strides=strides)(x)
    y = layers.advanced_activations.PReLU()(y)
    return y

def MBCNN(nFilters, multi=True):
    conv_func = conv_relu
    def pre_block(x, d_list, enbale = True):
        t = x
        for i in range(len(d_list)):
            _t = conv_func(t, nFilters, 3, dilation_rate=d_list[i])
            t = layers.Concatenate(axis=-1)([_t,t])
        t = conv(t, 64, 3)
        t = adaptive_implicit_trans()(t)
        t = conv(t,nFilters*2,1)
        t = ScaleLayer(s=0.1)(t)
        if not enbale:
            t = layers.Lambda(lambda x: x*0)(t)
        t = layers.Add()([x,t])
        return t

    def pos_block(x, d_list):
        t = x
        for i in range(len(d_list)):
            _t = conv_func(t, nFilters, 3, dilation_rate=d_list[i])
            t = layers.Concatenate(axis=-1)([_t,t])
        t = conv_func(t, nFilters*2, 1)
        return t

    def global_block(x):
        t = layers.ZeroPadding2D(padding=(1,1))(x)
        t = conv_func(t, nFilters*4, 3, strides=(2,2))
        t = layers.GlobalAveragePooling2D()(t)
        t = layers.Dense(nFilters*16,activation='relu')(t)
        t = layers.Dense(nFilters*8, activation='relu')(t)
        t = layers.Dense(nFilters*4)(t)
        _t = conv_func(x, nFilters*4, 1)
        _t = layers.Multiply()([_t,t])
        _t = conv_func(_t, nFilters*2, 1)
        return _t

    output_list = []
    d_list_a = (1,2,3,2,1)
    d_list_b = (1,2,3,2,1)
    d_list_c = (1,2,2,2,1)
    x = layers.Input(shape=(None, None, 3))                 #16m*16m
    _x = Space2Depth(scale=2)(x)
    t1 = conv_func(_x,nFilters*2,3, padding='same')          #8m*8m
    t1 = pre_block(t1, d_list_a, True)
    t2 = layers.ZeroPadding2D(padding=(1,1))(t1)
    t2 = conv_func(t2,nFilters*2,3, padding='valid',strides=(2,2))              #4m*4m
    t2 = pre_block(t2, d_list_b,True)
    t3 = layers.ZeroPadding2D(padding=(1,1))(t2)
    t3 = conv_func(t3,nFilters*2,3, padding='valid',strides=(2,2))              #2m*2m
    t3 = pre_block(t3,d_list_c, True)
    t3 = global_block(t3)
    t3 = pos_block(t3, d_list_c)
    t3_out = conv(t3, 12, 3)
    t3_out = Depth2Space(scale=2)(t3_out)           #4m*4m
    output_list.append(t3_out)
    _t2 = layers.Concatenate()([t3_out,t2])
    _t2 = conv_func(_t2, nFilters*2, 1)
    _t2 = global_block(_t2)
    _t2 = pre_block(_t2, d_list_b,True)
    _t2 = global_block(_t2)
    _t2 = pos_block(_t2, d_list_b)
    t2_out = conv(_t2, 12, 3)
    t2_out = Depth2Space(scale=2)(t2_out)           #8m*8m
    output_list.append(t2_out)
    _t1 = layers.Concatenate()([t1, t2_out])
    _t1 = conv_func(_t1, nFilters*2, 1)
    _t1 = global_block(_t1)
    _t1 = pre_block(_t1, d_list_a, True)
    _t1 = global_block(_t1)
    _t1 = pos_block(_t1, d_list_a)
    _t1 = conv(_t1,12,3)
    y = Depth2Space(scale=2)(_t1)                           #16m*16m
    output_list.append(y)
    if multi != True:
        return models.Model(x,y)
    else:
        return models.Model(x,output_list)
