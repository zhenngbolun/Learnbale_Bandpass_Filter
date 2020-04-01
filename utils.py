import numpy as np
import math, os
from keras import optimizers, backend
import tensorflow as tf
import cv2

def get_Y(x):
	r = x[:,:,0]
	g = x[:,:,1]
	b = x[:,:,2]
	y = 0.257*r + 0.504*g + 0.098*b + 0.0627
	return y

def calc_PSNR(x,y):
	mse = np.mean(np.square(x-y))
	psnr = 10*math.log10(1/mse)
	return psnr

def get_session():
   config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
   return tf.Session(config = config)

def list_filter(file_list, tail):
	r = []
	for f in file_list:
		s = os.path.splitext(f)
		if s[1] == tail:
			r.append(f)
	return r

def data_augmentation(x, method):
   if method == 0:
      return np.rot90(x)
   if method == 1:
      return np.fliplr(x)
   if method == 2:
      return np.flipud(x)
   if method == 3:
      return np.rot90(np.rot90(x))
   if method == 4:
      return np.rot90(np.fliplr(x))
   if method == 5:
      return np.rot90(np.flipud(x))

# clear 0.6806, 0.6876, 0.6954
# moire 0.3978, 0.4027, 0.4074
def calc_meanRGB(img_dirs, tail):
    file_list = os.listdir(img_dirs)
    file_list = list_filter(file_list, tail)
    m = np.zeros((3,))
    count = 0
    for f in file_list:
        count += 1
        
        img = cv2.imread(img_dirs+f)
        img = img.astype(np.float32)/255.0
        m += np.mean(img, axis=(0,1))
        _m = m/count
        print('%d: %s (%f, %f, %f)'%(count, f, _m[0], _m[1], _m[2]), end='\r')
    print(np.round(m/count,4))

def crop(x,scale):
    shape = x.shape
    h = shape[0]
    w = shape[1]
    h = h-h%scale
    w = w-w%scale
    return x[0:h,0:w]