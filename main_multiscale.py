import tensorflow as tf
import os, cv2, random, keras
from skimage.measure import compare_ssim
import numpy as np
from model import *
from keras import optimizers
from keras.utils import multi_gpu_model
from math import log10
from utils import *
from datetime import datetime
import time
from PIL import Image


# Validation or Testing
def validate_ssim(model, gt_list, ns_list, name_list, multi_output=False):
    print ("validating... ",datetime.now().strftime('%H:%M:%S'))
    psnr = 0
    ssim = 0
    count = 0
    for i in range(len(gt_list)):	
        count += 1
        gt = gt_list[i]
        ns = ns_list[i]	
        dn = model.predict(ns)#[-1]
        if multi_output:
            dn = dn[-1]
        _psnr = 10*log10(1/np.mean((dn-gt)**2))
        _ssim = compare_ssim(dn[0],gt[0],multichannel=True)
        psnr += _psnr
        ssim += _ssim
        _gt = dn[0]
        _gt[_gt>1] = 1
        _gt[_gt<0] = 0
        _gt = _gt*255.0
        _gt = np.round(_gt).astype(np.uint8)
        cv2.imwrite('validation_images/'+name_list[i],_gt)

    print (np.round(psnr/count,3),np.round(ssim/count,4))
    return psnr/count

def test(model,multi_output=False):
    psnr = 0
    count = 0
    valid_path = 'F:\\dataset\\AIM2019\\Testing\\'
    file_list = os.listdir(valid_path)
    file_list = list_filter(file_list,'.png')
    for f in file_list:
        ns = cv2.imread(valid_path+f)
        ns = ns.astype(np.float32)/255.0
        ns = ns.reshape((1,)+ns.shape)
        start = time.clock()
        _gt = model.predict(ns)
        if multi_output:
            _gt = _gt[-1]
        end = time.clock()
        print(f, end-start)
        _gt = _gt[0]
        _gt[_gt>1] = 1
        _gt[_gt<0] = 0
        _gt = _gt*255.0
        _gt = np.round(_gt).astype(np.uint8)
        cv2.imwrite('compare/'+f,_gt)

#Generating validation datas
def generate_validation(valid_list, multi_input, mode='sub'):
    valid_gt_list = []
    valid_ns_list = []
    name_list = []
    _width = 128
    for f in valid_list:
        _name = os.path.splitext(f)[0]
        gt = cv2.imread(valid_gt_path+f)
        ns = cv2.imread(valid_ns_path+f)
        gt = gt.astype(np.float32)/255.0
        ns = ns.astype(np.float32)/255.0
        name_list.append(f)
        if multi_input:
            ns_x2 = cv2.imread(ns_path+'X2\\'+_name+'.png')
            ns_x4 = cv2.imread(ns_path+'X4\\'+_name+'.png')
            ns_x8 = cv2.imread(ns_path+'X8\\'+_name+'.png')
            ns_x2 = ns_x2.astype(np.float32)/255.0
            ns_x4 = ns_x4.astype(np.float32)/255.0
            ns_x8 = ns_x8.astype(np.float32)/255.0
            
        if mode == 'sub':
            for i in range(0,1024,_width):
                for j in range(0,1024,_width):
                    _gt = gt[i:i+_width,j:j+_width]
                    _ns = ns[i:i+_width,j:j+_width]
                    _gt = _gt.reshape((1,)+_gt.shape)
                    _ns = _ns.reshape((1,)+_ns.shape)
                    valid_gt_list.append(_gt)
                    valid_ns_list.append(_ns)
        if mode == 'full':
            gt = gt.reshape((1,)+gt.shape)
            ns = ns.reshape((1,)+ns.shape)
            valid_gt_list.append(gt)
            if multi_input:
                ns_x2 = ns_x2.reshape((1,)+ns_x2.shape)
                ns_x4 = ns_x4.reshape((1,)+ns_x4.shape)
                ns_x8 = ns_x8.reshape((1,)+ns_x8.shape)
                valid_ns_list.append([ns,ns_x2,ns_x4,ns_x8])
            else:
                valid_ns_list.append(ns)
    return valid_gt_list, valid_ns_list, name_list 

#Training
multi_gpu = False
if multi_gpu:	
	print ('use multi gpu mode!')
	os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
else:
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
keras.backend.tensorflow_backend.set_session(get_session())

valid_gt_path = 'Validation\\clear\\'
valid_ns_path = 'Validation\\moire\\'


multi_input = False
multi_output = True
model = MBCNN(64,multi_output)

model.summary()
#exit(0)

valid_list = os.listdir(valid_gt_path)
valid_list = list_filter(valid_list, '.png')

valid_gt_list, valid_ns_list, name_list = generate_validation(valid_list, multi_input, 'full')
model.load_weights('model/MBCNN_weights.h5', by_name = True)
#output validation results
#min_loss = validate_ssim(model, valid_gt_list, valid_ns_list, name_list, multi_output)
#output testing results
test(model, multi_output)
exit(0)
