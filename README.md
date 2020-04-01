# Learnbale_Bandpass_Filter
Image Demoireing with Learnable Bandpass Filter, CVPR2020

If you find this work is helpful, please cite:

@inProceedings{zheng2020,  
author={B. {Zheng} and S. {Yuan} and g. {Slabaugh} and A. {Leonardis} ,  
booktitle={IEEE Conference on Computer Vision and Pattern Recongnition},  
title={Image Demoireing with Learnable Bandpass Filter},  
year={2020},  
}

## Run the code
This project requires:
* Tensorflow >1.10
* Keras > 2.0
* opencv > 2.0
* skImage

You can get the weight file for AIM2019 via:  
https://1drv.ms/u/s!ArU0YIIFiFuHilwyuwHZjSpvPUBz?e=iZ70Ga

You can get AIM2019 LCDMoire2019 dataset via:
validation:   
Moire: https://data.vision.ee.ethz.ch/timofter/AIM19demoire/ValidationMoire.zip  
Clean: https://data.vision.ee.ethz.ch/timofter/AIM19demoire/ValidationClear.zip  

testing:  
https://data.vision.ee.ethz.ch/timofter/AIM19demoire/TestingMoire.zip

You can get the weight file for TIP2018 dataset via:  


Then,  
1. edit the 'main_multiscale.py' by:
replacing the 'test_path', 'valid_gt_path', 'valid_ns_path' and 'weight_path' with your own settings.  

2. make the dirs 'testing_result' and 'validation_result' at current path.  

3. python main_multiscale.py.  

