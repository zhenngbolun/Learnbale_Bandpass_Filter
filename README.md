# Learnbale_Bandpass_Filter
Image Demoireing with Learnable Bandpass Filters, CVPR2020

If you find this work is helpful, please cite:

@article{zheng2021learn,
  title={Learning Frequency Domain Priors for Image Demoireing},
  author = {Bolun, Zheng and Shanxin, Yuan and Chenggang, Yan and Xiang, Tian and Jiyong, Zhang and Yaoqi, Sun and Lin, Liu and Ales, Leonardis and Gregory, Slabaugh},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}

@inProceedings{zheng2020,  
author={B. Zheng and S. Yuan and G. Slabaugh and A. Leonardis},  
booktitle={IEEE Conference on Computer Vision and Pattern Recongnition},  
title={Image Demoireing with Learnable Bandpass Filters},  
year={2020},  
}

@article{zheng2019implicit,
  title={Implicit dual-domain convolutional network for robust color image compression artifact reduction},
  author={Zheng, Bolun and Chen, Yaowu and Tian, Xiang and Zhou, Fan and Liu, Xuesong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={30},
  number={11},
  pages={3982--3994},
  year={2019},
  publisher={IEEE}
}

You can now get this paper at Arxiv preprint: https://arxiv.org/abs/2004.00406
## Run the code
This project requires:
* Tensorflow >1.10
* Keras > 2.0
* opencv > 2.0
* skImage

You can get the weight file for AIM2019 via:  
https://1drv.ms/u/s!ArU0YIIFiFuHilwyuwHZjSpvPUBz?e=iZ70Ga  
or via Baidu Disk：  
https://pan.baidu.com/s/1wsJYyYbQO-ETL5Jq4fN6hw code：jiae   

You can get AIM2019 LCDMoire2019 dataset via:
validation:   
Moire: https://data.vision.ee.ethz.ch/timofter/AIM19demoire/ValidationMoire.zip  
Clean: https://data.vision.ee.ethz.ch/timofter/AIM19demoire/ValidationClear.zip  

testing:  
https://data.vision.ee.ethz.ch/timofter/AIM19demoire/TestingMoire.zip


Then,  
1. edit the 'main_multiscale.py' by:
replacing the 'test_path', 'valid_gt_path', 'valid_ns_path' and 'weight_path' with your own settings.  

2. make the dirs 'testing_result' and 'validation_result' at current path.  

3. python main_multiscale.py.  

