


# Deformable Kernel Network for Joint Image Filtering
This is the pytorch implementation of our DKN [[Paper](https://arxiv.org/abs/1910.08373)] 
![image](https://user-images.githubusercontent.com/5655912/37342239-53239644-2707-11e8-85b1-9b25c290d81e.png)


## 1. Requirement

Pytorch 1.0.0

Cuda 10.0

Opencv 3.4

tqdm

logging


## 2. Dataset
NYU v2 dataset, our split can be downloaded: http://gofile.me/3G5St/2lFq5R3TL


## 3. Inference

- DKN
> python inference.py --rgb images/0\_rgb.png --depth images/0\_lr.png --k 3 --d 15 --scale 8 --parameter parameter/DKN_8x --output images/result_dkn.png --model DKN

- FDKN
> python inference.py --rgb images/0\_rgb.png --depth images/0\_lr.png --k 3 --d 15 --scale 8 --parameter parameter/FDKN_8x --output images/result_fdkn.png --model FDKN



## 4. Train

- DKN
> python train.py --k 3 --d 15 --scale 8 --model DKN

- FDKN
> python train.py --k 3 --d 15 --scale 8 --model FDKN

