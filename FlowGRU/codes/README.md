# Temporally Consistent Depth Prediction with Flow-Guided Memory Units
This repository contains a [TensorFlow](https://www.tensorflow.org/) implementation for our [flowGRU paper](). Our code is released only for scientific or personal use. Please contact us for commercial use.

## 1. Requirement

TensorFlow 1.4.0

Cuda 8.0

Opencv 3.3.1

## 2. Getting Started

- Datasets
We conduct experiments on [KITTI](http://www.cvlibs.net/datasets/kitti/) and [Cityscapes](https://www.cityscapes-dataset.com/). Our method needs additional optical flow and [DIS-flow](https://github.com/tikroeger/OF_DIS) is used. For convenience, we provide precomputed flow [here](https://drive.google.com/open?id=1IiK7XwRdWQYJ5-IKik2L-7VQ0FEOYu9J).

- Training
You can train our model using the below command on the specified GPUs by setting CUDA_VISIBLE_DEVICES. We also provide the link for our pre-trained weights [trained_on_KITTI](https://drive.google.com/file/d/1IYHORs4LI8o3h1XGGsLCBuf7X-Tr_52g/view?usp=sharing) and [trained_on_Cityscape_and_fine-tuned_on_KITTI](https://drive.google.com/open?id=1A2JcwoVg8D1tJTPmwz1Zb1vKrdVfI6hF).
> python python main.py --data_path '/path/to/dataset'

- Test
> python NoiseReduction.py --target images/target.png --guidance images/guidance.png --k 3 --d 15 --parameter parameter/Noise --output images/noise_reduction.png


## TODO

## Citation
Please cite our paper if you find the code useful for your research.
```
@inproceedings{
}
```

