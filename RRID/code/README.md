# PyTorch implementation of RRID

<img src="../RRID_files/Overview.png" alt="no_image"/>{: width="80%" height="80%"}
This is the implementation of the paper "Relation Network for Person Re-identification".

For more information, checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/RRID/)] and the paper [[PDF](https://arxiv.org/pdf/1911.09318.pdf)].

## Dependencies
* Python 3.6
* PyTorch >= 0.4.1
* numpy
* h5py

## Datasets
Download the datasets into ``datasets`` folder, like ``./datasets/market1501/``. The market1501 dataset is only available now, and other datasets will be uploaded later
* Market1501 [[market1501]()]
* DukeMTMC-ReID [[dukemtmc]()]
* CUHK03 labeled [[cuhk03_labeled]()]
* CUHK03 detected [[cuhk03_labeled]()]

## Training
```bash
git clone https://github.com/cvlab-yonsei/projects
cd projects/RRID/codes
python Train.py  # for training
```
* You can freely define parameters with your own settings like
```bash
python Train.py --gpus 0 1 --dataset_path 'your_dataset_directory' --dataset_type market1501 --exp_dir 'your_log_directory'
```
## Pre-trained model
* Download pre-trained weights <br>Link: [[weights](https://drive.google.com/file/d/1x7Hqb3MY8kPJhhWFHiI-fvwJ8MKS5muy/view?usp=sharing)]
* Two gpus are needed to implement this weights
* The version of pytorch must be 0.4.1 when you implement the model with this weights

## Evaluation
* Test the model with our pre-trained weights 
```bash
python Evaluate.py --gpus 0 1 --pretrained_weights_dir pretrained_weights.pth 
```
* Test your own model
```bash
python Evaluate.py --exp dir log
```

## Bibtex
```
@article{park2019relation,
  title={Relation Network for Person Re-identification},
  author={Park, Hyunjong and Ham, Bumsub},
  journal={arXiv preprint arXiv:1911.09318},
  year={2019}
}
```
