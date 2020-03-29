# PyTorch implementation of RRID

<img src="../RRID_files/Overview.png" alt="no_image"/>
This is the implementation of the paper "Relation Network for Person Re-identification".

For more information, checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/RRID/)] and the paper [[PDF](https://arxiv.org/pdf/1911.09318.pdf)].

## Dependencies
* Python 3.6
* PyTorch >= 0.4.1
* numpy
* h5py

## Datasets
Download the datasets into ``datasets`` folder.
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
## Pre-trained model
* Download pre-trained weights <br>Link: [[weights]()]
* Two gpus are needed to implement this weights
* The version of pytorch must be 0.4.1 when you implement the model with this weights

## Evaluation
```bash
python Evaluate.py --gpus 0 1 --pretrained_weights_dir pretrained_weights.pth 
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
