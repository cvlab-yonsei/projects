# PyTorch implementation of RRID

<img src="../RRID_files/Overview.png" alt="no_image"/>
This is the implementation of the paper "Relation Network for Person Re-identification".

For more information, checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/RRID/)] and the paper [[PDF](https://arxiv.org/pdf/1911.09318.pdf)].

## Dependencies
* Python 3.6
* PyTorch 0.4.1
* numpy
* h5py

## Datasets
* Market1501 [[market1501]()]
* DukeMTMC-ReID [[dukemtmc]()]
* CUHK03 labeled [[cuhk03_labeled]()]
* CUHK03 detected [[cuhk03_labeled]()]
Download the datasets into ``datasets`` folder.

## Training
```bash
git clone https://github.com/cvlab-yonsei/projects
cd projects/RRID/codes
python Train.py --gpus  # for training
```
## Trained model
* Download pre-trained weights into ``weights`` folder <br>Link: [[weights]()]

## Evaluation
```bash
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
