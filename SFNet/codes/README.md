# PyTorch implementation of SFNet

<img src="../SFNet_files/method.png" alt="no_image"/>
This is the implementation of the paper "SFNet: Learning Object-aware Semantic Correspondence".

For more information, checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/SFNet/)] and the paper [[PDF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_SFNet_Learning_Object-Aware_Semantic_Correspondence_CVPR_2019_paper.pdf)].

## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* pandas

## Datasets
* Pascal VOC 2012 segmentation dataset (excluding images that overlap with the test split in the PF-PASCAL) for training
* PF-Pascal & PF-WILLOW datasets for evaluation
* All datasets are automatically downloaded into the ``data`` folder by running ``download_datasets.py``
* FYI: Without the need of downloading the entire data, the csv file of PF-PASCAL test split is available in the following link: [[csv file](https://drive.google.com/open?id=1cdw5XaMfq3rptwIFNmzOnykUozUZfIMj)]

## Code
```bash
git clone https://github.com/cvlab-yonsei/projects
cd projects/SFNet/codes
python3 download_datasets.py # prepare the datasets for training/evaluation
python3 train.py # for training
python3 eval_pascal.py # evaluation on PF-Pascal dataset
python3 eval_willow.py # evaluation on PF-WILLOW dataset
```

## Trained model
* Download pre-trained weights into ``weights`` folder <br>Link: [[weights](https://drive.google.com/a/yonsei.ac.kr/file/d/1RmVcrla-7qUYVxRdr6ngqRmmu2RK-qrk/view?usp=sharing)]

## Bibtex
```
@inproceedings{lee2019sfnet,
  title={SFNet: Learning Object-aware Semantic Correspondence},
  author={Lee, Junghyup and Kim, Dohyung and Ponce, Jean and Ham, Bumsub},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2278--2287},
  year={2019}
}
```
