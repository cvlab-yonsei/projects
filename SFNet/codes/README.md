# PyTorch implementation of SFNet

## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* pandas

## Training data
* Pascal VOC 2012 segmentation dataset (excluding images that overlap with the test split in the PF-PASCAL)
* Download pre-processed data (.npy files) into ``data`` folder <br>Links: [[images](https://drive.google.com/a/yonsei.ac.kr/file/d/1RrAdFyTrSmK6Lee9gN1D4p-FhTzV8ict/view?usp=sharing)] [[binary masks](https://drive.google.com/a/yonsei.ac.kr/file/d/1S_MwNXYBV171hMCnOgY4e2weI7VUPWXj/view?usp=sharing)]

## Code
```bash
git clone https://github.com/cvlab-yonsei/projects
cd projects/SFNet/codes
python3 train.py # for training
python3 eval.py # for testing
```

## Trained model
* Download pre-trained weights into ``weights`` folder <br>Link: [[weights](https://drive.google.com/a/yonsei.ac.kr/file/d/1RmVcrla-7qUYVxRdr6ngqRmmu2RK-qrk/view?usp=sharing)]