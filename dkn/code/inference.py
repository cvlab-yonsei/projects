import torch
import numpy as np
import cv2
import argparse

from models import *



parser = argparse.ArgumentParser()
parser.add_argument('--rgb',  default='images/0_rgb.png', help='name of rgb image')
parser.add_argument('--depth',  default='images/0_lr.png', help='name of low resolution depth image')
parser.add_argument('--k', type=int, default=3, help='size of kernel')
parser.add_argument('--d', type=int, default=15, help='size of grid area')
parser.add_argument('--scale', type=int, default=8, help='scale factor')
parser.add_argument('--parameter',  default='parameter/FDKN_8x', help='name of parameter file')
parser.add_argument('--model',  default='FDKN', help='choose model FDKN or DKN')
parser.add_argument('--output',  default='images/0_dkn.png', help='name of output image')
opt = parser.parse_args()
print(opt)

def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo

    return image[:h,:w]


if opt.model == 'FDKN':
    net = FDKN(kernel_size=opt.k, filter_size=opt.d, residual=True).cuda()
elif opt.model == 'DKN':
    net = DKN(kernel_size=opt.k, filter_size=opt.d, residual=True).cuda()

net.load_state_dict(torch.load(opt.parameter))
net.eval()
print('parameter \"%s\" has loaded'%opt.parameter)


rgb = cv2.imread(opt.rgb).astype('float32')/255.0
rgb = modcrop(rgb, opt.scale)
rgb = np.transpose(rgb, (2,0,1))

lr = cv2.imread(opt.depth,cv2.IMREAD_GRAYSCALE).astype('float32')/255.0
lr = modcrop(lr, opt.scale)
lr = np.expand_dims(lr, 0)

image = torch.from_numpy(np.expand_dims(rgb,0)).cuda()
depth = torch.from_numpy(np.expand_dims(lr,0)).cuda()



with torch.no_grad():
    res_img = net((image, depth)).cpu().numpy()
    
cv2.imwrite(opt.output, res_img[0,0]*255)
