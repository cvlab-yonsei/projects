import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader

import time
import numpy as np
import os
import sys
sys.path.insert(0, '.')

from utils import get_data
from Relation_final_ver_last_multi_scale_large_losses import RelationModel as Model
import reid.evaluators as evaluators
from collections import OrderedDict
from reid.utils.meters import AverageMeter

import argparse

parser = argparse.ArgumentParser(description="RRID")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--split', type=int, default=0, help='split')
parser.add_argument('--batch_sample', type=int, default=4, help='same ids in batch')
parser.add_argument('--h', type=int, default=384, help='height of input images')
parser.add_argument('--w', type=int, default=128, help='width of input images')
parser.add_argument('--dataset_type', type=str, default='market1501', help='type of dataset: market1501, dukemtmc, cuhk03')
parser.add_argument('--dataset_path', type=str, default='./datasets/', help='directory of data')
parser.add_argument('--print_freq', type=int, default=10, help='frequency of printing')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--combine_trainval', action="store_true", default=False, help='select train or trainval')
parser.add_argument('--pretrained_weights_dir', type=str, default=None, help='pretrained weights')

args = parser.parse_args()

gpus = ""
for i in range(len(args.gpus)):
    gpus = gpus + args.gpus[i] + ","
    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

log_directory = os.path.join(args.exp_dir + '_' + args.dataset_type)

np_ratio = args.batch_sample - 1

dataset, _, _, test_loader = get_data(args.dataset_type, args.split, args.dataset_path, args.h, args.w, 
                                                args.batch_size, args.num_workers, args.combine_trainval, 
                                                np_ratio)

# Model (RRID)
model = Model(last_conv_stride=1, num_stripes=6, local_conv_out_channels=256)

# If single gpu
if len(args.gpus) < 2:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

# If multi gpus
if len(args.gpus) > 1:
    model = torch.nn.DataParallel(model, range(len(args.gpus))).cuda()


if args.pretrained_weights_dir:
    model = torch.load(args.pretrained_weights_dir)
    
else:
    model = torch.load(os.path.join(exp_dir, 'model.pth'))
    
model.eval()
batch_time = AverageMeter()
data_time = AverageMeter()

features = OrderedDict()
labels = OrderedDict()

end = time.time()
print('Extracting features... This may take a while...')
with torch.no_grad():
    for i, (imgs, fnames, pids, _) in enumerate(test_loader):
        data_time.update(time.time() - end)

        imgs_flip = torch.flip(imgs, [3])
        final_feat_list, _, _, _, _, = model(Variable(imgs).cuda())
        final_feat_list_flip, _, _, _, _ = model(Variable(imgs_flip).cuda())
        
        for j in range(len(final_feat_list)):
            if j == 0:
                outputs = (final_feat_list[j].cpu() + final_feat_list_flip[j].cpu())/2
            else:
                outputs = torch.cat((outputs, (final_feat_list[j].cpu() + final_feat_list_flip[j].cpu())/2), 1)

        outputs = F.normalize(outputs, p=2, dim=1)
        
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'.format(i + 1, len(test_loader),
                                                  batch_time.val, batch_time.avg,
                                                  data_time.val, data_time.avg))
            
print("Extracing features is finished... Now evaluating begins...")

#Evaluating distance matrix
distmat = evaluators.pairwise_distance(features, dataset.query, dataset.gallery)

evaluators.evaluate_all(distmat, dataset.query, dataset.gallery, dataset=args.dataset_type, top1=True)
