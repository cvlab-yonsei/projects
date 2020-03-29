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

from utils import get_data, adjust_lr_staircase
from triplet import TripletSemihardLoss
from Relation_final_ver_last_multi_scale_large_losses import RelationModel as Model
from reid.utils.meters import AverageMeter
from reid.evaluation_metrics import accuracy

import argparse


parser = argparse.ArgumentParser(description="RRID")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--epochs', type=int, default=80, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('--base_lr', type=float, default=1e-3, help='initial learning rate for resnet50')
parser.add_argument('--decay_schedule', nargs='+', type=int, help='learning rate decaying schedule')
parser.add_argument('--staircase_decay_multiply_factor', type=float, default=0.1, help='decaying coefficient')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--split', type=int, default=0, help='split')
parser.add_argument('--batch_sample', type=int, default=4, help='same ids in batch')
parser.add_argument('--h', type=int, default=384, help='height of input images')
parser.add_argument('--w', type=int, default=128, help='width of input images')
parser.add_argument('--margin', type=float, default=1.2, help='margin of triplet loss')
parser.add_argument('--dataset_type', type=str, default='market1501', help='type of dataset: market1501, dukemtmc, cuhk03')
parser.add_argument('--dataset_path', type=str, default='./datasets/', help='directory of data')
parser.add_argument('--combine_trainval', action="store_true", default=False, help='select train or trainval')
parser.add_argument('--steps_per_log', type=int, default=100, help='frequency of printing')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

if args.decay_schedule is None:
    decay_schedule = (40, 60)
else:
    decay_schedule = tuple(args.decay_schedule)

log_directory = os.path.join(args.exp_dir + '_' + args.dataset_type)

if not os.path.exists(log_directory):
    os.makedirs(log_directory)

np_ratio = args.batch_sample - 1

# Dataset Loader
dataset, train_loader, val_loader, _ = get_data(args.dataset_type, args.split, args.dataset_path, args.h, args.w, 
                                                args.batch_size, args.num_workers, args.combine_trainval, 
                                                np_ratio)

# Model (RRID)
model = Model(last_conv_stride=1, num_stripes=6, local_conv_out_channels=256, num_classes=dataset.num_trainval_ids)

# If single gpu
if len(gpus) < 2 :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

# Losses and optimizer
cross_entropy_loss = CrossEntropyLoss()
triplet_loss = TripletSemihardLoss(margin=args.margin)

finetuned_params = list(model.base.parameters())

new_params = [p for n, p in model.named_parameters()
              if not n.startswith('base.')]

param_groups = [{'params': finetuned_params, 'lr': args.lr * 0.1},
                {'params': new_params, 'lr': args.lr}]

optimizer = optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)

# If multi gpus
if len(gpus) > 1:
    model = torch.nn.DataParallel(model, range(len(args.gpus))).cuda()

# Training
for epoch in range(1, args.epochs+1):
    
    adjust_lr_staircase(
        optimizer.param_groups,
        [args.base_lr, args.lr],
        epoch,
        decay_schedule,
        args.staircase_decay_multiply_factor)
    
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precisions = AverageMeter()
    
    end = time.time()
    for i, inputs in enumerate(train_loader):
        data_time.update(time.time() - end)

        (imgs, _, labels, _) = inputs
        inputs = Variable(imgs).float().cuda()
        labels = Variable(labels).cuda()


        optimizer.zero_grad()
        final_feat_list, logits_local_rest_list, logits_local_list, logits_rest_list, logits_global_list = model(inputs)

        T_loss = torch.sum(
            torch.stack([triplet_loss(output, labels) for output in final_feat_list]), dim=0)

        C_loss_local = torch.sum(
            torch.stack([cross_entropy_loss(output, labels) for output in logits_local_list]), dim=0)

        C_loss_local_rest = torch.sum(
            torch.stack([cross_entropy_loss(output, labels) for output in logits_local_rest_list]), dim=0)

        C_loss_rest = torch.sum(
            torch.stack([cross_entropy_loss(output, labels) for output in logits_rest_list]), dim=0)

        C_loss_global = torch.sum(
            torch.stack([cross_entropy_loss(output, labels) for output in logits_global_list]), dim=0)

        C_loss = C_loss_local_rest + C_loss_global + C_loss_local + C_loss_rest

        loss = T_loss + 2 * C_loss
        
        losses.update(loss.data.item(), labels.size(0))
        prec1 = (sum([accuracy(output.data, labels.data)[0].item() for output in logits_local_rest_list])
                 + sum([accuracy(output.data, labels.data)[0].item() for output in logits_global_list])
                 + sum([accuracy(output.data, labels.data)[0].item() for output in logits_local_list])
                 + sum([accuracy(output.data, labels.data)[0].item() for output in logits_rest_list]))/(12+12+12+9)
        precisions.update(prec1, labels.size(0))

        loss.backward()

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.steps_per_log == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_loader),
                              batch_time.val, args.steps_per_log*batch_time.avg,
                              data_time.val, args.steps_per_log*data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
                
    

torch.save(model, os.path.join(log_directory, 'model.pth'))
