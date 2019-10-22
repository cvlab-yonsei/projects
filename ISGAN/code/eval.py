import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np

from opt import opt
# opt.data_path = '/home/chan/memory_256G/datasets/Market-1501-v15.09.15'
# opt.save_path = '/home/chan/memory_2T/weights_market1501_v4'
# opt.num_cls = 767
# opt.data_path = '/home/chan/memory_256G/datasets/cuhk03_release'
# opt.save_path = '/home/chan/memory_2T/weights_cuhk03_v4'
opt.num_cls = 702
opt.data_path = '/home/chan/memory_256G/datasets/DukeMTMC-reID'
opt.save_path = '/home/chan/memory_2T/weights_duke_v4'

from data import Data
from network import Model
from loss import Loss
from main0 import Main

data = Data()
model = Model()
loss = Loss(model)
main = Main(model, loss, data)

opt.name = '/imgr_sep_LD_LC_v0'
accr_save_path = opt.save_path + opt.name + '_accr.txt'
# main.load_model(opt.save_path + '/model_stage1_300.pt', 0)
main.load_model(opt.save_path + '/imgr_sep_LD_LC_v0_model_stage3_400.pt', 0)
main.evaluate(accr_save_path)
