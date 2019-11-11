import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
import torchvision.utils as vutils
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network import Model
from loss import Loss
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking

class Main():
    def __init__(self, model, loss, data):
        if opt.stage == 1:
            self.train_loader = data.train_loader
        else:
            self.train_loader = data.train_loader_woEr
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to(opt.device)
        self.loss = loss
        self.data = data
                
        self.scheduler = lr_scheduler.MultiStepLR(loss.optimizer, milestones=opt.lr_scheduler, gamma=0.1)
        self.scheduler_D = lr_scheduler.MultiStepLR(loss.optimizer_D, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):

        self.scheduler.step()
        self.scheduler_D.step()
        self.model.train()
                
        for batch, (inputs, labels) in enumerate(self.train_loader):
            if inputs.size()[0] != opt.batchid * opt.batchimage: continue
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)
            
            if opt.stage == 1:
                self.loss.optimizer.zero_grad()
                loss = self.loss(inputs, labels, batch)
                loss.backward()
                self.loss.optimizer.step()
                
            elif (opt.stage == 2) or (opt.stage == 3):
                self.loss.optimizer_D.zero_grad()
                self.loss.optimizer.zero_grad()
                G_loss = self.loss(inputs, labels, batch)
                G_loss.backward()
                self.loss.optimizer.step()
                                                                
    def evaluate(self, save_path):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(
                dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

#         #########################   re rank##########################
#         q_g_dist = np.dot(qf, np.transpose(gf))
#         q_q_dist = np.dot(qf, np.transpose(qf))
#         g_g_dist = np.dot(gf, np.transpose(gf))
#         dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

#         r, m_ap = rank(dist)

#         print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
#               .format(m_ap, r[0], r[2], r[4], r[9]))

#         #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
        
        with open(save_path, 'a') as f:
            f.write(
                '[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}\n'
                .format(m_ap, r[0], r[2], r[4], r[9]))
        
    def save_model(self, save_path):
        torch.save({
            'model_C': self.model.C.state_dict(),
            'model_G': self.model.G.state_dict(),
            'model_D': self.model.D.state_dict(),
            'optimizer' : self.loss.optimizer.state_dict(),
            'optimizer_D' : self.loss.optimizer_D.state_dict(),
        }, save_path)
        
    def load_model(self, load_path, last_epoch):
        checkpoint = torch.load(load_path)
        self.model.C.load_state_dict(checkpoint['model_C'], strict=False)
        if opt.stage == 3:
            self.model.G.load_state_dict(checkpoint['model_G'])
            self.model.D.load_state_dict(checkpoint['model_D'])
            self.loss.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.scheduler.last_epoch = last_epoch
        self.scheduler_D.last_epoch = last_epoch

if __name__ == '__main__':
    
    data = Data()
    model = Model()
    loss = Loss(model)
    main = Main(model, loss, data)

    if opt.mode == 'train':
        os.makedirs(opt.save_path, exist_ok=True)
        
        if opt.stage == 1:
            opt.start = 0
            opt.epoch = 300
        
        if opt.stage == 2:
            main.load_model(opt.save_path + '/isgan_stage1_300.pt', 0)
            opt.start = 0
            opt.epoch = 200
        
        if opt.stage == 3:
            main.load_model(opt.save_path + '/isgan_stage2_200.pt', 300)
            opt.start = 300
            opt.epoch = 400
                
        for epoch in range(opt.start+1, opt.epoch+1):
            
            print('\nepoch', epoch)
            main.train()
            
            if epoch % 50 == 0:
                os.makedirs(opt.save_path, exist_ok=True)
                weight_save_path = opt.save_path + opt.name + \
                                        '_stage{}_{:03d}.pt'.format(opt.stage, epoch)
                main.save_model(weight_save_path)

    if opt.mode == 'evaluate':
        print('start evaluate')
        main.load_model(opt.save_path + opt.weight, 0)
        main.evaluate(opt.save_path + opt.name + '_accr.txt')
