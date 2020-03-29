from torch.nn import CrossEntropyLoss, BCELoss, L1Loss, Tanh
from torch.nn.modules import loss
from utils.get_optimizer import get_optimizer
import torch
from torch.distributions import normal
import numpy as np
import copy

from opt import opt

batch_size = opt.batchid * opt.batchimage
num_gran = 8

class Loss(loss._Loss):
    def __init__(self, model):
        super(Loss, self).__init__()
        
        self.tanh = Tanh()
        self.l1_loss = L1Loss()
        self.bce_loss = BCELoss()
        self.cross_entropy_loss = CrossEntropyLoss()
        
        self.model = model
        self.optimizer, self.optimizer_D = get_optimizer(model)
        
    def get_positive_pairs(self):
        idx=[]
        for i in range(batch_size):
            r = i
            while r == i:
                r = int(torch.randint(
                        low=opt.batchid*(i//opt.batchid), high=opt.batchid*(i//opt.batchid+1),
                        size=(1,)).item())
            idx.append(r)
        return idx
    
    def region_wise_shuffle(self, id, ps_idx):
        sep_id = id.clone()
        idx = torch.tensor([0]*(num_gran))
        while (torch.sum(idx)==0) and (torch.sum(idx)==num_gran):
            idx = torch.randint(high=2, size=(num_gran,))
        
        for i in range(num_gran):
            if idx[i]:
                sep_id[:, opt.feat_id*i:opt.feat_id*(i+1)] = id[ps_idx][:, opt.feat_id*i:opt.feat_id*(i+1)]
        return sep_id
    
    def get_noise(self):
        return torch.randn(batch_size, opt.feat_niz, device=opt.device)
    
    def make_onehot(self, label):
        onehot_vec = torch.zeros(batch_size, opt.num_cls)
        for i in range(label.size()[0]):
            onehot_vec[i, label[i]] = 1
        return onehot_vec
    
    def set_parameter(self, m, train=True):
        if train:
            for param in m.parameters():
                param.requires_grad = True
            m.apply(self.set_bn_to_train)
        else:
            for param in m.parameters():
                param.requires_grad = False
            m.apply(self.set_bn_to_eval)
    
    def set_bn_to_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.eval()
        
    def set_bn_to_train(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.train()
    
    def set_model(self):
        self.model.C.zero_grad()
        self.model.G.zero_grad()
        self.model.D.zero_grad()
        
        if opt.stage == 1:
            self.set_parameter(self.model.C, train=True)
            nid_dict1 = self.model.C.get_modules(self.model.C.nid_dict1())
            nid_dict2 = self.model.C.get_modules(self.model.C.nid_dict2())
            for i in range(np.shape(nid_dict1)[0]):
                self.set_parameter(nid_dict1[i], train=False)
            for i in range(np.shape(nid_dict2)[0]):
                self.set_parameter(nid_dict2[i], train=False)
            self.set_parameter(self.model.G, train=False)
            self.set_parameter(self.model.D, train=False)
            
        elif opt.stage == 2:
            self.set_parameter(self.model.C, train=False)
            nid_dict1 = self.model.C.get_modules(self.model.C.nid_dict1())
            nid_dict2 = self.model.C.get_modules(self.model.C.nid_dict2())
            for i in range(np.shape(nid_dict1)[0]):
                self.set_parameter(nid_dict1[i], train=True)
            for i in range(np.shape(nid_dict2)[0]):
                self.set_parameter(nid_dict2[i], train=True)
            self.set_parameter(self.model.G, train=True)
            self.set_parameter(self.model.D, train=True)
    
    def id_related_loss(self, labels, outputs):
        CrossEntropy_Loss = [self.cross_entropy_loss(output, labels) for output in outputs[1:1+num_gran]]
        return sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)
    
    def KL_loss(self, outputs):
        list_mu = outputs[-3]
        list_lv = outputs[-2]
        loss_KL = 0.
        for i in range(np.size(list_mu)):
            loss_KL += torch.sum(0.5 * (list_mu[i]**2 + torch.exp(list_lv[i]) - list_lv[i] - 1))
        return loss_KL/np.size(list_mu)
    
    def GAN_loss(self, inputs, outputs, labels):
        id = outputs[0]
        nid = outputs[-1]
        one_hot_labels = self.make_onehot(labels).to(opt.device)
        
        # Auto Encoder
        auto_G_in = torch.cat((id, nid, self.get_noise()), dim=1)
        auto_G_out = self.model.G.forward(auto_G_in, one_hot_labels)
        
        # Positive Shuffle
        ps_idx = self.get_positive_pairs()
        ps_G_in = torch.cat((id[ps_idx], nid, self.get_noise()), dim=1)
        ps_G_out = self.model.G.forward(ps_G_in, one_hot_labels)
        
        # Separate Positive Shuffle
        sep_id = self.region_wise_shuffle(id, ps_idx)
        sep_G_in = torch.cat((sep_id, nid, self.get_noise()), dim=1)
        sep_G_out = self.model.G.forward(sep_G_in, one_hot_labels)
        
        ############################################## D_loss ############################################
        D_real, C_real = self.model.D(inputs)
        REAL_LABEL = torch.FloatTensor(D_real.size()).uniform_(0.7, 1.0).to(opt.device)
        D_real_loss = self.bce_loss(D_real, REAL_LABEL)
        C_real_loss = self.cross_entropy_loss(C_real, labels)
                
        auto_D_fake, auto_C_fake = self.model.D(auto_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(auto_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        auto_D_fake_loss = self.bce_loss(auto_D_fake, FAKE_LABEL)
        auto_C_fake_loss = self.cross_entropy_loss(auto_C_fake, labels)
        
        ps_D_fake, ps_C_fake = self.model.D(ps_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(ps_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        ps_D_fake_loss = self.bce_loss(ps_D_fake, FAKE_LABEL)
        ps_C_fake_loss = self.cross_entropy_loss(ps_C_fake, labels)
        
        sep_D_fake, sep_C_fake = self.model.D(sep_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(sep_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        sep_D_fake_loss = self.bce_loss(sep_D_fake, FAKE_LABEL)
        sep_C_fake_loss = self.cross_entropy_loss(sep_C_fake, labels)
                
        D_x = D_real.mean()
        C_x = C_real_loss

        D_loss = (D_real_loss + auto_D_fake_loss + ps_D_fake_loss + sep_D_fake_loss) + \
                    (C_real_loss + auto_C_fake_loss + ps_C_fake_loss + sep_C_fake_loss)/2
        D_loss.backward()
        self.optimizer_D.step()
        
        ############################################## G_loss ##############################################
        auto_D_fake, auto_C_fake = self.model.D(auto_G_out)
        REAL_LABEL = torch.ones_like(auto_D_fake)
        auto_D_fake_loss = self.bce_loss(auto_D_fake, REAL_LABEL)
        auto_C_fake_loss = self.cross_entropy_loss(auto_C_fake, labels)
        
        ps_D_fake, ps_C_fake = self.model.D(ps_G_out)
        REAL_LABEL = torch.ones_like(ps_D_fake)
        ps_D_fake_loss = self.bce_loss(ps_D_fake, REAL_LABEL)
        ps_C_fake_loss = self.cross_entropy_loss(ps_C_fake, labels)
        
        sep_D_fake, sep_C_fake = self.model.D(sep_G_out)
        REAL_LABEL = torch.ones_like(sep_D_fake)
        sep_D_fake_loss = self.bce_loss(sep_D_fake, REAL_LABEL)
        sep_C_fake_loss = self.cross_entropy_loss(sep_C_fake, labels)
                    
        auto_imgr_loss = self.l1_loss(auto_G_out, self.tanh(inputs))
        ps_imgr_loss = self.l1_loss(ps_G_out, self.tanh(inputs))
        sep_imgr_loss = self.l1_loss(sep_G_out, self.tanh(inputs))
        
        G_loss = (auto_D_fake_loss + ps_D_fake_loss + sep_D_fake_loss) + \
                    (auto_C_fake_loss + ps_C_fake_loss + sep_C_fake_loss)*2 + \
                    (auto_imgr_loss + ps_imgr_loss + sep_imgr_loss)*10
        ############################################################################################
        return D_loss, G_loss, auto_imgr_loss, ps_imgr_loss, sep_imgr_loss

    def forward(self, inputs, labels, batch):
        self.set_model()
        outputs = self.model.C(inputs)
                
        if opt.stage == 1:
            CrossEntropy_Loss = self.id_related_loss(labels, outputs)
            loss_sum = CrossEntropy_Loss
            
            print('\rCE:%.2f' % (CrossEntropy_Loss.data.cpu().numpy()), end=' ')
                    
        elif opt.stage == 2:
            D_loss, G_loss, auto_imgr_loss, ps_imgr_loss, sep_imgr_loss\
                    = self.GAN_loss(inputs, outputs, labels)
            KL_loss = self.KL_loss(outputs)
            
            loss_sum = G_loss + KL_loss/1000
                        
            print('\rD_loss:%.2f  G_loss:%.2f A_ImgR:%.2f  PS_ImgR:%.2f  Sep_PS:%.2f  KL:%.2f' % (
                D_loss.data.cpu().numpy(),
                G_loss.data.cpu().numpy(),
                auto_imgr_loss.data.cpu().numpy(),
                ps_imgr_loss.data.cpu().numpy(),
                sep_imgr_loss.data.cpu().numpy(),
                KL_loss.data.cpu().numpy()), end=' ')
                    
        elif opt.stage == 3:
            CrossEntropy_Loss = self.id_related_loss(labels, outputs)
            D_loss, G_loss, auto_imgr_loss, ps_imgr_loss, sep_imgr_loss\
                    = self.GAN_loss(inputs, outputs, labels)
            KL_loss = self.KL_loss(outputs)
                        
            loss_sum = (CrossEntropy_Loss*2)*10 + G_loss + KL_loss/100
            
            print('\rCE:%.2f  D_loss:%.2f  G_loss:%.2f  A_ImgR:%.2f  PS_ImgR:%.2f  Sep_PS:%.2f  KL:%.2f' % (
                CrossEntropy_Loss.data.cpu().numpy(),
                D_loss.data.cpu().numpy(),
                G_loss.data.cpu().numpy(),
                auto_imgr_loss.data.cpu().numpy(),
                ps_imgr_loss.data.cpu().numpy(),
                sep_imgr_loss.data.cpu().numpy(),
                KL_loss.data.cpu().numpy()), end=' ')
            
        return loss_sum
