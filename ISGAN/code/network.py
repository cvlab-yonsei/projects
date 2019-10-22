import copy
import functools
import torch
import torch.nn as nn
import numpy as np
from torchvision.models.resnet import resnet50, Bottleneck
from opt import opt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.C = Encoder()
        self.G = Generator()
        self.D = Discriminator()
                
def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
            nn.init.constant_(m.bias.data, 0.)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean=1., std=0.02)
            nn.init.constant_(m.bias.data, 0.0)
            
def init_weights(net):
    net.apply(weights_init_normal)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        resnet = resnet50(pretrained=True)
        
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3[0],) # conv4_1
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        
        res_g_conv5 = resnet.layer4
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())
        
        self.p0_id = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p1_id = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2_id = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        
        self.maxpool_zg_p0 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp1 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(8, 8))
        
        self.reduction_zg_p0_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_zg_p1_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z0_p1_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z1_p1_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_zg_p2_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z0_p2_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z1_p2_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z2_p2_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        
        self.fc_fg_p0_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_fg_p1_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f0_p1_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f1_p1_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_fg_p2_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f0_p2_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f1_p2_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f2_p2_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        
        self.p0_nid = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p1_nid = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2_nid = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        
        self.avgpool_zg_p0 = nn.AvgPool2d(kernel_size=(12, 4))
        self.avgpool_zg_p1 = nn.AvgPool2d(kernel_size=(24, 8))
        self.avgpool_zp1 = nn.AvgPool2d(kernel_size=(12, 8))
        self.avgpool_zg_p2 = nn.AvgPool2d(kernel_size=(24, 8))
        self.avgpool_zp2 = nn.AvgPool2d(kernel_size=(8, 8))
        
        self.fc_zg_p0_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p0_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p1_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p1_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z0_p1_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z0_p1_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z1_p1_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z1_p1_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p2_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p2_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z0_p2_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z0_p2_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z1_p2_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z1_p2_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z2_p2_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z2_p2_nid_lv = nn.Linear(2048, int(opt.feat_nid))
                
        id_dict2 = self.get_modules(self.id_dict2())
        for i in range(np.size(id_dict2)):
            init_weights(id_dict2[i])
                    
    def reparameterization(self, mu, lv):
        std = torch.exp(lv / 2)
        sampled_z = torch.FloatTensor(np.random.normal(0, 1, mu.size())).to(opt.device)
        return sampled_z * std + mu
    
    def id_dict1(self):
        return ['p0_id', 'p1_id', 'p2_id']
    def id_dict2(self):
        return ['reduction_zg_p0_id', 'reduction_zg_p1_id', 'reduction_zg_p2_id', 
                'reduction_z0_p1_id', 'reduction_z1_p1_id', 
                'reduction_z0_p2_id', 'reduction_z1_p2_id', 'reduction_z2_p2_id',
                'fc_fg_p0_id', 'fc_fg_p1_id', 'fc_fg_p2_id',
                'fc_f0_p1_id', 'fc_f1_p1_id',
                'fc_f0_p2_id', 'fc_f1_p2_id', 'fc_f2_p2_id']
    def nid_dict1(self):
        return ['p0_nid', 'p1_nid', 'p2_nid']
    def nid_dict2(self):
        return ['fc_zg_p0_nid_mu', 'fc_zg_p0_nid_lv', 
                'fc_zg_p1_nid_mu', 'fc_zg_p1_nid_lv', 'fc_zg_p2_nid_mu', 'fc_zg_p2_nid_lv',
                'fc_z0_p1_nid_mu', 'fc_z0_p1_nid_lv', 'fc_z1_p1_nid_mu', 'fc_z1_p1_nid_lv',
                'fc_z0_p2_nid_mu', 'fc_z0_p2_nid_lv', 'fc_z1_p2_nid_mu', 'fc_z1_p2_nid_lv', 
                'fc_z2_p2_nid_mu', 'fc_z2_p2_nid_lv']
    
    def get_modules(self, list):
        modules = []
        for name, module in self.named_children():
            if name in list:
                modules.append(module)
        return modules
        
    def forward(self, x):
        x = self.backbone(x)
        ##################################### identity-related #########################################
        p0_id = self.p0_id(x)
        p1_id = self.p1_id(x)
        p2_id = self.p2_id(x)
        
        zg_p0_id = self.maxpool_zg_p0(p0_id)
        zg_p1_id = self.maxpool_zg_p1(p1_id)
        zp1_id = self.maxpool_zp1(p1_id)
        z0_p1_id = zp1_id[:, :, 0:1, :]
        z1_p1_id = zp1_id[:, :, 1:2, :]
        zg_p2_id = self.maxpool_zg_p2(p2_id)
        zp2_id = self.maxpool_zp2(p2_id)
        z0_p2_id = zp2_id[:, :, 0:1, :]
        z1_p2_id = zp2_id[:, :, 1:2, :]
        z2_p2_id = zp2_id[:, :, 2:3, :]
        
        fg_p0_id = self.reduction_zg_p0_id(zg_p0_id).squeeze(dim=3).squeeze(dim=2)
        fg_p1_id = self.reduction_zg_p1_id(zg_p1_id).squeeze(dim=3).squeeze(dim=2)
        f0_p1_id = self.reduction_z0_p1_id(z0_p1_id).squeeze(dim=3).squeeze(dim=2)
        f1_p1_id = self.reduction_z1_p1_id(z1_p1_id).squeeze(dim=3).squeeze(dim=2)
        fg_p2_id = self.reduction_zg_p2_id(zg_p2_id).squeeze(dim=3).squeeze(dim=2)
        f0_p2_id = self.reduction_z0_p2_id(z0_p2_id).squeeze(dim=3).squeeze(dim=2)
        f1_p2_id = self.reduction_z1_p2_id(z1_p2_id).squeeze(dim=3).squeeze(dim=2)
        f2_p2_id = self.reduction_z2_p2_id(z2_p2_id).squeeze(dim=3).squeeze(dim=2)
        
        lg_p0 = self.fc_fg_p0_id(fg_p0_id)
        lg_p1 = self.fc_fg_p1_id(fg_p1_id)
        l0_p1 = self.fc_f0_p1_id(f0_p1_id)
        l1_p1 = self.fc_f1_p1_id(f1_p1_id)
        lg_p2 = self.fc_fg_p2_id(fg_p2_id)
        l0_p2 = self.fc_f0_p2_id(f0_p2_id)
        l1_p2 = self.fc_f1_p2_id(f1_p2_id)
        l2_p2 = self.fc_f2_p2_id(f2_p2_id)

        ###################################### identity-unrelated ########################################
        p0_nid = self.p0_nid(x)
        p1_nid = self.p1_nid(x)
        p2_nid = self.p2_nid(x)

        zg_p0_nid = self.avgpool_zg_p0(p0_nid)
        zg_p1_nid = self.avgpool_zg_p1(p1_nid)
        zp1_nid = self.avgpool_zp1(p1_nid)
        z0_p1_nid = zp1_nid[:, :, 0:1, :]
        z1_p1_nid = zp1_nid[:, :, 1:2, :]
        zg_p2_nid = self.avgpool_zg_p2(p2_nid)
        zp2_nid = self.avgpool_zp2(p2_nid)
        z0_p2_nid = zp2_nid[:, :, 0:1, :]
        z1_p2_nid = zp2_nid[:, :, 1:2, :]
        z2_p2_nid = zp2_nid[:, :, 2:3, :]
        
        fc_zg_p0_nid_mu = self.fc_zg_p0_nid_mu(zg_p0_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p0_nid_lv = self.fc_zg_p0_nid_lv(zg_p0_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p1_nid_mu = self.fc_zg_p1_nid_mu(zg_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p1_nid_lv = self.fc_zg_p1_nid_lv(zg_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z0_p1_nid_mu = self.fc_z0_p1_nid_mu(z0_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z0_p1_nid_lv = self.fc_z0_p1_nid_lv(z0_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z1_p1_nid_mu = self.fc_z1_p1_nid_mu(z1_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z1_p1_nid_lv = self.fc_z1_p1_nid_lv(z1_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p2_nid_mu = self.fc_zg_p2_nid_mu(zg_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p2_nid_lv = self.fc_zg_p2_nid_lv(zg_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z0_p2_nid_mu = self.fc_z0_p2_nid_mu(z0_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z0_p2_nid_lv = self.fc_z0_p2_nid_lv(z0_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z1_p2_nid_mu = self.fc_z1_p2_nid_mu(z1_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z1_p2_nid_lv = self.fc_z1_p2_nid_lv(z1_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z2_p2_nid_mu = self.fc_z2_p2_nid_mu(z2_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z2_p2_nid_lv = self.fc_z2_p2_nid_lv(z2_p2_nid.squeeze(dim=3).squeeze(dim=2))
        
        fc_zg_p0_nid = self.reparameterization(fc_zg_p0_nid_mu, fc_zg_p0_nid_lv)
        fc_zg_p1_nid = self.reparameterization(fc_zg_p1_nid_mu, fc_zg_p1_nid_lv)
        fc_z0_p1_nid = self.reparameterization(fc_z0_p1_nid_mu, fc_z0_p1_nid_lv)
        fc_z1_p1_nid = self.reparameterization(fc_z1_p1_nid_mu, fc_z1_p1_nid_lv)
        fc_zg_p2_nid = self.reparameterization(fc_zg_p2_nid_mu, fc_zg_p2_nid_lv)
        fc_z0_p2_nid = self.reparameterization(fc_z0_p2_nid_mu, fc_z0_p2_nid_lv)
        fc_z1_p2_nid = self.reparameterization(fc_z1_p2_nid_mu, fc_z1_p2_nid_lv)
        fc_z2_p2_nid = self.reparameterization(fc_z2_p2_nid_mu, fc_z2_p2_nid_lv)
        
        list_mu = [fc_zg_p0_nid_mu, fc_zg_p1_nid_mu, fc_z0_p1_nid_mu, fc_z1_p1_nid_mu, 
                   fc_zg_p2_nid_mu, fc_z0_p2_nid_mu, fc_z1_p2_nid_mu, fc_z2_p2_nid_mu]
        list_lv = [fc_zg_p0_nid_lv, fc_zg_p1_nid_lv, fc_z0_p1_nid_lv, fc_z1_p1_nid_lv, 
                   fc_zg_p2_nid_lv, fc_z0_p2_nid_lv, fc_z1_p2_nid_lv, fc_z2_p2_nid_lv]
        
        id = torch.cat(
            [fg_p0_id, fg_p1_id, f0_p1_id, f1_p1_id, fg_p2_id, f0_p2_id, f1_p2_id, f2_p2_id], dim=1)
        nid = torch.cat(
            [fc_zg_p0_nid, fc_zg_p1_nid, fc_z0_p1_nid, fc_z1_p1_nid, 
             fc_zg_p2_nid, fc_z0_p2_nid, fc_z1_p2_nid, fc_z2_p2_nid], dim=1)
        
        return id, lg_p0, lg_p1, l0_p1, l1_p1, lg_p2, l0_p2, l1_p2, l2_p2, list_mu, list_lv, nid
    
class Generator(nn.Module):
    def __init__(self, output_dim=3):
        super(Generator, self).__init__()
        
        self.G_fc = nn.Sequential(
            nn.Linear(opt.feat_id*8 + opt.feat_nid*8 + opt.feat_niz + opt.num_cls, opt.feat_G*8),
            nn.BatchNorm1d(opt.feat_G*8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout))
        
        self.G_deconv = nn.Sequential(
            # 1st block
            nn.ConvTranspose2d(opt.feat_G*8, opt.feat_G*8, kernel_size=(6,2),bias=False),
            nn.BatchNorm2d(opt.feat_G*8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 2nd block
            nn.ConvTranspose2d(opt.feat_G*8, opt.feat_G*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G*8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 3rd block
            nn.ConvTranspose2d(opt.feat_G*8, opt.feat_G*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G*8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 4th block
            nn.ConvTranspose2d(opt.feat_G*8, opt.feat_G*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G*4),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 5th block
            nn.ConvTranspose2d(opt.feat_G*4, opt.feat_G*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G*2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 6th block
            nn.ConvTranspose2d(opt.feat_G*2, opt.feat_G*1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G*1),
            nn.LeakyReLU(0.2, True),
            # 7th block
            nn.ConvTranspose2d(opt.feat_G*1, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())
        init_weights(self.G_fc)
        init_weights(self.G_deconv)

    def forward(self, inputs, labels):
        x = torch.cat([inputs, labels], 1)
        x = self.G_fc(x).view(-1, opt.feat_G*8, 1, 1)
        x = self.G_deconv(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        n_layers = 5
        kw = 4
        padw = 1
        backbone = [
            nn.Tanh(),
            nn.Conv2d(3, opt.feat_D, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            backbone += [
                nn.Conv2d(
                    opt.feat_D*nf_mult_prev, opt.feat_D*nf_mult, kernel_size=kw, 
                    stride=2, padding=padw, bias=use_bias),
                norm_layer(opt.feat_D * nf_mult),
                nn.LeakyReLU(0.2, True),]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        image_D = [
            nn.Conv2d(
                opt.feat_D*nf_mult_prev, opt.feat_D*nf_mult, kernel_size=kw, 
                stride=1, padding=padw, bias=use_bias),
            norm_layer(opt.feat_D*nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(opt.feat_D*nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            nn.Sigmoid()]
        
        label_D1 = [
            nn.Conv2d(
                opt.feat_D*nf_mult_prev, opt.feat_D*nf_mult, kernel_size=kw, 
                stride=1, padding=padw, bias=use_bias),
            norm_layer(opt.feat_D*nf_mult),]
        self.avgp = nn.AvgPool2d(kernel_size=(12, 4))
        label_D2 = [nn.Linear(opt.feat_D*nf_mult, int(opt.num_cls))]
        
        self.backbone = nn.Sequential(*backbone)
        self.image_D = nn.Sequential(*image_D)
        self.label_D1 = nn.Sequential(*label_D1)
        self.label_D2 = nn.Sequential(*label_D2)

    def forward(self, input):
        backbone = self.backbone(input)
        image_D = self.image_D(backbone)
        label_D1 = self.label_D1(backbone)
        avgp = self.avgp(label_D1).squeeze(dim=3).squeeze(dim=2)
        label_D2 = self.label_D2(avgp)
        return image_D, label_D2