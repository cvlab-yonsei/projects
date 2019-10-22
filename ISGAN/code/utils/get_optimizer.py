from torch.optim import Adam, SGD
import numpy as np

from opt import opt

def get_optimizer(model):    
    if (opt.stage == 1) or (opt.stage == 2):
        param_groups = [{'params': model.C.parameters(), 'lr_mult': 1.0},
                        {'params': model.G.parameters(), 'lr_mult': 1.0}]
        optimizer = Adam(param_groups, lr=opt.lr, weight_decay=5e-4, amsgrad=True)
        optimizer_D = SGD(model.D.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
        
    if opt.stage == 3:
        id_dict1 = model.C.get_modules(model.C.id_dict1())
        id_dict2 = model.C.get_modules(model.C.id_dict2())
        id_modules = id_dict1 + id_dict2
        nid_dict1 = model.C.get_modules(model.C.nid_dict1())
        nid_dict2 = model.C.get_modules(model.C.nid_dict2())
        nid_modules = nid_dict1 + nid_dict2
        
        param_groups = []
        param_groups = [{'params': model.C.backbone.parameters(), 'lr_mult': 1.0},
                        {'params': model.G.parameters(), 'lr_mult': 0.01}]
        for i in range(np.shape(id_modules)[0]):
            param_groups.append({'params': id_modules[i].parameters(), 'lr_mult': 1.0})
        for i in range(np.shape(nid_modules)[0]):
            param_groups.append({'params': nid_modules[i].parameters(), 'lr_mult': 0.01})
            
        optimizer = Adam(param_groups, lr=opt.lr, weight_decay=5e-4, amsgrad=True)
        optimizer_D = SGD(model.D.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)

    return optimizer, optimizer_D