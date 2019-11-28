import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import os
    
class Pascal_Seg_Synth(Dataset):
    def __init__(self, image_path, mask_path, feature_H, feature_W):
        
        self.feature_H = feature_H # height of feature volume
        self.feature_W = feature_W # width of feature volume
        
        self.image_H = self.feature_H * 16
        self.image_W = self.feature_W * 16

        self.image_transform1 = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2)])

        self.image_transform2 = transforms.Compose([transforms.transforms.ColorJitter(brightness=0.1,contrast=0.1, saturation=0.1, hue=0.1),
                                                    transforms.ToTensor()])

        self.mask_transform1 = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2), 
                                                   transforms.ToTensor()])

        self.mask_transform2 = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((feature_H,feature_W)),
                                                   transforms.ToTensor()])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.IMAGE_DATA = np.load(image_path,allow_pickle=True)
        self.MASK_DATA = np.load(mask_path,allow_pickle=True)

    
    def affine_transform(self, x, theta):
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
        
    def __getitem__(self, index):
        image = self.IMAGE_DATA[index]
        image = Image.fromarray(image.astype('uint8'))
        mask = self.MASK_DATA[index]
        mask = Image.fromarray(mask.astype('uint8'))

        p = np.random.uniform()
        if p < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask) # pair filp

        image = self.image_transform1(image) # resize
        image1 = self.image_transform2(image).unsqueeze(0) # jitter -> image1
        image2 = self.image_transform2(image).unsqueeze(0) # jitter -> image2
        mask = self.mask_transform1(mask).unsqueeze(0) # resize
        
        # generate source image/mask
        theta1 = np.zeros(9)
        theta1[0:6] = np.random.randn(6) * 0.15
        theta1 = theta1 + np.array([1,0,0,0,1,0,0,0,1])
        affine1 = np.reshape(theta1, (3,3))
        affine_inverse1 = np.linalg.inv(affine1)
        affine1 = np.reshape(affine1, -1)[0:6]
        affine_inverse1 = np.reshape(affine_inverse1, -1)[0:6]
        affine1 = torch.from_numpy(affine1).type(torch.FloatTensor)
        affine_inverse1 = torch.from_numpy(affine_inverse1).type(torch.FloatTensor)

        image1 = self.affine_transform(image1,affine1) # source image
        
        mask = self.affine_transform(mask,affine1)
        mask = self.affine_transform(mask, affine_inverse1) # convert truncated pixels to 0

        # generate target image/mask
        theta2 = np.zeros(9)
        theta2[0:6] = np.random.randn(6) * 0.15
        theta2 = theta2 + np.array([1,0,0,0,1,0,0,0,1])
        affine2 = np.reshape(theta2, (3,3))
        affine_inverse2 = np.linalg.inv(affine2)
        affine2 = np.reshape(affine2, -1)[0:6]
        affine_inverse2 = np.reshape(affine_inverse2, -1)[0:6]
        affine2 = torch.from_numpy(affine2).type(torch.FloatTensor)
        affine_inverse2 = torch.from_numpy(affine_inverse2).type(torch.FloatTensor)

        image2 = self.affine_transform(image2,affine2) # target image
        mask2 = self.affine_transform(mask,affine2) # target mask

        mask = self.affine_transform(mask2, affine_inverse2)
        mask1 = self.affine_transform(mask, affine1) # source mask : convert truncated pixels to 0

        image1, image2, mask1, mask2 = image1.squeeze(0).data, image2.squeeze(0).data, mask1.squeeze(0).data, mask2.squeeze(0).data
        
        mask1 = self.mask_transform2(mask1) # resize
        mask2 = self.mask_transform2(mask2) # resize
        mask1 = (mask1>0.1).float() # binarize
        mask2 = (mask2>0.1).float() # binarize
            
        # Return image and the label
        return {'image1_rgb':image1.clone(),'image2_rgb':image2.clone(),'image1':self.normalize(image1),
                'image2':self.normalize(image2),'mask1':mask1, 'mask2':mask2}
    
    def __len__(self):
        return len(self.IMAGE_DATA)

# some parts of codes are from 'https://github.com/ignacio-rocco/weakalign'
class PF_Pascal(Dataset):
    def __init__(self, csv_path, image_path, feature_H, feature_W, eval_type='image_size'):
        self.feature_H = feature_H
        self.feature_W = feature_W
        
        self.image_H = (self.feature_H-2) * 16
        self.image_W = (self.feature_W-2) * 16
        
        self.data_info = pd.read_csv(csv_path)
        
        self.transform = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2),
                                              transforms.Pad(16), # pad zeros around borders to avoid boundary artifacts
                                              transforms.ToTensor()])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_A_names = self.data_info.iloc[:, 0]
        self.image_B_names = self.data_info.iloc[:, 1]
        self.class_num = self.data_info.iloc[:, 2]
        self.point_A_coords = self.data_info.iloc[:, 3:5]
        self.point_B_coords = self.data_info.iloc[:, 5:7]        
        self.L_pck = self.data_info.iloc[:,7].values.astype('float') # L_pck of source
        self.image_path = image_path
        self.eval_type = eval_type

    def get_image(self, image_name_list, idx):
        image_name = os.path.join(self.image_path, image_name_list[idx])
        image = Image.open(image_name)
        width, height = image.size
        return image, torch.FloatTensor([height, width])

    def get_points(self, point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=';')
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=';')
        point_coords = np.concatenate((X.reshape(1, len(X)), Y.reshape(1, len(Y))), axis=0)
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords


    def __getitem__(self, idx):
        # get pre-processed images
        image1, image1_size = self.get_image(self.image_A_names, idx)
        image2, image2_size = self.get_image(self.image_B_names, idx)
        class_num = int(self.class_num[idx])-1
        image1_var = self.transform(image1)
        image2_var = self.transform(image2)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords, idx)
        point_B_coords = self.get_points(self.point_B_coords, idx)
        # compute PCK reference length L_pck (equal to max bounding box side in image_B)
        if self.eval_type == 'bounding_box':
            # L_pck = torch.FloatTensor([torch.max(point_B_coords.max(1)[0] - point_B_coords.min(1)[0])]) # for PF WILLOW
            L_pck = torch.FloatTensor(np.fromstring(self.L_pck[idx]).astype(np.float32)) # max(h,w), where h&w are height&width of bounding-box provided by Pascal dataset
        elif self.eval_type == 'image_size':
            N_pts = torch.sum(torch.ne(point_A_coords[0,:],-1))
            point_A_coords[0,0:N_pts] = point_A_coords[0,0:N_pts] * self.image_W / image1_size[1] # rescale x coord.
            point_A_coords[1,0:N_pts] = point_A_coords[1,0:N_pts] * self.image_H / image1_size[0] # rescale y coord.
            point_B_coords[0,0:N_pts] = point_B_coords[0,0:N_pts] * self.image_W / image2_size[1] # rescale x coord.
            point_B_coords[1,0:N_pts] = point_B_coords[1,0:N_pts] * self.image_H / image2_size[0] # rescale y coord.
            image1_size = torch.FloatTensor([self.image_H,self.image_W])
            image2_size = torch.FloatTensor([self.image_H,self.image_W])
            L_pck = torch.FloatTensor([self.image_H]) if self.image_H >= self.image_W else torch.FloatTensor([self.image_W])
        else:
            raise ValueError('Invalid eval_type')

        return {'image1_rgb': transforms.ToTensor()(image1), 'image2_rgb': transforms.ToTensor()(image2),
                'image1': self.normalize(image1_var), 'image2': self.normalize(image2_var),
                'image1_points': point_A_coords, 'image2_points': point_B_coords, 'L_pck': L_pck,
                'image1_size': image1_size, 'image2_size': image2_size, 'class_num':class_num}

    def __len__(self):
        return len(self.data_info.index)

# some parts of codes are from 'https://github.com/ignacio-rocco/weakalign'
class PF_WILLOW(Dataset):
    def __init__(self, csv_path, image_path, feature_H, feature_W, eval_type='bounding_box'):
        self.feature_H = feature_H
        self.feature_W = feature_W
        
        self.image_H = (self.feature_H-2) * 16
        self.image_W = (self.feature_W-2) * 16
        
        self.data_info = pd.read_csv(csv_path)
        
        self.transform = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2),
                                              transforms.Pad(16), # pad zeros around borders to avoid boundary artifacts
                                              transforms.ToTensor()])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_A_names = self.data_info.iloc[:,0]
        self.image_B_names = self.data_info.iloc[:,1]
        self.point_A_coords = self.data_info.iloc[:, 2:22].values.astype('float')
        self.point_B_coords = self.data_info.iloc[:, 22:].values.astype('float')
        self.L_pck = self.data_info.iloc[:,7].values.astype('float') # L_pck of source
        self.image_path = image_path
        self.eval_type = eval_type

    def get_image(self, image_name_list, idx):
        image_name = os.path.join(self.image_path, image_name_list[idx])
        image = Image.open(image_name)
        width, height = image.size
        return image, torch.FloatTensor([height, width])

    def get_points(self,point_coords_list,idx):
        point_coords = point_coords_list[idx, :].reshape(2,10)
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords


    def __getitem__(self, idx):
        # get pre-processed images
        image1, image1_size = self.get_image(self.image_A_names, idx)
        image2, image2_size = self.get_image(self.image_B_names, idx)
        image1_var = self.transform(image1)
        image2_var = self.transform(image2)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords, idx)
        point_B_coords = self.get_points(self.point_B_coords, idx)
        # compute PCK reference length L_pck (equal to max bounding box side in image_B)
        if self.eval_type == 'bounding_box':
            L_pck = torch.FloatTensor([torch.max(point_A_coords.max(1)[0] - point_A_coords.min(1)[0])]) # for PF WILLOW
        elif self.eval_type == 'image_size':
            N_pts = torch.sum(torch.ne(point_A_coords[0,:],-1))
            point_A_coords[0,0:N_pts] = point_A_coords[0,0:N_pts] * self.image_W / image1_size[1] # rescale x coord.
            point_A_coords[1,0:N_pts] = point_A_coords[1,0:N_pts] * self.image_H / image1_size[0] # rescale y coord.
            point_B_coords[0,0:N_pts] = point_B_coords[0,0:N_pts] * self.image_W / image2_size[1] # rescale x coord.
            point_B_coords[1,0:N_pts] = point_B_coords[1,0:N_pts] * self.image_H / image2_size[0] # rescale y coord.
            image1_size = torch.FloatTensor([self.image_H,self.image_W])
            image2_size = torch.FloatTensor([self.image_H,self.image_W])
            L_pck = torch.FloatTensor([self.image_H]) if self.image_H >= self.image_W else torch.FloatTensor([self.image_W])
        else:
            raise ValueError('Invalid eval_type')

        return {'image1_rgb': transforms.ToTensor()(image1), 'image2_rgb': transforms.ToTensor()(image2),
                'image1': self.normalize(image1_var), 'image2': self.normalize(image2_var),
                'image1_points': point_A_coords, 'image2_points': point_B_coords, 'L_pck': L_pck,
                'image1_size': image1_size, 'image2_size': image2_size}

    def __len__(self):
        return len(self.data_info.index)