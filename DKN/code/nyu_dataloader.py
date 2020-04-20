import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils


class NYU_v2_datset(Dataset):
    """NYUDataset."""

    def __init__(self, root_dir, scale=8, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            self.images = np.load('%s/train_images_split.npy'%root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            self.images = np.load('%s/test_images_v2.npy'%root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        
        h, w = depth.shape
        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w, h), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth}
        
        return sample
    
        """
        return:
            sample:
            guidance (np.array float): H x W x 3 
            target ((np.array float)): H x W x 1
            gt ((np.array float)): H x W x 1
            
        """