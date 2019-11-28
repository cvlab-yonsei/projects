import torch
import torch.nn.functional as F
import numpy as np
import os
import random
from custom_dataset import PF_WILLOW
from model import SFNet
#import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="SFNet evaluation")
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--feature_h', type=int, default=20, help='height of feature volume')
parser.add_argument('--feature_w', type=int, default=20, help='width of feature volume')
parser.add_argument('--test_csv_path', type=str, default='./data/PF_WILLOW/test_pairs_pf.csv', help='directory of test csv file')
parser.add_argument('--test_image_path', type=str, default='./data/PF_WILLOW/', help='directory of test data')
parser.add_argument('--beta', type=float, default=50, help='inverse temperature of softmax @ kernel soft argmax')
parser.add_argument('--kernel_sigma', type=float, default=5, help='standard deviation of Gaussian kerenl @ kernel soft argmax')
parser.add_argument('--eval_type', type=str, default='bounding_box', choices=('bounding_box','image_size'), help='evaluation type for PCK threshold (bounding box | image size)')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Data Loader
print("Instantiate dataloader")
test_dataset = PF_WILLOW(args.test_csv_path, args.test_image_path, args.feature_h, args.feature_w, args.eval_type)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=1,
                                           shuffle=False, num_workers = args.num_workers)


# Instantiate model
print("Instantiate model")
net = SFNet(args.feature_h, args.feature_w, beta=args.beta, kernel_sigma = args.kernel_sigma)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load weights
print("Load pre-trained weights")
best_weights = torch.load("./weights/best_checkpoint.pt")
adap3_dict = best_weights['state_dict1']
adap4_dict = best_weights['state_dict2']
net.adap_layer_feat3.load_state_dict(adap3_dict, strict=False)
net.adap_layer_feat4.load_state_dict(adap4_dict, strict=False)

# PCK metric from 'https://github.com/ignacio-rocco/weakalign/blob/master/util/eval_util.py'
def correct_keypoints(source_points, warped_points, L_pck, alpha=0.1):
    # compute correct keypoints
    p_src = source_points[0,:]
    p_wrp = warped_points[0,:]

    N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
    point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
    L_pck_mat = L_pck[0].expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * alpha)
    pck = torch.mean(correct_points.float())
    return pck

with torch.no_grad():
    print('Computing PCK@Test set...')
    net.eval()
    total_correct_points = 0
    total_points = 0
    for i, batch in enumerate(test_loader):
        src_image = batch['image1'].to(device)
        tgt_image = batch['image2'].to(device)
        output = net(src_image, tgt_image, train=False)

        small_grid = output['grid_T2S'][:,1:-1,1:-1,:]
        small_grid[:,:,:,0] = small_grid[:,:,:,0] * (args.feature_w//2)/(args.feature_w//2 - 1)
        small_grid[:,:,:,1] = small_grid[:,:,:,1] * (args.feature_h//2)/(args.feature_h//2 - 1)
        src_image_H = int(batch['image1_size'][0][0])
        src_image_W = int(batch['image1_size'][0][1])
        tgt_image_H = int(batch['image2_size'][0][0])
        tgt_image_W = int(batch['image2_size'][0][1])
        small_grid = small_grid.permute(0,3,1,2)
        grid = F.interpolate(small_grid, size = (tgt_image_H,tgt_image_W), mode='bilinear', align_corners=True)
        grid = grid.permute(0,2,3,1)
        grid_np = grid.cpu().data.numpy()

        image1_points = batch['image1_points'][0]
        image2_points = batch['image2_points'][0]

        est_image1_points = np.zeros((2,image1_points.size(1)))
        for j in range(image2_points.size(1)):
            point_x = int(np.round(image2_points[0,j]))
            point_y = int(np.round(image2_points[1,j]))

            if point_x == -1 and point_y == -1:
                continue

            if point_x == tgt_image_W:
                point_x = point_x - 1

            if point_y == tgt_image_H:
                point_y = point_y - 1

            est_y = (grid_np[0,point_y,point_x,1] + 1)*(src_image_H-1)/2
            est_x = (grid_np[0,point_y,point_x,0] + 1)*(src_image_W-1)/2
            est_image1_points[:,j] = [est_x,est_y]

        total_correct_points += correct_keypoints(batch['image1_points'], torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'], alpha=0.1)
    PCK = total_correct_points / len(test_dataset)
    print('PCK: %5f' % PCK)
                
