import torch
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
import numpy as np
import os
import random
from custom_dataset import Pascal_Seg_Synth, PF_Pascal
from custom_loss import loss_function
from model import SFNet
#import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="SFNet")
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='decaying factor')
parser.add_argument('--decay_schedule', type=str, default='30', help='learning rate decaying schedule')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--feature_h', type=int, default=20, help='height of feature volume')
parser.add_argument('--feature_w', type=int, default=20, help='width of feature volume')
parser.add_argument('--train_image_path', type=str, default='./data/training_data/VOC2012_seg_img.npy', help='directory of pre-processed(.npy) images')
parser.add_argument('--train_mask_path', type=str, default='./data/training_data/VOC2012_seg_msk.npy', help='directory of pre-processed(.npy) foreground masks')
parser.add_argument('--valid_csv_path', type=str, default='./data/PF_Pascal/bbox_val_pairs_pf_pascal.csv', help='directory of validation csv file')
parser.add_argument('--valid_image_path', type=str, default='./data/PF_Pascal/', help='directory of validation data')
parser.add_argument('--beta', type=float, default=50, help='inverse temperature of softmax @ kernel soft argmax')
parser.add_argument('--kernel_sigma', type=float, default=5, help='standard deviation of Gaussian kerenl @ kernel soft argmax')
parser.add_argument('--lambda1', type=float, default=3, help='weight parameter of mask consistency loss')
parser.add_argument('--lambda2', type=float, default=16, help='weight parameter of flow consistency loss')
parser.add_argument('--lambda3', type=float, default=0.5, help='weight parameter of smoothness loss')
parser.add_argument('--eval_type', type=str, default='bounding_box', choices=('bounding_box','image_size'), help='evaluation type for PCK threshold (bounding box | image size)')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Set seed
if args.seed == None:
    args.seed = np.random.randint(10000)
    print('Seed number: ', args.seed)
global global_seed
global_seed = args.seed
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)
torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

# Make a log file & directory for saving weights
def log(text, LOGGER_FILE):
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()

LOGGER_FILE = './training_log.txt'

if os.path.exists(LOGGER_FILE):
    os.remove(LOGGER_FILE)

if not os.path.exists("./weights/"):
    os.mkdir("./weights/")

# Data Loader
train_dataset = Pascal_Seg_Synth(args.train_image_path, args.train_mask_path, args.feature_h, args.feature_w)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers = args.num_workers,
                                           worker_init_fn = _init_fn)

valid_dataset = PF_Pascal(args.valid_csv_path, args.valid_image_path, args.feature_h, args.feature_w, args.eval_type)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=1,
                                           shuffle=False, num_workers = args.num_workers)


# Instantiate model
net = SFNet(args.feature_h, args.feature_w, beta=args.beta, kernel_sigma = args.kernel_sigma)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Instantiate loss
criterion = loss_function(args).to(device)

# Instantiate optimizer
param = list(net.adap_layer_feat3.parameters())+list(net.adap_layer_feat4.parameters())
optimizer = torch.optim.Adam(param, lr=args.lr)
decay_schedule = list(map(lambda x: int(x), args.decay_schedule.split('-')))
scheduler = lrs.MultiStepLR(optimizer, milestones = decay_schedule, gamma = args.gamma)

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

# Training
best_pck = 0
print('Training started')
for ep in range(args.epochs):
    print('Current epoch : %d' % ep)
    log('Current epoch : %d\n' % ep, LOGGER_FILE)
    log('Current learning rate : %e\n' % optimizer.state_dict()['param_groups'][0]['lr'], LOGGER_FILE)

    net.train()
    net.feature_extraction.eval()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        src_image = batch['image1'].to(device)
        tgt_image = batch['image2'].to(device)
        GT_src_mask = batch['mask1'].to(device)
        GT_tgt_mask = batch['mask2'].to(device)

        output = net(src_image, tgt_image, GT_src_mask, GT_tgt_mask)

        optimizer.zero_grad()
        loss,L1,L2,L3 = criterion(output, GT_src_mask, GT_tgt_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        log("Epoch %03d (%04d/%04d) = Loss : %5f (Now : %5f)\t" % (ep, i, len(train_dataset) // args.batch_size, total_loss / (i+1), loss.cpu().data), LOGGER_FILE)
        log("L1 : %5f, L2 : %5f, L3 : %5f\n" % (L1.item(), L2.item(), L3.item()), LOGGER_FILE)
    scheduler.step()
    log("Epoch %03d finished... Average loss : %5f\n"%(ep,total_loss/len(train_loader)), LOGGER_FILE)

    with torch.no_grad():
        log('Computing PCK@Validation set...', LOGGER_FILE)
        net.eval()
        total_correct_points = 0
        total_points = 0
        for i, batch in enumerate(valid_loader):
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
        PCK = total_correct_points / len(valid_dataset)
        log('PCK: %5f\n\n' % PCK, LOGGER_FILE)
        if PCK > best_pck:
            best_pck = PCK
            torch.save({'state_dict1' : net.adap_layer_feat3.state_dict(), 
                        'state_dict2' : net.adap_layer_feat4.state_dict()},
                        './weights/best_checkpoint.pt')
                
print('Done')