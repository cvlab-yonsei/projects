import torch
import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="Market-1501-v15.09.15",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')

parser.add_argument('--weight',
                    default='weights/model.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=400,
                    type=int,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[300],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=4,
                    type=int,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    type=int,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=8,
                    type=int,
                    help='the batch size for test')

parser.add_argument("--device",
                    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    help='cuda is available?')

parser.add_argument("--num_cls",
                    default=751, # (Market1501: 751, Cuhk-03: 767, Duke-MTMC: 702)
                    type=int,
                    help='# of classes')

parser.add_argument("--feat_id",
                    default=256, #--> 2048
                    type=int,
                    help='size of id features')

parser.add_argument("--feat_nid",
                    default=64, #--> 512
                    type=int,
                    help='size of id features')

parser.add_argument("--feat_niz",
                    default=128,
                    type=int,
                    help='size of id features')

parser.add_argument("--feat_G",
                    default=64,
                    type=int,
                    help='size of Generator')

parser.add_argument("--feat_D",
                    default=32,
                    type=int,
                    help='size of Discriminator')

parser.add_argument("--dropout",
                    default=0.2,
                    help='probaility of dropout')

parser.add_argument("--stage",
                    default=1,
                    type=int,
                    help='# of training stage')

parser.add_argument("--save_path",
                    default='weights',
                    help='the path for saving weights')

parser.add_argument("--name",
                    default='/isgan',
                    help='the additional path to identify')

parser.add_argument("--start",
                    default=0,
                    type=int,
                    help='start epoch')

parser.add_argument("--stage2_weight_path", default='/model_stage2_200.pt')

opt = parser.parse_args()
