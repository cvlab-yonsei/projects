import argparse

parser = argparse.ArgumentParser(description='flowGRU')

parser.add_argument('--data_path',
                    help='path to datasets') #'/mnt/sdb/datasets/KITTI_raw_data'

parser.add_argument('--weight_path',
                    default='./weights/flowGRU_final/',
                    help='path to save weights')

parser.add_argument("--h",
                    default=320,
                    type=int,
                    help='size of cropped height')

parser.add_argument("--w",
                    default=960,
                    type=int,
                    help='size of cropped width')

parser.add_argument("--bs",
                    default=16,
                    type=int,
                    help='size of batch')

parser.add_argument("--bs_test",
                    default=1,
                    type=int,
                    help='size of batch')

parser.add_argument("--video_split",
                    default=50,
                    type=int,
                    help='total length of video')

parser.add_argument("--frame_split",
                    default=20,
                    type=int,
                    help='the number of samples to be randomly chosen')

########
parser.add_argument('--split',
                    type=str,
                    help='data split, kitti or eigen')

parser.add_argument('--gt_path',
                    type=str,
                    help='path to ground truth disparities')

parser.add_argument('--min_depth',
                    default=1e-3,
                    type=float,
                    help='minimum depth for evaluation')

parser.add_argument('--max_depth',
                    default=80,
                    type=float,
                    help='maximum depth for evaluation')

parser.add_argument('--eigen_crop',
                    help='if set, crops according to Eigen NIPS14',
                    action='store_true')

parser.add_argument('--garg_crop',
                    help='if set, crops according to Garg  ECCV16',
                    action='store_true')

opt = parser.parse_args()