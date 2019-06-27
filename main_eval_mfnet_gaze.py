"""
26/June/19

Script to evaluate the gaze predictions based on https://github.com/hyf015/egocentric-gaze-prediction/blob/master/utils.py#L96

which is related to [1],[2]

[1] Predicting Gaze in Egocentric Video byLearning Task-dependent Attention Transition Yifei Huang, Minjie Cai, Zhenqiang Li, and Yoichi Sato
[2] Mutual Context Network for Jointly Estimating Egocentric Gaze and Actions Yifei Huang, Minjie Cai, Zhenqiang Li, Yoichi Sato

"""

import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from models.mfnet_3d_mo import MFNET_3D as MFNET_3D_MO
from utils.argparse_utils import parse_args, make_log_file_name
from utils.file_utils import print_and_save
from utils.dataset_loader import VideoFromImagesDatasetLoader
from utils.dataset_loader_utils import Resize, RandomCrop, ToTensorVid, Normalize, CenterCrop
from utils.video_sampler import RandomSampling, MiddleSampling, DoubleFullSampling
from utils.train_utils import validate_mfnet_mo_gaze

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
torch.set_printoptions(linewidth=1000000, threshold=1000000)
mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]

GTEA_CLASSES = [106, 19, 53]
def main():
    args = parse_args('mfnet', val=True)

    output_dir = os.path.dirname(args.ckpt_path)
    log_file = make_log_file_name(output_dir, args)
    print_and_save(args, log_file)
    cudnn.benchmark = True

    mfnet_3d = MFNET_3D_MO
    num_classes = [args.action_classes, args.verb_classes, args.noun_classes]
    validate = validate_mfnet_mo_gaze

    kwargs = {}
    num_coords = 0
    if args.use_gaze:
        num_coords += 1
    else:
        print('Expected a gaze model. Exiting...')
        return
    if args.use_hands:
        num_coords += 2
    kwargs['num_coords'] = num_coords

    model_ft = mfnet_3d(num_classes, **kwargs)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.ckpt_path, map_location={'cuda:1': 'cuda:0'})
    model_ft.load_state_dict(checkpoint['state_dict'])
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)

    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    num_valid_classes = len([cls for cls in num_classes if cls > 0])
    valid_classes = [cls for cls in num_classes if cls > 0]
    overall_top1 = [0]*num_valid_classes
    overall_mean_cls_acc = [0]*num_valid_classes
    for i in range(args.mfnet_eval):
        crop_type = CenterCrop((224, 224)) if args.eval_crop == 'center' else RandomCrop((224, 224))
        if args.eval_sampler == 'middle':
            val_sampler = MiddleSampling(num=args.clip_length)
        elif args.eval_sampler == 'doublefull':
            val_sampler = DoubleFullSampling()
        else:
            val_sampler = RandomSampling(num=args.clip_length,
                                         interval=args.frame_interval,
                                         speed=[1.0, 1.0], seed=i)

        val_transforms = transforms.Compose([Resize((256, 256), False), crop_type,
                                             ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])

        # val_loader = FromVideoDatasetLoaderGulp(val_sampler, args.val_list, 'GTEA', num_classes, GTEA_CLASSES,
        #                                         use_gaze=args.use_gaze, gaze_list_prefix=args.gaze_list_prefix,
        #                                         use_hands=args.use_hands, hand_list_prefix=args.hand_list_prefix,
        #                                         batch_transform=val_transforms, extra_nouns=False, validation=True)
        val_loader = VideoFromImagesDatasetLoader(val_sampler, args.val_list, 'GTEA', num_classes, GTEA_CLASSES,
                                                  use_gaze=args.use_gaze, gaze_list_prefix=args.gaze_list_prefix,
                                                  use_hands=args.use_hands, hand_list_prefix=args.hand_list_prefix,
                                                  batch_transform=val_transforms, extra_nouns=False, validation=True)
        val_iter = torch.utils.data.DataLoader(val_loader,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

        # evaluate dataset
        validate(model_ft, ce_loss, val_iter, num_valid_classes, args.use_gaze, args.use_hands,
                                 checkpoint['epoch'], args.val_list.split("\\")[-1], log_file)


if __name__ == '__main__':
    main()