import os
import numpy as np
import json

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from models.mfnet_3d_mo import MFNET_3D as MFNET_3D_MO
from utils.argparse_utils import parse_args, make_log_file_name
from utils.file_utils import print_and_save
from utils.dataset_loader import VideoAndPointDatasetLoader
from utils.dataset_loader_utils import Resize, ToTensorVid, Normalize, CenterCrop
from utils.video_sampler import MiddleSampling
from utils.train_utils import validate_mfnet_mo_json

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
torch.set_printoptions(linewidth=1000000, threshold=1000000)
mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]

def get_task_type_epic(action_classes, verb_classes, noun_classes):
    """
    This snippet to decided what type of task is given for evaluation. This is really experiment specific and needs to be
    updated if things change. The only use for the task types is to make the evaluation on the classes with more than 100
    samples at training for the epic evaluation.
    If actions are trained explicitly then they are task0
    if verbs are trained with actions they they are task1 else they are task0
    if nouns are trained they are always verbtask+1, so either task2 or task1
    if hands are trained they are always the last task so they do not change the above order.
    :return: a list of task names that follows the size of 'num_valid_classes'
    """
    task_types = []
    if action_classes > 0:
        task_types.append("EpicActions")
    if verb_classes > 0:
        task_types.append("EpicVerbs")
    if noun_classes > 0:
        task_types.append("EpicNouns")
    return task_types


EPIC_CLASSES = [2521, 125, 322]
def main():
    args = parse_args('mfnet', val=True)

    output_dir = os.path.dirname(args.ckpt_path)
    log_file = make_log_file_name(output_dir, args)
    print_and_save(args, log_file)
    cudnn.benchmark = True

    mfnet_3d = MFNET_3D_MO
    num_classes = [args.action_classes, args.verb_classes, args.noun_classes]
    validate = validate_mfnet_mo_json

    kwargs = {}
    num_coords = 0
    if args.use_hands:
        num_coords += 2
    kwargs['num_coords'] = num_coords

    model_ft = mfnet_3d(num_classes, **kwargs)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.ckpt_path, map_location={'cuda:1': 'cuda:0'})
    if args.old_mfnet_eval:
        checkpoint['state_dict']['module.classifier_list.classifier_list.0.weight'] = checkpoint['state_dict']['module.classifier.weight']
        checkpoint['state_dict']['module.classifier_list.classifier_list.0.bias'] = checkpoint['state_dict']['module.classifier.bias']
    model_ft.load_state_dict(checkpoint['state_dict'], strict=False)
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)

    num_valid_classes = len([cls for cls in num_classes if cls > 0])
    valid_classes = [cls for cls in num_classes if cls > 0]


    crop_type = CenterCrop((224, 224))
    val_sampler = MiddleSampling(num=args.clip_length, window=64)
    val_transforms = transforms.Compose([Resize((256, 256), False), crop_type,
                                         ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    val_loader = VideoAndPointDatasetLoader(val_sampler, args.val_list, point_list_prefix=args.bpv_prefix,
                                            num_classes=num_classes, img_tmpl='frame_{:010d}.jpg',
                                            norm_val=[456., 256., 456., 256.], batch_transform=val_transforms,
                                            use_hands=False, validation=True)
    val_iter = torch.utils.data.DataLoader(val_loader,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True)

    outputs = validate(model_ft, val_iter, num_valid_classes, args.val_list.split("\\")[-1], action_file=args.epic_actions_path)

    eval_mode = 'seen' if 's1' in args.val_list else 'unseen' if 's2' in args.val_list else 'unknown'
    json_file = "{}.json".format(os.path.join(output_dir, eval_mode))
    with open(json_file, 'w') as jf:
        json.dump(outputs, jf)

if __name__ == '__main__':
    main()
