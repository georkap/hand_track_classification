# -*- coding: utf-8 -*-
"""
Created on 16-Apr-2019

main eval mfnet hands

@author Georgios Kapidis
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
from utils.dataset_loader import VideoAndPointDatasetLoader
from utils.dataset_loader_utils import Resize, RandomCrop, ToTensorVid, Normalize, CenterCrop
from utils.calc_utils import eval_final_print, eval_final_print_mt
from utils.video_sampler import RandomSampling, MiddleSampling
from utils.train_utils import validate_mfnet_mo

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
    validate = validate_mfnet_mo

    kwargs = {}
    num_coords = 0
    # if args.use_gaze:
    #     num_coords += 1
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

    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    num_valid_classes = len([cls for cls in num_classes if cls > 0])
    valid_classes = [cls for cls in num_classes if cls > 0]
    overall_top1 = [0]*num_valid_classes
    overall_mean_cls_acc = [0]*num_valid_classes
    for i in range(args.mfnet_eval):
        crop_type = CenterCrop((224, 224)) if args.eval_crop == 'center' else RandomCrop((224, 224))
        if args.eval_sampler == 'middle':
            val_sampler = MiddleSampling(num=args.clip_length)
        else:
            val_sampler = RandomSampling(num=args.clip_length,
                                         interval=args.frame_interval,
                                         speed=[1.0, 1.0], seed=i)

        val_transforms = transforms.Compose([Resize((256, 256), False), crop_type,
                                             ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])

        val_loader = VideoAndPointDatasetLoader(val_sampler, args.val_list, point_list_prefix=args.bpv_prefix,
                                                num_classes=num_classes, img_tmpl='frame_{:010d}.jpg',
                                                norm_val=[456., 256., 456., 256.], batch_transform=val_transforms,
                                                use_hands=args.use_hands, validation=True)
        val_iter = torch.utils.data.DataLoader(val_loader,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

        top1, outputs = validate(model_ft, ce_loss, val_iter, num_valid_classes, False, args.use_hands,
                                 checkpoint['epoch'], args.val_list.split("\\")[-1], log_file)

        task_types = get_task_type_epic(args.action_classes, args.verb_classes, args.noun_classes)
        # calculate statistics
        for ind in range(num_valid_classes):
            task_type = task_types[ind]
            video_preds = [x[0] for x in outputs[ind]]
            video_labels = [x[1] for x in outputs[ind]]
            mean_cls_acc, top1_acc = eval_final_print_mt(video_preds, video_labels, ind, valid_classes[ind], log_file,
                                                         args.annotations_path, args.val_list, task_type=task_type,
                                                         action_file=args.epic_action_file)
            overall_mean_cls_acc[ind] += mean_cls_acc
            overall_top1[ind] += top1_acc

    print_and_save("", log_file)
    text_mean_cls_acc = "Mean Cls Acc ({} times)".format(args.mfnet_eval)
    text_dataset_acc = "Dataset Acc ({} times)".format(args.mfnet_eval)
    for ind in range(num_valid_classes):
        text_mean_cls_acc += ", T{}::{} ".format(ind, (overall_mean_cls_acc[ind] / args.mfnet_eval))
        text_dataset_acc += ", T{}::{} ".format(ind, (overall_top1[ind] / args.mfnet_eval))
    print_and_save(text_mean_cls_acc, log_file)
    print_and_save(text_dataset_acc, log_file)

    #     if not isinstance(top1, tuple):
    #         video_preds = [x[0] for x in outputs]
    #         video_labels = [x[1] for x in outputs]
    #         mean_cls_acc, top1_acc = eval_final_print(video_preds, video_labels, "Verbs", args.annotations_path,
    #                                                   args.val_list, num_classes, log_file)
    #         overall_mean_cls_acc += mean_cls_acc
    #         overall_top1 += top1_acc
    #     else:
    #         video_preds_a, video_preds_b = [x[0] for x in outputs[0]], [x[0] for x in outputs[1]]
    #         video_labels_a, video_labels_b = [x[1] for x in outputs[0]], [x[1] for x in outputs[1]]
    #         mean_cls_acc_a, top1_acc_a = eval_final_print(video_preds_a, video_labels_a, "Verbs", args.annotations_path,
    #                                                       args.val_list, num_classes, log_file)
    #         mean_cls_acc_b, top1_acc_b = eval_final_print(video_preds_b, video_labels_b, "Nouns", args.annotations_path,
    #                                                       args.val_list, num_classes, log_file)
    #         overall_mean_cls_acc = (overall_mean_cls_acc[0] + mean_cls_acc_a, overall_mean_cls_acc[1] + mean_cls_acc_b)
    #         overall_top1 = (overall_top1[0] + top1_acc_a, overall_top1[1] + top1_acc_b)
    #
    # print_and_save("", log_file)
    # if not isinstance(top1, tuple):
    #     print_and_save("Mean Cls Acc {}".format(overall_mean_cls_acc / args.mfnet_eval), log_file)
    #     print_and_save("Dataset Acc ({} times) {}".format(args.mfnet_eval, overall_top1 / args.mfnet_eval), log_file)
    # else:
    #     print_and_save("Mean Cls Acc a {}, b {}".format(overall_mean_cls_acc[0] / args.mfnet_eval,
    #                                                     overall_mean_cls_acc[1] / args.mfnet_eval), log_file)
    #     print_and_save("Dataset Acc ({} times) a {}, b {}".format(args.mfnet_eval, overall_top1[0] / args.mfnet_eval,
    #                                                               overall_top1[1] / args.mfnet_eval), log_file)


if __name__ == '__main__':
    main()