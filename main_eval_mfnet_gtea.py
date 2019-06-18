# -*- coding: utf-8 -*-
"""
Created on Thu Jun 06 2019

main eval mfnet multitask for the egtea gaze+ dataset

@author: Georgios Kapidis
"""

import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import dsntnn

from models.mfnet_3d_mo import MFNET_3D as MFNET_3D_MO
from utils.argparse_utils import parse_args, make_log_file_name
from utils.file_utils import print_and_save
from utils.dataset_loader import FromVideoDatasetLoaderGulp
from utils.dataset_loader_utils import Resize, RandomCrop, ToTensorVid, Normalize, CenterCrop
from utils.calc_utils import AverageMeter, accuracy, eval_final_print_gtea
from utils.video_sampler import RandomSampling, MiddleSampling

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
torch.set_printoptions(linewidth=1000000, threshold=1000000)
mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]

def calc_coord_loss(coords, heatmaps, target_var):
    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(coords, target_var)  # shape:[B, D, L, 2] batch, depth, locations, feature
    # Per-location regularization losses

    reg_losses = []
    for i in range(heatmaps.shape[1]):
        hms = heatmaps[:, 1]
        target = target_var[:, 1]
        reg_loss = dsntnn.js_reg_losses(hms, target, sigma_t=1.0)
        reg_losses.append(reg_loss)
    reg_losses = torch.stack(reg_losses, 1)
    # reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0) # shape: [B, D, L, 7, 7]
    # Combine losses into an overall loss
    coord_loss = dsntnn.average_loss(euc_losses + reg_losses)
    return coord_loss

def validate_mfnet_mo(model, criterion, test_iterator, num_outputs, use_gaze, use_hands, cur_epoch, dataset, log_file):
    loss_meters = [AverageMeter() for _ in range(num_outputs)]
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_outputs)]
    task_outputs = [[] for _ in range(num_outputs)]

    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, video_names) in enumerate(test_iterator):
            inputs = inputs.cuda()
            outputs, coords, heatmaps = model(inputs)
            targets = targets.cuda().transpose(0, 1)

            if use_gaze or use_hands:
                cls_targets = targets[:num_outputs, :].long()
            else:
                cls_targets = targets
            assert len(cls_targets) == num_outputs

            losses_per_task = []
            for output, target in zip(outputs, cls_targets):
                loss_for_task = criterion(output, target)
                losses_per_task.append(loss_for_task)

            loss = sum(losses_per_task)

            gaze_coord_loss, hand_coord_loss = 0, 0
            if use_gaze:  # need some debugging for the gaze targets
                gaze_targets = targets[num_outputs:num_outputs + 16, :].reshape(-1, 8, 2)
                # for a single shared layer representation of the two signals
                # for gaze slice the first element
                gaze_coords = coords[:, :, 0, :]
                gaze_heatmaps = heatmaps[:, :, 0, :, :]
                gaze_coord_loss = calc_coord_loss(gaze_coords, gaze_heatmaps, gaze_targets)
                loss = loss + gaze_coord_loss
            if use_hands:
                hand_targets = targets[-32:, :].reshape(-1, 8, 2, 2)
                # for hands slice the last two elements, first is left, second is right hand
                hand_coords = coords[:, :, -2:, :]
                hand_heatmaps = heatmaps[:, :, -2:, :]
                hand_coord_loss = calc_coord_loss(hand_coords, hand_heatmaps, hand_targets)
                loss = loss + hand_coord_loss

            batch_size = outputs[0].size(0)

            batch_preds = []
            for j in range(batch_size):
                txt_batch_preds = "{}".format(video_names[j])
                for ind in range(num_outputs):
                    txt_batch_preds += ", "
                    res = np.argmax(outputs[ind][j].detach().cpu().numpy())
                    label = cls_targets[ind][j].detach().cpu().numpy()
                    task_outputs[ind].append([res, label])
                    txt_batch_preds += "T{} P-L:{}-{}".format(ind, res, label)
                batch_preds.append(txt_batch_preds)

            losses.update(loss.item(), batch_size)
            for ind in range(num_outputs):
                t1, t5 = accuracy(outputs[ind].detach().cpu(), cls_targets[ind].detach().cpu(), topk=(1, 5))
                top1_meters[ind].update(t1.item(), batch_size)
                top5_meters[ind].update(t5.item(), batch_size)
                loss_meters[ind].update(losses_per_task[ind].item(), batch_size)

            to_print = '[Batch {}/{}]'.format(batch_idx, len(test_iterator))
            for ind in range(num_outputs):
                to_print += '[T{}::Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]],'.format(ind,
                                                                                 top1_meters[ind].val, top1_meters[ind].avg,
                                                                                 top5_meters[ind].val, top5_meters[ind].avg)
            to_print+= '\n\t{}'.format(batch_preds)
            print_and_save(to_print, log_file)

        to_print = '{} Results: Loss {:.3f}'.format(dataset, losses.avg)
        for ind in range(num_outputs):
            to_print += ', T{}::Top1 {:.3f}, Top5 {:.3f}'.format(ind, top1_meters[ind].avg, top5_meters[ind].avg)
        print_and_save(to_print, log_file)
    return [tasktop1.avg for tasktop1 in top1_meters], task_outputs

GTEA_CLASSES = [106, 19, 53]
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
    if args.use_gaze:
        num_coords += 1
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
        else:
            val_sampler = RandomSampling(num=args.clip_length,
                                         interval=args.frame_interval,
                                         speed=[1.0, 1.0], seed=i)

        val_transforms = transforms.Compose([Resize((256, 256), False), crop_type,
                                             ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])

        val_loader = FromVideoDatasetLoaderGulp(val_sampler, args.val_list, 'GTEA', num_classes, GTEA_CLASSES,
                                                use_gaze=args.use_gaze, gaze_list_prefix=args.gaze_list_prefix,
                                                use_hands=args.use_hands, hand_list_prefix=args.hand_list_prefix,
                                                batch_transform=val_transforms, extra_nouns=False, validation=True)
        val_iter = torch.utils.data.DataLoader(val_loader,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

        # evaluate dataset
        top1, outputs = validate(model_ft, ce_loss, val_iter, num_valid_classes, args.use_gaze, args.use_hands,
                                 checkpoint['epoch'], args.val_list.split("\\")[-1], log_file)

        # calculate statistics
        for ind in range(num_valid_classes):
            video_preds = [x[0] for x in outputs[ind]]
            video_labels = [x[1] for x in outputs[ind]]
            mean_cls_acc, top1_acc = eval_final_print_gtea(video_preds, video_labels, ind, valid_classes[ind], log_file)
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

if __name__ == '__main__':
    main()