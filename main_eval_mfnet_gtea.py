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

def validate_mfnet_mo(model, criterion, test_iterator, num_outputs, cur_epoch, dataset, log_file):
    loss_meters = [AverageMeter() for _ in range(num_outputs)]
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_outputs)]
    task_outputs = []*num_outputs

    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, video_names) in enumerate(test_iterator):
            inputs = inputs.cuda()
            targets = targets.cuda().transpose(0, 1)

            outputs = model(inputs)

            losses_per_task = []
            for output, target in zip(outputs, targets):
                loss_for_task = criterion(output, target)
                losses_per_task.append(loss_for_task)

            loss = sum(losses_per_task)

            batch_size = outputs[0].size(0)

            batch_preds = []*num_outputs
            for j in range(batch_size):
                txt_batch_preds = "{}".format(video_names[j])
                for ind in range(num_outputs):
                    txt_batch_preds += ", "
                    res = np.argmax(outputs[ind][j].detach().cpu().numpy())
                    label = targets[ind][j].detach().cpu().numpy()
                    task_outputs[ind].append([res, label])
                    txt_batch_preds += "T{} P-L:{}-{}".format(ind, res, label)
                batch_preds.append(txt_batch_preds)

            losses.update(loss.item(), batch_size)
            for ind in range(num_outputs):
                t1, t5 = accuracy(outputs[ind].detach().cpu(), targets[ind].detach().cpu(), topk=(1, 5))
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

    model_ft = mfnet_3d(num_classes)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.ckpt_path, map_location={'cuda:1': 'cuda:0'})
    model_ft.load_state_dict(checkpoint['state_dict'])
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)

    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    num_valid_classes = len([cls for cls in num_classes if cls > 0])
    valid_classes = [cls for cls in num_classes if cls > 0]
    overall_top1, overall_mean_cls_acc = []*num_valid_classes, []*num_valid_classes
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
                                                batch_transform=val_transforms, extra_nouns=False,
                                                validation=True)
        val_iter = torch.utils.data.DataLoader(val_loader,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

        # evaluate dataset
        top1, outputs = validate(model_ft, ce_loss, val_iter, num_valid_classes, checkpoint['epoch'],
                                 args.val_list.split("\\")[-1], log_file)

        # calculate statistics
        for ind in range(num_valid_classes):
            video_preds = [x[0] for x in outputs[ind]]
            video_labels = [x[1] for x in outputs[ind]]
            mean_cls_acc, top1_acc = eval_final_print_gtea(video_preds, video_labels, ind, valid_classes[ind], log_file)
            overall_mean_cls_acc[ind] += mean_cls_acc
            overall_top1 += top1_acc

    print_and_save("", log_file)
    text_mean_cls_acc = "Mean Cls Acc ({} times)".format(args.mfnet_eval)
    text_dataset_acc = "Dataset Acc ({} times)".format(args.mfnet_eval)
    for ind in range(num_valid_classes):
        text_mean_cls_acc += ", T{}::{} ".format(ind, overall_mean_cls_acc[ind] / args.mfnet_eval)
        text_dataset_acc += ", T{}::{} ".format(ind, overall_top1[ind] / args.mfnet_eval)
    print_and_save(text_mean_cls_acc, log_file)
    print_and_save(text_dataset_acc, log_file)

if __name__ == '__main__':
    main()