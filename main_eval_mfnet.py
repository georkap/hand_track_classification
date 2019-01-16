# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 00:06:13 2018

main eval mfnet

@author: Γιώργος
"""

import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.mfnet_3d import MFNET_3D
from models.mfnet_3d_do import MFNET_3D as MFNET_3D_DO
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save
from utils.dataset_loader import VideoDatasetLoader
from utils.dataset_loader_utils import Resize, RandomCrop, ToTensorVid, Normalize
from utils.calc_utils import AverageMeter, accuracy, analyze_preds_labels
from utils.video_sampler import RandomSampling

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
torch.set_printoptions(linewidth=1000000, threshold=1000000)
mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]

def validate_resnet_do(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, losses_a, losses_b, top1_a, top5_a, top1_b, top5_b = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    outputs_a, outputs_b = [], []
    
    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, video_names) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets_a = torch.tensor(targets[0]).cuda()
            targets_b = torch.tensor(targets[1]).cuda() 
            output_a, output_b = model(inputs)
            loss_a = criterion(output_a, targets_a)
            loss_b = criterion(output_b, targets_b)
            loss = 0.75*loss_a + 0.25*loss_b
            
            batch_preds = []
            for j in range(output_a.size(0)):
                res_a = np.argmax(output_a[j].detach().cpu().numpy())
                res_b = np.argmax(output_b[j].detach().cpu().numpy())
                label_a = targets_a[j].cpu().numpy()
                label_b = targets_b[j].cpu().numpy()
                outputs_a.append([res_a, label_a])
                outputs_b.append([res_b, label_b])
                batch_preds.append("{}, a P-L:{}-{}, b P-L:{}-{}".format(video_names[j], res_a, label_a, res_b, label_b))
            
            t1_a, t5_a = accuracy(output_a.detach().cpu(), targets_a.detach.cpu(), topk=(1,5))
            t1_b, t5_b = accuracy(output_b.detach().cpu(), targets_b.detach.cpu(), topk=(1,5))
            top1_a.update(t1_a.item(), output_a.size(0))
            top5_a.update(t5_a.item(), output_a.size(0))
            top1_b.update(t1_b.item(), output_b.size(0))
            top5_b.update(t5_b.item(), output_b.size(0))
            losses_a.update(loss_a.item(), output_a.size(0))
            losses_b.update(loss_b.item(), output_b.size(0))
            losses.update(loss.item(), output_a.size(0))
            
            to_print = "[Batch {}/{}]" + \
            "[Top1_a {:.3f}[avg:{:.3f}], Top5_a {:.3f}[avg:{:.3f}]," + \
            "Top1_b {:.3f}[avg:{:.3f}], Top5_b {:.3f}[avg:{:.3f}]]\n\t".format(
            batch_idx, len(test_iterator),
            top1_a.val, top1_a.avg, top5_a.val, top5_a.avg,
            top1_b.val, top1_b.avg, top5_b.val, top5_b.avg,
            batch_preds)
            print_and_save(to_print, log_file)
            
        print_and_save('{} Results: Loss {:.3f}, Top1_a {:.3f}, Top5_a {:.3f}, Top1_b {:.3f}, Top5_b {:.3f}'.format(dataset, losses.avg, top1_a.avg, top5_a.avg, top1_b.avg, top5_b.avg), log_file)
    return (top1_a.avg, top1_b.avg), (outputs_a, outputs_b)

def validate_resnet(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    outputs = []
    
    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, video_names) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()
            
            output = model(inputs)            
            loss = criterion(output, targets)
            
            batch_preds = []
            for j in range(output.size(0)):
                res = np.argmax(output[j].detach().cpu().numpy())
                label = targets[j].cpu().numpy()
                outputs.append([res, label])
                batch_preds.append("{}, P-L:{}-{}".format(video_names[j], res, label))
                
            t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1,5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))

            print_and_save('[Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]\n\t{}'.format(
                    batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg, batch_preds), log_file)
        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg, outputs

def main():
    args = parse_args('mfnet', val=True)
    
    output_dir = os.path.dirname(args.ckpt_path)
    log_file = os.path.join(output_dir, "results-accuracy-validation.txt") if args.logging else None
    print_and_save(args, log_file)
    cudnn.benchmark = True
    
    if not args.double_output:
        mfnet_3d = MFNET_3D
        num_classes = args.verb_classes 
        validate = validate_resnet
        overall_top1, overall_mean_cls_acc = 0.0, 0.0
    else:
        mfnet_3d = MFNET_3D_DO
        num_classes = (args.verb_classes, args.noun_classes)
        validate = validate_resnet_do
        overall_top1, overall_mean_cls_acc = (0.0, 0.0), (0.0, 0.0)
        
    model_ft = mfnet_3d(num_classes)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.ckpt_path, map_location={'cuda:1':'cuda:0'})
    model_ft.load_state_dict(checkpoint['state_dict'])
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)
    
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    
    for i in range(args.mfnet_eval):
        val_sampler = RandomSampling(num=args.clip_length,
                                     interval=args.frame_interval,
                                     speed=[1.0, 1.0], seed=i)
        val_transforms = transforms.Compose([Resize((256,256), False), RandomCrop((224,224)),
                                             ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
        val_loader = VideoDatasetLoader(val_sampler, args.val_list, 
                                        num_classes=num_classes, 
                                        batch_transform=val_transforms,
                                        img_tmpl='frame_{:010d}.jpg',
                                        validation=True)
        val_iter = torch.utils.data.DataLoader(val_loader,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

        top1, outputs = validate(model_ft, ce_loss, val_iter, checkpoint['epoch'], args.val_list.split("\\")[-1], log_file)

        if len(top1) == 1:
            video_preds = [x[0] for x in outputs]
            video_labels = [x[1] for x in outputs]
            mean_cls_acc, top1_acc = eval_final_print(video_preds, video_labels, "Verbs", log_file)
            overall_mean_cls_acc += mean_cls_acc
            overall_top1 += top1_acc
        elif len(top1) == 2:
            video_preds_a, video_preds_b = [x[0] for x in outputs[0]], [x[0] for x in outputs[1]]
            video_labels_a, video_labels_b = [x[1] for x in outputs[0]], [x[1] for x in outputs[1]]
            mean_cls_acc_a, top1_acc_a = eval_final_print(video_preds_a, video_labels_a, "Verbs", log_file)
            mean_cls_acc_b, top1_acc_b = eval_final_print(video_preds_b, video_labels_b, "Nouns", log_file)
            overall_mean_cls_acc = (overall_mean_cls_acc[0] + mean_cls_acc_a, overall_mean_cls_acc[1] + mean_cls_acc_b)
            overall_top1 = (overall_top1[0] + top1_acc_a, overall_top1[1] + top1_acc_b)
        
    print_and_save("", log_file)
    if len(top1) == 1:
        print_and_save("Mean Cls Acc {}".format(overall_mean_cls_acc/args.mfnet_eval), log_file)
        print_and_save("Dataset Acc ({} times) {}".format(args.mfnet_eval, overall_top1/args.mfnet_eval), log_file)
    elif len(top1) == 2:
        print_and_save("Mean Cls Acc a {}, b {}".format(overall_mean_cls_acc[0]/args.mfnet_eval, overall_mean_cls_acc[1]/args.mfnet_eval), log_file)
        print_and_save("Dataset Acc ({} times) a {}, b {}".format(args.mfnet_eval, overall_top1[0]/args.mfnet_eval, overall_top1[1]/args.mfnet_eval), log_file)

def eval_final_print(video_preds, video_labels, cls_type, log_file):
    cf, recall, precision, cls_acc, mean_cls_acc, top1_acc = analyze_preds_labels(video_preds, video_labels)
    print_and_save(cls_type, log_file)
    print_and_save(cf, log_file)
    print_and_save("Cls Rec {}".format(recall), log_file)
    print_and_save("Cls Pre {}".format(precision), log_file)
    print_and_save("Cls Acc {}".format(cls_acc), log_file)
    print_and_save("Mean Cls Acc {:.02f}%".format(mean_cls_acc), log_file)
    print_and_save("Dataset Acc {}".format(top1_acc), log_file)
    return mean_cls_acc, top1_acc

if __name__ == '__main__':
    main()