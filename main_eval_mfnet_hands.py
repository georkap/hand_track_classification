# -*- coding: utf-8 -*-
"""
Created on 16-Apr-2019

main eval mfnet hands

@author Georgios Kapidis
"""

import os
import torch, dsntnn
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.mfnet_3d_hands import MFNET_3D
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save
from utils.dataset_loader import VideoAndPointDatasetLoader
from utils.dataset_loader_utils import Resize, RandomCrop, ToTensorVid, Normalize
from utils.calc_utils import AverageMeter, accuracy, eval_final_print
from utils.video_sampler import RandomSampling

from models.mfnet_3d_do import MFNET_3D as MFNET_3D_DO

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

def validate_resnet(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, cls_losses, coo_losses,  top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    outputs = []

    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, points, video_names) in enumerate(test_iterator):
            inputs = inputs.cuda()
            target_class = targets.cuda()
            target_var = points.cuda()

            output, coords, heatmaps = model(inputs)

            cls_loss = criterion(output, target_class)
            coord_loss = calc_coord_loss(coords, heatmaps, target_var)
            loss = cls_loss + coord_loss

            batch_preds = []
            for j in range(output.size(0)):
                res = np.argmax(output[j].detach().cpu().numpy())
                label = targets[j].cpu().numpy()
                outputs.append([res, label])
                batch_preds.append("{}, P-L:{}-{}".format(video_names[j], res, label))

            t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1, 5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))
            cls_losses.update(cls_loss.item(), output.size(0))
            coo_losses.update(coord_loss.item(), output.size(0))

            print_and_save('[Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]\n\t{}'.format(
                batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg, batch_preds), log_file)
        print_and_save(
            '{} Results: Loss(f|cls|coo) {:.4f} | {:.4f} | {:.4f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg,
                                                                                                    cls_losses.avg,
                                                                                                    coo_losses.avg,
                                                                                                    top1.avg, top5.avg),
            log_file)
    return top1.avg, outputs

def validate_resnet_do():
    pass

def main():
    args = parse_args('mfnet', val=True)

    output_dir = os.path.dirname(args.ckpt_path)
    log_file = os.path.join(output_dir, "results-accuracy-validation.txt") if args.logging else None
    if args.double_output and args.logging:
        if 'verb' in args.ckpt_path:
            log_file = os.path.join(output_dir, "results-accuracy-validation-verb.txt")
        if 'noun' in args.ckpt_path:
            log_file = os.path.join(output_dir, "results-accuracy-validation-noun.txt")

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

    model_ft = mfnet_3d(num_classes, 2)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.ckpt_path, map_location={'cuda:1': 'cuda:0'})
    model_ft.load_state_dict(checkpoint['state_dict'])
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)

    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    for i in range(args.mfnet_eval):
        val_sampler = RandomSampling(num=args.clip_length,
                                     interval=args.frame_interval,
                                     speed=[1.0, 1.0], seed=i)
        val_transforms = transforms.Compose([Resize((256, 256), False), RandomCrop((224, 224)),
                                             ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])

        val_loader = VideoAndPointDatasetLoader(val_sampler, args.val_list,
                                                 num_classes=args.verb_classes,
                                                 point_list_prefix=args.bpv_prefix,
                                                 batch_transform=val_transforms,
                                                 img_tmpl='frame_{:010d}.jpg',
                                                 norm_val=[456., 256., 456., 256.],
                                                 validation=True)
        val_iter = torch.utils.data.DataLoader(val_loader,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

        top1, outputs = validate(model_ft, ce_loss, val_iter, checkpoint['epoch'], args.val_list.split("\\")[-1],
                                 log_file)

        if not isinstance(top1, tuple):
            video_preds = [x[0] for x in outputs]
            video_labels = [x[1] for x in outputs]
            mean_cls_acc, top1_acc = eval_final_print(video_preds, video_labels, "Verbs", args.annotations_path,
                                                      args.val_list, log_file)
            overall_mean_cls_acc += mean_cls_acc
            overall_top1 += top1_acc
        else:
            video_preds_a, video_preds_b = [x[0] for x in outputs[0]], [x[0] for x in outputs[1]]
            video_labels_a, video_labels_b = [x[1] for x in outputs[0]], [x[1] for x in outputs[1]]
            mean_cls_acc_a, top1_acc_a = eval_final_print(video_preds_a, video_labels_a, "Verbs", args.annotations_path,
                                                          args.val_list, log_file)
            mean_cls_acc_b, top1_acc_b = eval_final_print(video_preds_b, video_labels_b, "Nouns", args.annotations_path,
                                                          args.val_list, log_file)
            overall_mean_cls_acc = (overall_mean_cls_acc[0] + mean_cls_acc_a, overall_mean_cls_acc[1] + mean_cls_acc_b)
            overall_top1 = (overall_top1[0] + top1_acc_a, overall_top1[1] + top1_acc_b)

    print_and_save("", log_file)
    if not isinstance(top1, tuple):
        print_and_save("Mean Cls Acc {}".format(overall_mean_cls_acc / args.mfnet_eval), log_file)
        print_and_save("Dataset Acc ({} times) {}".format(args.mfnet_eval, overall_top1 / args.mfnet_eval), log_file)
    else:
        print_and_save("Mean Cls Acc a {}, b {}".format(overall_mean_cls_acc[0] / args.mfnet_eval,
                                                        overall_mean_cls_acc[1] / args.mfnet_eval), log_file)
        print_and_save("Dataset Acc ({} times) a {}, b {}".format(args.mfnet_eval, overall_top1[0] / args.mfnet_eval,
                                                                  overall_top1[1] / args.mfnet_eval), log_file)


if __name__ == '__main__':
    main()