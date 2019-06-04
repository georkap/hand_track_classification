# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:32:45 2019

main mfnet that classifies activities and predicts hand locations

@author: Γιώργος
"""

import time
import torch, dsntnn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.mfnet_3d_hands import MFNET_3D
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, save_checkpoints, resume_checkpoint, init_folders
from utils.dataset_loader import VideoAndPointDatasetLoader, prepare_sampler
from utils.dataset_loader_utils import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS, ToTensorVid, Normalize, \
    Resize, CenterCrop
from utils.train_utils import load_lr_scheduler, CyclicLR
from utils.calc_utils import AverageMeter, accuracy

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


def train_mfnet_h(model, optimizer, criterion, train_iterator, mixup_alpha, cur_epoch, log_file, gpus,
              lr_scheduler=None):
    batch_time, losses, cls_losses, c_losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(),\
                                                           AverageMeter(), AverageMeter()
    model.train()

    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()

    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets, points) in enumerate(train_iterator): # left_track.shape = [batch, 8, 2]
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()

        inputs = torch.tensor(inputs, requires_grad=True).cuda(gpus[0])
        target_class = torch.tensor(targets).cuda(gpus[0])
        target_var = torch.tensor(points).cuda(gpus[0])

        output, coords, heatmaps = model(inputs)

        cls_loss = criterion(output, target_class)
        coord_loss = calc_coord_loss(coords, heatmaps, target_var)
        loss = cls_loss + coord_loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        t1, t5 = accuracy(output.detach().cpu(), target_class.cpu(), topk=(1, 5))
        top1.update(t1.item(), output.size(0))
        top5.update(t5.item(), output.size(0))
        cls_losses.update(cls_loss.item(), output.size(0))
        c_losses.update(coord_loss.item(), output.size(0))
        losses.update(loss.item(), output.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        print_and_save(
            '[Epoch:{}, Batch {}/{} in {:.3f} s][Loss(f|cls|coo) {:.4f} | {:.4f} | {:.4f} [avg:{:.4f} | {:.4f} | {:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]], LR {:.6f}'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, cls_losses.val, c_losses.val,
                losses.avg, cls_losses.avg, c_losses.avg, top1.val, top1.avg, top5.val, top5.avg, lr_scheduler.get_lr()[0]),
            log_file)


def test_mfnet_h(model, criterion, test_iterator, cur_epoch, dataset, log_file, gpus):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets, points) in enumerate(test_iterator):
            inputs = inputs.cuda(gpus[0])
            target_class = targets.cuda(gpus[0])
            target_var = points.cuda(gpus[0])

            output, coords, heatmaps = model(inputs)

            cls_loss = criterion(output, target_class)
            coord_loss = calc_coord_loss(coords, heatmaps, target_var)
            loss = cls_loss + coord_loss

            t1, t5 = accuracy(output.detach().cpu(), target_class.detach().cpu(), topk=(1, 5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))

            print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        print_and_save(
            '{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg),
            log_file)
    return top1.avg


def main():
    args, model_name = parse_args('mfnet', val=False)

    output_dir, log_file = init_folders(args.base_output_dir, model_name, args.resume, args.logging)
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)
    cudnn.benchmark = True

    model_ft = MFNET_3D(args.verb_classes, 2, dropout=args.dropout)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained_model_path)
        # below line is needed if network is trained with DataParallel
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        base_dict = {k: v for k, v in list(base_dict.items()) if 'classifier' not in k}
        model_ft.load_state_dict(base_dict, strict=False)  # model.load_state_dict(checkpoint['state_dict'])
    model_ft.cuda(device=args.gpus[0])
    model_ft = torch.nn.DataParallel(model_ft, device_ids=args.gpus, output_device=args.gpus[0])
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)
    if args.resume:
        model_ft, ckpt_path = resume_checkpoint(model_ft, output_dir, model_name, args.resume_from)
        print_and_save("Resuming training from: {}".format(ckpt_path), log_file)

    # load train-val sampler
    train_sampler = prepare_sampler("train", args.clip_length, args.frame_interval)
    test_sampler = prepare_sampler("val", args.clip_length, args.frame_interval)

    # load train-val transforms
    train_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1. / 0.8], slen=[224, 288]),
        RandomCrop((224, 224)),
        RandomHorizontalFlip(),
        RandomHLS(vars=[15, 35, 25]),
        ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    test_transforms = transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)),
         ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])

    # make train-val dataset loaders
    train_loader = VideoAndPointDatasetLoader(train_sampler, args.train_list, point_list_prefix=args.bpv_prefix,
                                              num_classes=args.verb_classes, img_tmpl='frame_{:010d}.jpg',
                                              norm_val=[456., 256., 456., 256.], batch_transform=train_transforms)
    test_loader = VideoAndPointDatasetLoader(test_sampler, args.test_list, point_list_prefix=args.bpv_prefix,
                                             num_classes=args.verb_classes, img_tmpl='frame_{:010d}.jpg',
                                             norm_val=[456., 256., 456., 256.], batch_transform=test_transforms)

    # make train-val iterators
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers,
                                                 pin_memory=True)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers,
                                                pin_memory=True)

    # config optimizer
    param_base_layers = []
    param_new_layers = []
    name_base_layers = []
    for name, param in model_ft.named_parameters():
        if args.pretrained:
            if 'classifier' in name:
                param_new_layers.append(param)
            else:
                param_base_layers.append(param)
                name_base_layers.append(name)
        else:
            param_new_layers.append(param)

    optimizer = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': args.lr_mult_base},
                                 {'params': param_new_layers, 'lr_mult': args.lr_mult_new}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay,
                                nesterov=True)

    # if args.resume and 'optimizer' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    ce_loss = torch.nn.CrossEntropyLoss().cuda(device=args.gpus[0])
    # mse_loss = torch.nn.MSELoss().cuda(device=args.gpus[0])
    lr_scheduler = load_lr_scheduler(args.lr_type, args.lr_steps, optimizer, len(train_iterator))

    new_top1, top1 = 0.0, 0.0
    train = train_mfnet_h
    test = test_mfnet_h
    for epoch in range(args.max_epochs):
        train(model_ft, optimizer, ce_loss, train_iterator, args.mixup_a, epoch, log_file, args.gpus,
                  lr_scheduler)
        if (epoch + 1) % args.eval_freq == 0:
            if args.eval_on_train:
                test(model_ft, ce_loss, train_iterator, epoch, "Train", log_file, args.gpus)
            new_top1 = test(model_ft, ce_loss, test_iterator, epoch, "Test", log_file, args.gpus)
            top1 = save_checkpoints(model_ft, optimizer, top1, new_top1,
                                    args.save_all_weights, output_dir, model_name, epoch,
                                    log_file)


if __name__ == '__main__':
    main()
