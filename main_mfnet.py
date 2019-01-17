# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:04:16 2018

main train mfnet

@author: Γιώργος
"""
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.mfnet_3d import MFNET_3D
from models.mfnet_3d_do import MFNET_3D as MFNET_3D_DO
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, save_checkpoints, init_folders
from utils.dataset_loader import VideoDatasetLoader, prepare_sampler
from utils.dataset_loader_utils import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS, ToTensorVid, Normalize, Resize, CenterCrop
from utils.train_utils import load_lr_scheduler, CyclicLR, mixup_data, mixup_criterion
from utils.calc_utils import AverageMeter, accuracy

mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]

def train_cnn_do(model, optimizer, criterion, train_iterator, mixup_alpha, cur_epoch, log_file, gpus, lr_scheduler=None):
    batch_time, losses_a, losses_b, losses, top1_a, top5_a, top1_b, top5_b = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    
    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()
    
    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()
            
        inputs = torch.tensor(inputs, requires_grad=True).cuda(gpus[0])

        targets_a = torch.tensor(targets[0]).cuda(gpus[0])
        targets_b = torch.tensor(targets[1]).cuda(gpus[0])
        
        output_a, output_b = model(inputs)

        loss_a = criterion(output_a, targets_a)
        loss_b = criterion(output_b, targets_b)
        loss = 0.75*loss_a + 0.25*loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1_a, t5_a = accuracy(output_a.detach().cpu(), targets_a.detach().cpu(), topk=(1,5))
        t1_b, t5_b = accuracy(output_b.detach().cpu(), targets_b.detach().cpu(), topk=(1,5))
        top1_a.update(t1_a.item(), output_a.size(0))
        top5_a.update(t5_a.item(), output_a.size(0))
        top1_b.update(t1_b.item(), output_b.size(0))
        top5_b.update(t5_b.item(), output_b.size(0))
        losses_a.update(loss_a.item(), output_a.size(0))
        losses_b.update(loss_b.item(), output_b.size(0))
        losses.update(loss.item(), output_a.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        to_print = '[Epoch:{}, Batch {}/{} in {:.3f} s]'\
                   '[Losses {:.4f}[avg:{:.4f}], loss_a {:.4f}[avg:{:.4f}], loss_b {:.4f}[avg:{:.4f}],' \
                   'Top1_a {:.3f}[avg:{:.3f}], Top5_a {:.3f}[avg:{:.3f}],' \
                   'Top1_b {:.3f}[avg:{:.3f}], Top5_b {:.3f}[avg:{:.3f}]],' \
                   'LR {:.6f}'.format(
                           cur_epoch, batch_idx, len(train_iterator), batch_time.val,
                           losses_a.val, losses_a.avg, losses_b.val, losses_b.avg, losses.val, losses.avg,
                           top1_a.val, top1_a.avg, top5_a.val, top5_a.avg,
                           top1_b.val, top1_b.avg, top5_b.val, top5_b.avg,
                           lr_scheduler.get_lr()[0])
        print_and_save(to_print, log_file)

def test_cnn_do(model, criterion, test_iterator, cur_epoch, dataset, log_file, gpus):
    losses, losses_a, losses_b, top1_a, top5_a, top1_b, top5_b = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda(gpus[0])
            
            targets_a = torch.tensor(targets[0]).cuda(gpus[0])
            targets_b = torch.tensor(targets[1]).cuda(gpus[0])        
            output_a, output_b = model(inputs)
            loss_a = criterion(output_a, targets_a)
            loss_b = criterion(output_b, targets_b)
            loss = 0.75*loss_a + 0.25*loss_b

            t1_a, t5_a = accuracy(output_a.detach().cpu(), targets_a.detach().cpu(), topk=(1,5))
            t1_b, t5_b = accuracy(output_b.detach().cpu(), targets_b.detach().cpu(), topk=(1,5))
            top1_a.update(t1_a.item(), output_a.size(0))
            top5_a.update(t5_a.item(), output_a.size(0))
            top1_b.update(t1_b.item(), output_b.size(0))
            top5_b.update(t5_b.item(), output_b.size(0))
            losses_a.update(loss_a.item(), output_a.size(0))
            losses_b.update(loss_b.item(), output_b.size(0))
            losses.update(loss.item(), output_a.size(0))
            
            to_print = '[Epoch:{}, Batch {}/{}]' \
                       '[Top1_a {:.3f}[avg:{:.3f}], Top5_a {:.3f}[avg:{:.3f}],' \
                       'Top1_b {:.3f}[avg:{:.3f}], Top5_b {:.3f}[avg:{:.3f}]]'.format(
                       cur_epoch, batch_idx, len(test_iterator),
                       top1_a.val, top1_a.avg, top5_a.val, top5_a.avg,
                       top1_b.val, top1_b.avg, top5_b.val, top5_b.avg)
            print_and_save(to_print, log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1_a {:.3f}, Top5_a {:.3f}, Top1_b {:.3f}, Top5_b {:.3f}'.format(dataset, losses.avg, top1_a.avg, top5_a.avg, top1_b.avg, top5_b.avg), log_file)
    return top1_a.avg, top1_b.avg

def train_cnn(model, optimizer, criterion, train_iterator, mixup_alpha, cur_epoch, log_file, gpus, lr_scheduler=None):
    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    
    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()
    
    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()
            
        inputs = torch.tensor(inputs, requires_grad=True).cuda(gpus[0])
        targets = torch.tensor(targets).cuda(gpus[0])

        # TODO: Fix mixup and cuda integration, especially for mfnet
        if mixup_alpha != 1:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
        
        output = model(inputs)

        if mixup_alpha != 1:
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        
#        if clip_gradient is not None:
#            total_norm = torch.nn.clip_grad_norm_(model.parameters(), clip_gradient)
#            if total_norm > clip_gradient:
#                to_print = "clipping gradient: {} with coef {}".format(total_norm, clip_gradient / total_norm)
#                print_and_save(to_print, log_file)
        
        optimizer.step()

        t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1,5))
        top1.update(t1.item(), output.size(0))
        top5.update(t5.item(), output.size(0))
        losses.update(loss.item(), output.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        print_and_save('[Epoch:{}, Batch {}/{} in {:.3f} s][Loss {:.4f}[avg:{:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]], LR {:.6f}'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg, 
                lr_scheduler.get_lr()[0]), log_file)

def test_cnn(model, criterion, test_iterator, cur_epoch, dataset, log_file, gpus):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda(gpus[0])
            targets = torch.tensor(targets).cuda(gpus[0])

            output = model(inputs)
            loss = criterion(output, targets)

            t1, t5 = accuracy(output.detach().cpu(), targets.detach().cpu(), topk=(1,5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))

            print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                    cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg

def main():
    args, model_name = parse_args('mfnet', val=False)
    
    output_dir, log_file = init_folders(args.base_output_dir, model_name, args.resume, args.logging)
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)    
    cudnn.benchmark = True
    
    mfnet_3d = MFNET_3D if not args.double_output else MFNET_3D_DO
    num_classes = args.verb_classes if not args.double_output else (args.verb_classes, args.noun_classes)
    model_ft = mfnet_3d(num_classes, dropout=args.dropout)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained_model_path)
        # below line is needed if network is trained with DataParallel
        base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
        base_dict = {k:v for k, v in list(base_dict.items()) if 'classifier' not in k}
        model_ft.load_state_dict(base_dict, strict=False) #model.load_state_dict(checkpoint['state_dict'])
    model_ft.cuda(device=args.gpus[0])
    model_ft = torch.nn.DataParallel(model_ft, device_ids=args.gpus, output_device=args.gpus[0])
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)

    # load dataset and train and validation iterators
    train_sampler = prepare_sampler("train", args.clip_length, args.frame_interval)
    train_transforms = transforms.Compose([
            RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
            RandomCrop((224, 224)), RandomHorizontalFlip(), RandomHLS(vars=[15, 35, 25]),
            ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    train_loader = VideoDatasetLoader(train_sampler, args.train_list, 
                                      num_classes=num_classes, 
                                      batch_transform=train_transforms,
                                      img_tmpl='frame_{:010d}.jpg')
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers,
                                                 pin_memory=True)
    
    test_sampler = prepare_sampler("val", args.clip_length, args.frame_interval)
    test_transforms=transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)),
                                        ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    test_loader = VideoDatasetLoader(test_sampler, args.test_list, 
                                     num_classes=num_classes,
                                     batch_transform=test_transforms,
                                     img_tmpl='frame_{:010d}.jpg')
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers,
                                                pin_memory=True)

    # config optimizatερ
    param_base_layers = []
    param_new_layers = []
    name_base_layers = []
    for name, param in model_ft.named_parameters():
        if args.pretrained:
            if name.startswith('classifier'):
                param_new_layers.append(param)
            else:
                param_base_layers.append(param)
                name_base_layers.append(name)
        else:
            param_new_layers.append(param)

    optimizer = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': 0.2},
                                 {'params': param_new_layers, 'lr_mult': 1.0}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay,
                                nesterov=True)

    if args.resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    ce_loss = torch.nn.CrossEntropyLoss().cuda(device=args.gpus[0])
    lr_scheduler = load_lr_scheduler(args.lr_type, args.lr_steps, optimizer, len(train_iterator))

    if not args.double_output:
        new_top1, top1 = 0.0, 0.0
    else:
        new_top1, top1 = (0.0, 0.0), (0.0, 0.0)
    train = train_cnn if not args.double_output else train_cnn_do
    test = test_cnn if not args.double_output else test_cnn_do
    for epoch in range(args.max_epochs):
        train(model_ft, optimizer, ce_loss, train_iterator, args.mixup_a, epoch, log_file, args.gpus, lr_scheduler)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test(model_ft, ce_loss, train_iterator, epoch, "Train", log_file, args.gpus)
            new_top1 = test(model_ft, ce_loss, test_iterator, epoch, "Test", log_file, args.gpus)
            top1 = save_checkpoints(model_ft, optimizer, top1, new_top1,
                                    args.save_all_weights, output_dir, model_name, epoch,
                                    log_file)
            
if __name__ == '__main__':
    main()