# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:57:09 2018

training on the hand images using resnet

@author: Γιώργος
"""

import os
import time
import shutil
import torch
import cv2
from models.resnet_zoo import resnet_loader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils.dataset_loader import DatasetLoader, RandomHorizontalFlip, Resize, ResizePadFirst, To01Range
from utils.calc_utils import AverageMeter, accuracy
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, print_model_config

def train(model, optimizer, criterion, train_iterator, cur_epoch, log_file):
    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    
    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        inputs = torch.tensor(inputs, requires_grad=True).cuda()
        targets = torch.tensor(targets).cuda()

        output = model(inputs)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1,5))
        top1.update(t1.item(), inputs.size(0))
        top5.update(t5.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        print_and_save('[Epoch:{}, Batch {}/{} in {:.3f} s][Loss {:.4f}[avg:{:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg), log_file)

def test(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            output = model(inputs)
            loss = criterion(output, targets)

            t1, t5 = accuracy(output.detach().cpu(), targets.detach().cpu(), topk=(1,5))
            top1.update(t1.item(), inputs.size(0))
            top5.update(t5.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                    cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg

meanRGB=[0.485, 0.456, 0.406]
stdRGB=[0.229, 0.224, 0.225]
meanG = [0.5]
stdG = [1.]

interpolation_methods = {'linear':cv2.INTER_LINEAR, 'cubic':cv2.INTER_CUBIC,
                         'nn':cv2.INTER_NEAREST, 'area':cv2.INTER_AREA,
                         'lanc':cv2.INTER_LANCZOS4, 'linext':cv2.INTER_LINEAR_EXACT}

def main():
    args = parse_args()
    verb_classes = 120
    
    base_output_dir = args.base_output_dir
    model_name = args.model_name
    train_list = args.train_list # r'splits\hand_track_train_1.txt'
    test_list = args.test_list # r'splits\hand_track_val_1.txt'
    
    output_dir = os.path.join(base_output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    if args.logging:
        log_file = os.path.join(base_output_dir, model_name, model_name+".txt")
    else:
        log_file = None
        
    print_model_config(args, log_file)
    
    mean = meanRGB if args.channels == 'RGB' else meanG
    std = stdRGB if args.channels == 'RGB' else stdG

    model_ft = resnet_loader(verb_classes, args.dropout, args.pretrained, args.feature_extraction, args.resnet_version) # 120, True, False
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    print_and_save("Model loaded to gpu", log_file)
    cudnn.benchmark = True

    params_to_update = model_ft.parameters()
    print_and_save("Params to learn:", log_file)
    if args.feature_extraction:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print_and_save("\t{}".format(name), log_file)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print_and_save("\t{}".format(name), log_file)

    optimizer = torch.optim.SGD(params_to_update,
                                lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_steps[0])

    normalize = transforms.Normalize(mean=mean, std=std)

    if args.pad:
        resize = ResizePadFirst(224, args.bin_img, interpolation_methods[args.inter])
    else:
        resize = Resize((224,224), args.bin_img, interpolation_methods[args.inter])
    
    train_transforms = transforms.Compose([resize, 
                                           RandomHorizontalFlip(), To01Range(),
                                           transforms.ToTensor(), normalize])
    train_loader = DatasetLoader(train_list, train_transforms)
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    test_transforms = transforms.Compose([resize, To01Range(),
                                          transforms.ToTensor(), normalize])
    test_loader = DatasetLoader(test_list, test_transforms)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    new_top1, top1 = 0.0, 0.0
    isbest = False
    for epoch in range(args.max_epochs):
        lr_scheduler.step()
        train(model_ft, optimizer, ce_loss, train_iterator, epoch, log_file)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test(model_ft, ce_loss, train_iterator, epoch, "Train", log_file)
            new_top1 = test(model_ft, ce_loss, test_iterator, epoch, "Test", log_file)
            isbest = True if new_top1 >= top1 else False
            top1 = new_top1
            
            weight_file = os.path.join(output_dir, model_name + '_{:03d}.pth'.format(epoch))
            print_and_save('Saving weights to {}'.format(weight_file), log_file)
            torch.save({'epoch': epoch + 1,
                        'state_dict': model_ft.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'top1': new_top1}, weight_file)
            if isbest:
                best = os.path.join(output_dir, model_name+'_best.pth')
                shutil.copyfile(weight_file, best)

if __name__=='__main__':
    main()