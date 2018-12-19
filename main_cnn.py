# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:57:09 2018

training on the hand images using resnet

@author: Γιώργος
"""

import torch
import cv2

from models.resnet_zoo import resnet_loader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils.dataset_loader import ImageDatasetLoader
from utils.dataset_loader_utils import WidthCrop, RandomHorizontalFlip, Resize, ResizePadFirst, To01Range
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, save_checkpoints, init_folders
from utils.train_utils import load_lr_scheduler, train_cnn, test_cnn

meanRGB=[0.485, 0.456, 0.406]
stdRGB=[0.229, 0.224, 0.225]
meanG = [0.5]
stdG = [1.]

interpolation_methods = {'linear':cv2.INTER_LINEAR, 'cubic':cv2.INTER_CUBIC,
                         'nn':cv2.INTER_NEAREST, 'area':cv2.INTER_AREA,
                         'lanc':cv2.INTER_LANCZOS4, 'linext':cv2.INTER_LINEAR_EXACT}

def main():
    args, model_name = parse_args('resnet', val=False)

    output_dir, log_file = init_folders(args.base_output_dir, model_name, args.resume, args.logging)
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)
    cudnn.benchmark = True

    model_ft = resnet_loader(args.verb_classes, args.dropout, args.pretrained, 
                             args.feature_extraction, args.resnet_version, 
                             1 if args.channels == 'G' else 3,
                             args.no_resize)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    print_and_save("Model loaded to gpu", log_file)

    mean = meanRGB if args.channels == 'RGB' else meanG
    std = stdRGB if args.channels == 'RGB' else stdG
    normalize = transforms.Normalize(mean=mean, std=std)
    if args.no_resize:
        resize = WidthCrop()
    else:
        if args.pad:
            resize = ResizePadFirst(224, False, interpolation_methods[args.inter]) # currently set this binarize to False, because it is false duh
        else:
            resize = Resize((224,224), False, interpolation_methods[args.inter])
    
    train_transforms = transforms.Compose([resize, 
                                           RandomHorizontalFlip(), To01Range(args.bin_img),
                                           transforms.ToTensor(), normalize])
    train_loader = ImageDatasetLoader(args.train_list, num_classes=args.verb_classes, 
                                      batch_transform=train_transforms, channels=args.channels)
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    test_transforms = transforms.Compose([resize, To01Range(args.bin_img),
                                          transforms.ToTensor(), normalize])
    test_loader = ImageDatasetLoader(args.test_list, num_classes=args.verb_classes, 
                                     batch_transform=test_transforms, channels=args.channels)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

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

    lr_scheduler = load_lr_scheduler(args.lr_type, args.lr_steps, optimizer, len(train_iterator))

    new_top1, top1 = 0.0, 0.0
    for epoch in range(args.max_epochs):
        train_cnn(model_ft, optimizer, ce_loss, train_iterator, args.mixup_a, epoch, log_file, lr_scheduler)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test_cnn(model_ft, ce_loss, train_iterator, epoch, "Train", log_file)
            new_top1 = test_cnn(model_ft, ce_loss, test_iterator, epoch, "Test", log_file)
            top1 = save_checkpoints(model_ft, optimizer, top1, new_top1,
                                    args.save_all_weights, output_dir, model_name, epoch,
                                    log_file)
            
if __name__=='__main__':
    main()