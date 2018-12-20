# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:04:16 2018

main train mfnet

@author: Γιώργος
"""
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.mfnet_3d import MFNET_3D
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, save_checkpoints, init_folders
from utils.dataset_loader import VideoDatasetLoader
from utils.dataset_loader_utils import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS, ToTensorVid, Normalize, Resize, CenterCrop
from utils.train_utils import load_lr_scheduler, train_cnn, test_cnn
from utils.video_sampler import RandomSampling, SequentialSampling

mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]

def prepare_sampler(sampler_type, clip_length, frame_interval):
    if sampler_type == "train":
        train_sampler = RandomSampling(num=clip_length,
                                       interval=frame_interval,
                                       speed=[1.0, 1.0])
        out_sampler = train_sampler
    else:
        val_sampler = SequentialSampling(num=clip_length,
                                         interval=frame_interval,
                                         fix_cursor=True,
                                         shuffle=True)
        out_sampler = val_sampler
    return out_sampler

def main():
    args, model_name = parse_args('mfnet', val=False)
    
    output_dir, log_file = init_folders(args.base_output_dir, model_name, args.resume, args.logging)
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)    
    cudnn.benchmark = True

    model_ft = MFNET_3D(args.num_classes)
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
                                      num_classes=args.verb_classes, 
                                      batch_transform=train_transforms,
                                      img_tmpl='frame_{:010d}.jpg')
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers,
                                                 pin_memory=True)
    
    test_sampler = prepare_sampler("val", args.clip_length, args.frame_interval)
    test_transforms=transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)),
                                        ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    test_loader = VideoDatasetLoader(test_sampler, args.test_list, 
                                     num_classes=args.verb_classes,
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

    new_top1, top1 = 0.0, 0.0
    for epoch in range(args.epochs):
        train_cnn(model_ft, optimizer, ce_loss, train_iterator, args.mixup_a, epoch, log_file, lr_scheduler)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test_cnn(model_ft, ce_loss, train_iterator, epoch, "Train", log_file)
            new_top1 = test_cnn(model_ft, ce_loss, test_iterator, epoch, "Test", log_file)
            top1 = save_checkpoints(model_ft, optimizer, top1, new_top1,
                                    args.save_all_weights, output_dir, model_name, epoch,
                                    log_file)
            
if __name__ == '__main__':
    main()