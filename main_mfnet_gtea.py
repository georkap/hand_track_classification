# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:04:16 2018

main train mfnet

@author: Γιώργος
"""

import torch
from torch.optim import SGD
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.mfnet_3d_mo import MFNET_3D as MFNET_3D_MO
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, save_mt_checkpoints, init_folders, resume_checkpoint
from utils.dataset_loader import FromVideoDatasetLoader, FromVideoDatasetLoaderGulp, VideoFromImagesDatasetLoader, prepare_sampler
from utils.dataset_loader_utils import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS, ToTensorVid, Normalize, Resize, CenterCrop
from utils.train_utils import load_lr_scheduler, train_mfnet_mo, test_mfnet_mo

mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]

GTEA_CLASSES = [106, 19, 53]
def main():
    args, model_name = parse_args('mfnet', val=False)
    model_name = 'gtea_' + model_name
    output_dir, log_file = init_folders(args.base_output_dir, model_name, args.resume, args.logging)
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)
    cudnn.benchmark = True

    mfnet_3d = MFNET_3D_MO  # mfnet 3d multi output
    kwargs = {}
    num_coords = 0
    objectives_text = "Objectives: "
    num_classes = [args.action_classes, args.verb_classes, args.noun_classes]
    num_objectives = 0
    if args.action_classes > 0:
        objectives_text += " actions {}, ".format(args.action_classes)
        num_objectives += 1
    if args.verb_classes > 0:
        objectives_text += " verbs {}, ".format(args.verb_classes)
        num_objectives += 1
    if args.noun_classes > 0:
        objectives_text += " nouns {}, ".format(args.noun_classes)
        num_objectives += 1
    if args.use_gaze:
        objectives_text += " gaze, "
        num_coords += 1
        num_objectives += 1
    if args.use_hands:
        objectives_text += " hands, "
        num_coords += 2
        num_objectives += 1
    kwargs["num_coords"] = num_coords
    print_and_save("Training for {} objective(s)".format(num_objectives), log_file)
    print_and_save(objectives_text, log_file)
    # for now just limit the tasks to max 3 and dont take extra nouns into account
    model_ft = mfnet_3d(num_classes, dropout=args.dropout, **kwargs)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained_model_path)
        # below line is needed if network is trained with DataParallel
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        base_dict = {k: v for k, v in list(base_dict.items()) if 'classifier' not in k}
        model_ft.load_state_dict(base_dict, strict=False)  # model.load_state_dict(checkpoint['state_dict'])
    model_ft.cuda(device=args.gpus[0])
    model_ft = torch.nn.DataParallel(model_ft, device_ids=args.gpus, output_device=args.gpus[0])
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)
    if args.resume: # resuming is untested for GTEA and multitask
        model_ft, ckpt_path = resume_checkpoint(model_ft, output_dir, model_name, args.resume_from)
        print_and_save("Resuming training from: {}".format(ckpt_path), log_file)

    # load dataset and train and validation iterators
    train_sampler = prepare_sampler("train", args.clip_length, args.frame_interval)
    train_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
        RandomCrop((224, 224)), RandomHorizontalFlip(), RandomHLS(vars=[15, 35, 25]),
        ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    # train_loader = FromVideoDatasetLoaderGulp(train_sampler, args.train_list, 'GTEA', num_classes, GTEA_CLASSES,
    #                                           use_gaze=args.use_gaze, gaze_list_prefix=args.gaze_list_prefix,
    #                                           use_hands=args.use_hands, hand_list_prefix=args.hand_list_prefix,
    #                                           batch_transform=train_transforms, extra_nouns=False)
    train_loader = VideoFromImagesDatasetLoader(train_sampler, args.train_list, 'GTEA', num_classes, GTEA_CLASSES,
                                                use_gaze=args.use_gaze, gaze_list_prefix=args.gaze_list_prefix,
                                                use_hands=args.use_hands, hand_list_prefix=args.hand_list_prefix,
                                                batch_transform=train_transforms, extra_nouns=False)
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers,
                                                 pin_memory=True)

    test_sampler = prepare_sampler("val", args.clip_length, args.frame_interval)
    test_transforms = transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)),
                                          ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    # test_loader = FromVideoDatasetLoaderGulp(test_sampler, args.test_list, 'GTEA', num_classes, GTEA_CLASSES,
    #                                          use_gaze=args.use_gaze, gaze_list_prefix=args.gaze_list_prefix,
    #                                          use_hands=args.use_hands, hand_list_prefix=args.hand_list_prefix,
    #                                          batch_transform=test_transforms, extra_nouns=False)
    test_loader = VideoFromImagesDatasetLoader(test_sampler, args.test_list, 'GTEA', num_classes, GTEA_CLASSES,
                                               use_gaze=args.use_gaze, gaze_list_prefix=args.gaze_list_prefix,
                                               use_hands=args.use_hands, hand_list_prefix=args.hand_list_prefix,
                                               batch_transform=test_transforms, extra_nouns=False)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers,
                                                pin_memory=True)

    # config optimizer
    param_base_layers = []
    param_new_layers = []
    name_base_layers = []
    for name, param in model_ft.named_parameters():
        if args.pretrained:
            # changing from startswith to this, because 'module' is added to the name of the parameter,
            # so the if clause doesn't work for multigpu training
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
    lr_scheduler = load_lr_scheduler(args.lr_type, args.lr_steps, optimizer, len(train_iterator))

    train = train_mfnet_mo
    test = test_mfnet_mo
    num_valid_classes = len([cls for cls in num_classes if cls > 0])
    new_top1, top1 = [0.0] * num_valid_classes, [0.0] * num_valid_classes
    for epoch in range(args.max_epochs):
        train(model_ft, optimizer, ce_loss, train_iterator, num_valid_classes, args.use_gaze, args.use_hands, epoch,
              log_file, args.gpus, lr_scheduler)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test(model_ft, ce_loss, train_iterator, num_valid_classes, args.use_gaze, args.use_hands, epoch,
                     "Train", log_file, args.gpus)
            new_top1 = test(model_ft, ce_loss, test_iterator, num_valid_classes, args.use_gaze, args.use_hands, epoch,
                            "Test", log_file, args.gpus)
            top1 = save_mt_checkpoints(model_ft, optimizer, top1, new_top1, args.save_all_weights, output_dir,
                                       model_name, epoch, log_file)


if __name__ == '__main__':
    main()
