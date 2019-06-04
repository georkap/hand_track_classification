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

from models.mfnet_3d_mo import MFNET_3D as MFNET_3D_MO
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, save_mt_checkpoints, init_folders
from utils.dataset_loader import FromVideoDatasetLoader, prepare_sampler
from utils.dataset_loader_utils import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS, ToTensorVid, Normalize, Resize, CenterCrop
from utils.train_utils import load_lr_scheduler, CyclicLR
from utils.calc_utils import AverageMeter, accuracy

mean_3d = [124 / 255, 117 / 255, 104 / 255]
std_3d = [0.229, 0.224, 0.225]

def train_cnn_mo(model, optimizer, criterion, train_iterator, num_outputs, cur_epoch, log_file, gpus, lr_scheduler=None):
    batch_time = AverageMeter()
    loss_meters = [AverageMeter() for _ in range(num_outputs)]
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_outputs)]

    model.train()
    
    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()
    
    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()

        inputs = inputs.cuda(gpus[0])
        targets = targets.cuda(gpus[0]).transpose(0,1) # needs transpose to get the first dim to be the task and the second dim to be the batch
        assert len(targets) == num_outputs

        outputs = model(inputs)

        losses_per_task = []
        for output, target in zip(outputs, targets):
            loss_for_task = criterion(output, target)
            losses_per_task.append(loss_for_task)

        loss = sum(losses_per_task)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metrics
        batch_size = outputs[0].size(0)
        losses.update(loss.item(), batch_size)

        for ind in range(num_outputs):
            t1, t5 = accuracy(outputs[ind].detach().cpu(), targets[ind].detach().cpu(), topk=(1,5))
            top1_meters[ind].update(t1.item(), batch_size)
            top5_meters[ind].update(t5.item(), batch_size)
            loss_meters[ind].update(losses_per_task[ind].item(), batch_size)

        batch_time.update(time.time() - t0)
        t0 = time.time()
        to_print = '[Epoch:{}, Batch {}/{} in {:.3f} s]'.format(cur_epoch, batch_idx, len(train_iterator), batch_time.val)
        to_print += '[Losses {:.4f}[avg:{:.4f}], '.format(losses.val, losses.avg)
        for ind in range(num_outputs):
            to_print += 'T{}::loss {:.4f}[avg:{:.4f}], '.format(ind, loss_meters[ind].val ,loss_meters[ind].avg)
        for ind in range(num_outputs):
            to_print += 'T{}::Top1 {:.3f}[avg:{:.3f}],Top5 {:.3f}[avg:{:.3f}],'.format(
                ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)
        to_print += 'LR {:.6f}'.format(lr_scheduler.get_lr()[0])
        print_and_save(to_print, log_file)

def test_cnn_mo(model, criterion, test_iterator, num_outputs, cur_epoch, dataset, log_file, gpus):
    loss_meters = [AverageMeter() for _ in range(num_outputs)]
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_outputs)]
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = inputs.cuda(gpus[0])
            targets = targets.cuda(gpus[0]).transpose(0, 1)

            outputs = model(inputs)

            losses_per_task = [criterion(output, target) for output, target in zip(outputs, targets)]
            loss = torch.sum(torch.tensor(losses_per_task).cuda())

            # update metrics
            batch_size = outputs[0].size(0)
            losses.update(loss.item(), batch_size)

            for ind in range(num_outputs):
                t1, t5 = accuracy(outputs[ind].detach().cpu(), targets[ind].detach().cpu(), topk=(1, 5))
                top1_meters[ind].update(t1.item(), batch_size)
                top5_meters[ind].update(t5.item(), batch_size)
                loss_meters[ind].update(losses_per_task[ind].item(), batch_size)

            to_print = '[Epoch:{}, Batch {}/{}] '.format(cur_epoch, batch_idx, len(test_iterator))
            for ind in range(num_outputs):
                to_print += 'T{}::Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}],'.format(
                    ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)

            print_and_save(to_print, log_file)

        final_print = '{} Results: Loss {:.3f},'.format(dataset, losses.avg)
        for ind in range(num_outputs):
            final_print += 'T{}::Top1 {:.3f}, Top5 {:.3f},'.format(ind, top1_meters[ind].avg, top5_meters[ind].avg)
        print_and_save(final_print, log_file)
    return [tasktop1.avg for tasktop1 in top1_meters]

GTEA_CLASSES = [106, 19, 53]
def main():
    args, model_name = parse_args('mfnet', val=False)
    model_name = 'gtea_' + model_name
    output_dir, log_file = init_folders(args.base_output_dir, model_name, args.resume, args.logging)
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)    
    cudnn.benchmark = True
    
    mfnet_3d = MFNET_3D_MO # mfnet 3d multi output
    objectives_text = "Objectives: "
    num_classes = [args.action_classes, args.verb_classes, args.noun_classes]
    if args.action_classes > 0:
        objectives_text += " actions {}, ".format(args.action_classes)
    if args.verb_classes > 0:
        # num_classes.append(args.verb_classes)
        objectives_text += " verbs {}, ".format(args.verb_classes)
    if args.noun_classes > 0:
        # num_classes.append(args.noun_classes)
        objectives_text += " nouns {}, ".format(args.noun_classes)
    print_and_save("Training for {} objective(s)".format(len(num_classes)), log_file)
    print_and_save(objectives_text, log_file)
    # for now just limit the tasks to max 3 and dont take extra nouns into account
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
    train_loader = FromVideoDatasetLoader(train_sampler, args.train_list, 'GTEA', num_classes, GTEA_CLASSES,
                                          batch_transform=train_transforms, extra_nouns=False)
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers,
                                                 pin_memory=True)
    
    test_sampler = prepare_sampler("val", args.clip_length, args.frame_interval)
    test_transforms=transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)),
                                        ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    test_loader = FromVideoDatasetLoader(test_sampler, args.test_list, 'GTEA', num_classes, GTEA_CLASSES,
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

    if args.resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    ce_loss = torch.nn.CrossEntropyLoss().cuda(device=args.gpus[0])
    lr_scheduler = load_lr_scheduler(args.lr_type, args.lr_steps, optimizer, len(train_iterator))

    train = train_cnn_mo
    test = test_cnn_mo
    num_valid_classes = len([cls for cls in num_classes if cls > 0])
    new_top1, top1 = [0.0] * num_valid_classes, [0.0] * num_valid_classes
    for epoch in range(args.max_epochs):
        train(model_ft, optimizer, ce_loss, train_iterator, num_valid_classes, epoch, log_file, args.gpus, lr_scheduler)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test(model_ft, ce_loss, train_iterator, num_valid_classes, epoch, "Train", log_file, args.gpus)
            new_top1 = test(model_ft, ce_loss, test_iterator, num_valid_classes, epoch, "Test", log_file, args.gpus)
            top1 = save_mt_checkpoints(model_ft, optimizer, top1, new_top1,
                                       args.save_all_weights, output_dir, model_name, epoch, log_file)
            
if __name__ == '__main__':
    main()