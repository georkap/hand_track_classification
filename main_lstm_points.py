# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:19:35 2018

Training on the hand locations using resnet

@author: Γιώργος
"""

import os
import time
import shutil
import torch
from models.lstm_hands import LSTM_Hands
import torch.backends.cudnn as cudnn
from utils.dataset_loader import PointDatasetLoader
from utils.dataset_loader_utils import lstm_collate
from utils.calc_utils import AverageMeter, accuracy
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, print_model_config
from utils.train_utils import CyclicLR

def train(model, optimizer, criterion, train_iterator, cur_epoch, log_file, lr_scheduler):
    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    
    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()
    
    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, seq_lengths, targets) in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()    
        
        inputs = torch.tensor(inputs, requires_grad=True).cuda()
        targets = torch.tensor(targets).cuda()

        inputs = inputs.transpose(1,0)
        output = model(inputs, seq_lengths)

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
        print_and_save('[Epoch:{}, Batch {}/{} in {:.3f} s][Loss {:.4f}[avg:{:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]], LR {:.6f}'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg, 
                lr_scheduler.get_lr()[0]), log_file)

def test(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, seq_lengths, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            inputs = inputs.transpose(1,0)
            output = model(inputs, seq_lengths)
            
            loss = criterion(output, targets)

            t1, t5 = accuracy(output.detach().cpu(), targets.detach().cpu(), topk=(1,5))
            top1.update(t1.item(), inputs.size(0))
            top5.update(t5.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                    cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg

norm_val = [456., 256., 456., 256.]

def main():
    args = parse_args()
    verb_classes = 120
    
    base_output_dir = args.base_output_dir
    model_name = args.model_name
    train_list = args.train_list # r'splits\hand_tracks\hand_locs_train_1.txt'
    test_list = args.test_list # r'splits\hand_tracks\hand_locs_val_1.txt'
    
    output_dir = os.path.join(base_output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    if args.logging:
        log_file = os.path.join(base_output_dir, model_name, model_name+".txt")
    else:
        log_file = None
        
    print_model_config(args, log_file)

    # 120, 0.5, True, False, '18', 
    model_ft = LSTM_Hands(4, args.lstm_hidden, args.lstm_layers, verb_classes)
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
    
#    train_transforms = transforms.Compose([torch.from_numpy])
    train_loader = PointDatasetLoader(train_list, norm_val=norm_val)
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lstm_collate, pin_memory=True)

#    test_transforms = transforms.Compose([torch.from_numpy])
    test_loader = PointDatasetLoader(test_list, norm_val=norm_val)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=lstm_collate, pin_memory=True)

    if args.lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=int(args.lr_steps[0]),
                                                       gamma=float(args.lr_steps[1]))
    elif args.lr_type == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(x) for x in args.lr_steps[:-1]],
                                                            gamma=float(args.lr_steps[-1]))
    elif args.lr_type == 'clr':
        lr_scheduler = CyclicLR(optimizer, base_lr=float(args.lr_steps[0]), 
                                max_lr=float(args.lr_steps[1]), step_size_up=int(args.lr_steps[2])*len(train_iterator),
                                step_size_down=int(args.lr_steps[3])*len(train_iterator), mode=str(args.lr_steps[4]),
                                gamma=float(args.lr_steps[5]))

    new_top1, top1 = 0.0, 0.0
    isbest = False
    for epoch in range(args.max_epochs):
        train(model_ft, optimizer, ce_loss, train_iterator, epoch, log_file, lr_scheduler)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test(model_ft, ce_loss, train_iterator, epoch, "Train", log_file)
            new_top1 = test(model_ft, ce_loss, test_iterator, epoch, "Test", log_file)
            isbest = True if new_top1 >= top1 else False
            
            if args.save_all_weights:
                weight_file = os.path.join(output_dir, model_name + '_{:03d}.pth'.format(epoch))
            else:
                weight_file = os.path.join(output_dir, model_name + '_ckpt.pth')
            print_and_save('Saving weights to {}'.format(weight_file), log_file)
            torch.save({'epoch': epoch + 1,
                        'state_dict': model_ft.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'top1': new_top1}, weight_file)
            if isbest:
                best = os.path.join(output_dir, model_name+'_best.pth')
                shutil.copyfile(weight_file, best)
                top1 = new_top1
                

if __name__=='__main__':
    main()