# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:19:35 2018

Training on the hand locations using lstm

@author: Γιώργος
"""

import os
import sys
import time
import torch
from models.lstm_hands import LSTM_Hands, LSTM_per_hand
#from models.lstm_hands_enc_dec import LSTM_Hands_encdec
import torch.backends.cudnn as cudnn
from utils.dataset_loader import PointDatasetLoader, PointVectorSummedDatasetLoader
from utils.dataset_loader_utils import lstm_collate
from utils.calc_utils import AverageMeter, accuracy
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, save_checkpoints, resume_checkpoint
from utils.train_utils import CyclicLR, load_lr_scheduler

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

        inputs = inputs.transpose(1, 0)
        output = model(inputs, seq_lengths)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
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

def test(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, seq_lengths, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            inputs = inputs.transpose(1, 0)
            output = model(inputs, seq_lengths)
            
            loss = criterion(output, targets)

            t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1,5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))                
            losses.update(loss.item(), output.size(0))

            print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                    cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg

def main():
    args, model_name = parse_args('lstm', val=False)
    
    output_dir = os.path.join(args.base_output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if not args.resume:
            sys.exit("Attempted to overwrite previous folder, exiting..")
    
    log_file = os.path.join(args.base_output_dir, model_name, model_name+".txt") if args.logging else None
        
    norm_val = [1., 1., 1., 1.] if args.no_norm_input else [456., 256., 456., 256.]
        
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)

    lstm_model = LSTM_per_hand if args.lstm_dual else LSTM_Hands

    model_ft = lstm_model(args.lstm_input, args.lstm_hidden, args.lstm_layers, args.verb_classes, args.dropout)
#    model_ft = LSTM_Hands_encdec(456, 64, 32, args.lstm_layers, verb_classes, 0)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    print_and_save("Model loaded to gpu", log_file)
    cudnn.benchmark = True
    
    if args.resume:
        model_ft = resume_checkpoint(model_ft, output_dir, model_name)
    
    params_to_update = model_ft.parameters()
    print_and_save("Params to learn:", log_file)
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print_and_save("\t{}".format(name), log_file)

    optimizer = torch.optim.SGD(params_to_update,
                                lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    
    if args.lstm_feature == "coords" or args.lstm_feature == "coords_dual":
        if args.lstm_clamped and (not args.lstm_dual or args.lstm_seq_size == 0):
            sys.exit("Clamped tracks require dual lstms and a fixed lstm sequence size.")
        train_loader = PointDatasetLoader(args.train_list, max_seq_length=args.lstm_seq_size,
                                          num_classes=args.verb_classes, norm_val=norm_val,
                                          dual=args.lstm_dual, clamp=args.lstm_clamped)
        test_loader = PointDatasetLoader(args.test_list, max_seq_length=args.lstm_seq_size,
                                         num_classes=args.verb_classes, norm_val=norm_val, 
                                         dual=args.lstm_dual, clamp=args.lstm_clamped)
    elif args.lstm_feature == "vec_sum" or args.lstm_feature == "vec_sum_dual":
        train_loader = PointVectorSummedDatasetLoader(args.train_list, 
                                                      max_seq_length=args.lstm_seq_size,
                                                      num_classes=args.verb_classes, 
                                                      dual=args.lstm_dual)
        test_loader = PointVectorSummedDatasetLoader(args.test_list,
                                                     max_seq_length=args.lstm_seq_size,
                                                     num_classes=args.verb_classes,
                                                     dual=args.lstm_dual)
    else:
        sys.exit("Unsupported lstm feature")
#    train_loader = PointImageDatasetLoader(train_list, norm_val=norm_val)  
#    test_loader = PointImageDatasetLoader(test_list, norm_val=norm_val)

    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lstm_collate, pin_memory=True)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lstm_collate, pin_memory=True)

    lr_scheduler = load_lr_scheduler(args.lr_type, args.lr_steps, optimizer, len(train_iterator))

    new_top1, top1 = 0.0, 0.0
    for epoch in range(args.max_epochs):
        train(model_ft, optimizer, ce_loss, train_iterator, epoch, log_file, lr_scheduler)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test(model_ft, ce_loss, train_iterator, epoch, "Train", log_file)
            new_top1 = test(model_ft, ce_loss, test_iterator, epoch, "Test", log_file)
            top1 = save_checkpoints(model_ft, optimizer, top1, new_top1,
                                    args.save_all_weights, output_dir, model_name, epoch,
                                    log_file)
                

if __name__=='__main__':
    main()