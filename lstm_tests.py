# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:37:30 2018

hyperparameter search for lstm training

Given: small dataset, no dropout, no bn, 1 layer lstm

To find: max-min lr, momentum and wd optimal values

@author: Γιώργος
"""

import os
import shutil
import torch
import torch.backends.cudnn as cudnn
from models.lstm_hands import LSTM_Hands
from utils.dataset_loader import PointDatasetLoader
from utils.dataset_loader_utils import lstm_collate

from utils.file_utils import print_and_save
from utils.train_utils import CyclicLR, train, test

base_output_dir = r"outputs\hyperparams"
train_list = r"splits\hand_tracks_select\hand_locs_train_1.txt"
test_list = r"splits\hand_tracks_select\hand_locs_val_1.txt"
save_all_weights = False

model_name = "lrrangetest"
output_dir = os.path.join(base_output_dir, model_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

norm_val = [456., 256., 456., 256.]
lstm_input = 4; lstm_hidden = 16; lstm_layers=1; verb_classes=2; dropout=0;
start_lr = 0.001; start_mom = 0.9; start_decay = 0.001
batch_size = 128
max_epochs = 100

for lr in [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.04, 0.07, 0.1, 0.5, 1]:
    log_file = os.path.join(base_output_dir, model_name, model_name+"_{}.txt".format(lr))
    
    model_ft = LSTM_Hands(lstm_input, lstm_hidden, lstm_layers, verb_classes, dropout)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    cudnn.benchmark = True
    
    params_to_update = model_ft.parameters()
    print_and_save("Params to learn:", log_file)
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print_and_save("\t{}".format(name), log_file)
    
    optimizer = torch.optim.SGD(params_to_update,
                                lr=lr, momentum=start_mom, weight_decay=start_decay)
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    
    train_loader = PointDatasetLoader(train_list, max_seq_length=16,
                                      norm_val=norm_val, dual=False, clamp=False)
    test_loader = PointDatasetLoader(test_list, max_seq_length=16,
                                     norm_val=norm_val, dual=False, clamp=False)
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lstm_collate, pin_memory=True)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lstm_collate, pin_memory=True)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max_epochs)
    
    
    new_top1, top1 = 0.0, 0.0
    isbest = False
    for epoch in range(max_epochs):
        train(model_ft, optimizer, ce_loss, train_iterator, epoch, log_file, lr_scheduler)
        
        new_top1 = test(model_ft, ce_loss, test_iterator, epoch, "Test", log_file)
#        isbest = True if new_top1 >= top1 else False
#            
#        if save_all_weights:
#            weight_file = os.path.join(output_dir, model_name + '_{:03d}.pth'.format(epoch))
#        else:
#            weight_file = os.path.join(output_dir, model_name + '_ckpt.pth')
#        print_and_save('Saving weights to {}'.format(weight_file), log_file)
#        torch.save({'epoch': epoch + 1,
#                    'state_dict': model_ft.state_dict(),
#                    'optimizer': optimizer.state_dict(),
#                    'top1': new_top1}, weight_file)
#        if isbest:
#            best = os.path.join(output_dir, model_name+'_best.pth')
#            shutil.copyfile(weight_file, best)
#            top1 = new_top1