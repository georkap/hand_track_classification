# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:19:35 2018

Training on the hand locations using lstm

@author: Γιώργος
"""

import sys
import torch
import torch.backends.cudnn as cudnn

from models.lstm_hands import LSTM_Hands, LSTM_per_hand, LSTM_Hands_attn
#from models.lstm_hands_enc_dec import LSTM_Hands_encdec
from utils.dataset_loader import PointDatasetLoader, PointVectorSummedDatasetLoader
from utils.dataset_loader_utils import lstm_collate
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save, save_checkpoints, resume_checkpoint, init_folders
from utils.train_utils import load_lr_scheduler, train_lstm, test_lstm, train_attn_lstm, test_attn_lstm

def main():
    args, model_name = parse_args('lstm', val=False)
    
    output_dir, log_file = init_folders(args.base_output_dir, model_name, args.resume, args.logging)
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)
    cudnn.benchmark = True

    lstm_model = LSTM_per_hand if args.lstm_dual else LSTM_Hands_attn if args.lstm_attn else LSTM_Hands
    kwargs = {'dropout':args.dropout, 'max_seq_len':args.lstm_seq_size}
    model_ft = lstm_model(args.lstm_input, args.lstm_hidden, args.lstm_layers, args.verb_classes, **kwargs)
#    model_ft = LSTM_Hands_encdec(456, 64, 32, args.lstm_layers, verb_classes, 0)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    if args.resume:
        model_ft = resume_checkpoint(model_ft, output_dir, model_name)
    print_and_save("Model loaded to gpu", log_file)

    norm_val = [1., 1., 1., 1.] if args.no_norm_input else [456., 256., 456., 256.]
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

    params_to_update = model_ft.parameters()
    print_and_save("Params to learn:", log_file)
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print_and_save("\t{}".format(name), log_file)

    optimizer = torch.optim.SGD(params_to_update,
                                lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    lr_scheduler = load_lr_scheduler(args.lr_type, args.lr_steps, optimizer, len(train_iterator))

    train_fun, test_fun = (train_attn_lstm, test_attn_lstm) if args.lstm_attn else (train_lstm, test_lstm)
    new_top1, top1 = 0.0, 0.0
    for epoch in range(args.max_epochs):
        train_fun(model_ft, optimizer, ce_loss, train_iterator, epoch, log_file, lr_scheduler)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test_fun(model_ft, ce_loss, train_iterator, epoch, "Train", log_file)
            new_top1 = test_fun(model_ft, ce_loss, test_iterator, epoch, "Test", log_file)
            top1 = save_checkpoints(model_ft, optimizer, top1, new_top1,
                                    args.save_all_weights, output_dir, model_name, epoch,
                                    log_file)
                

if __name__=='__main__':
    main()