# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:01:06 2018

argparse utils

@author: GEO
"""

import argparse

def parse_args_val():
    parser = argparse.ArgumentParser(description='Hand activity recognition - validation')
    
    # Load the necessary paths
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('val_list', type=str)
    
    # Network configuration
    parser.add_argument('--resnet_version', type=str, default=None, 
                        choices=['18','34','50','101','152'], 
                        help="One of 18, 34, 50, 101, 152")
    parser.add_argument('--channels', default=None, choices=['RGB', 'G'], 
                        help="optional to train on one input channel with binary inputs.")
    parser.add_argument('--lstm_input', type=int, default=4)
    parser.add_argument('--lstm_hidden', type=int, default=8)
    parser.add_argument('--lstm_layers', type=int, default=2)

    # Dataset parameters
    parser.add_argument('--verb_classes', type=int, default=120)
    parser.add_argument('--no_resize', default=False, action='store_true')
    parser.add_argument('--inter', type=str, default='cubic',
                        choices=['linear', 'cubic', 'nn', 'area', 'lanc', 'linext'])
    parser.add_argument('--bin_img', default=False, action='store_true')
    parser.add_argument('--pad', default=False, action='store_true')    
    parser.add_argument('--no_norm_input', default=False, action='store_true', help='lstm')
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Program parameters
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--logging', default=False, action='store_true')

    args = parser.parse_args()
    return args

def make_base_parser(val):
    parser = argparse.ArgumentParser(description='Hand activity recognition')

    # Load the necessary paths    
    if not val:
        parser.add_argument('train_list', type=str)
        parser.add_argument('test_list', type=str)
        parser.add_argument('--base_output_dir', type=str, default=r'outputs/')
        parser.add_argument('--model_name', type=str, default=None, help='if left to None it will be automatically created from the args')
    else:
        parser.add_argument('ckpt_path', type=str)
        parser.add_argument('val_list', type=str)
    
    return parser
    
def parse_args_dataset(parser, net_type):
    # Dataset parameters
    parser.add_argument('--verb_classes', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=1)
    if net_type == 'resnet':
        parser.add_argument('--no_resize', default=False, action='store_true', help='keep the original 256x456 image size for the network input')
        parser.add_argument('--inter', type=str, default='cubic',
                            choices=['linear', 'cubic', 'nn', 'area', 'lanc', 'linext'])
        parser.add_argument('--bin_img', default=False, action='store_true')
        parser.add_argument('--pad', default=False, action='store_true')
    elif net_type == 'lstm':
        parser.add_argument('--lstm_feature', default='coords',
                            choices=['coords', 'coords_dual', 'vec_sum', 'vec_sum_dual'],
                            help="lstm_input changes based on the choice."\
                            + "For: coords 4, coords_dual 2," \
                            + "vec_sum 712 (i.e. 256+456), vec_sum_dual 712")
        parser.add_argument('--lstm_clamped', default=False, action='store_true', 
                            help='will remove the non existing hand points in a sequence and result in each hand having a starting sequence of different length. Sampling from these sequences is possible afterwards. Works only for dual lstm and coords feature.')
        parser.add_argument('--no_norm_input', default=False, action='store_true', help='will not normalize input to 0-1')
    
    return parser
    
def parse_args_network(parser, net_type):
    # Network configuration
    if net_type == 'resnet':
        parser.add_argument('--resnet_version', type=str, default='18', 
                            choices=['18','34','50','101','152'], 
                            help="One of 18, 34, 50, 101, 152")
        parser.add_argument('--pretrained', default=False, action='store_true')
        parser.add_argument('--feature_extraction', default=False, action='store_true')
        parser.add_argument('--channels', default='RGB', choices=['RGB', 'G'], help="optional to train on one input channel with binary inputs.")
    elif net_type == 'lstm':
        parser.add_argument('--lstm_input', type=int, default=4)
        parser.add_argument('--lstm_hidden', type=int, default=8)
        parser.add_argument('--lstm_layers', type=int, default=2)
        parser.add_argument('--lstm_dual', default=False, action='store_true')
        parser.add_argument('--lstm_seq_size', type=int, default=0, help="If not 0, it will perform a uniform sampling over the sequence to get to the desired number.")
        
    return parser
    
def parse_args_training(parser):
    # Training parameters
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--mixup_a', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_type', type=str, default='step',
                        choices=['step', 'multistep', 'clr'])
    parser.add_argument('--lr_steps', nargs='+', type=str, default=[7],
                        help="The value of lr_steps depends on lr_type. If lr_type is:"\
                            +"'step' then lr_steps is a list of size 2 that contains the number of epochs needed to reduce the lr at lr_steps[0] and the gamma to reduce by, at lr_steps[1]."\
                            +"'multistep' then lr_steps is a list of size n+1 for n number of learning rate decreases and the gamma to reduce by at lr_steps[-1]."\
                            +"'clr' then lr_steps is a list of size 6: [base_lr, max_lr, num_epochs_up, num_epochs_down, mode, gamma]. In the clr case, argument 'lr' is ignored.")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--max_epochs', type=int, default=20)
    
    return parser
    
def parse_args_program(parser):
    # Program parameters
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--eval_on_train', default=False, action='store_true')
    parser.add_argument('--save_all_weights', default=False, action='store_true')
    parser.add_argument('--logging', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    
    return parser
    
def parse_args(net_type, val=False):
    parser = make_base_parser(val)
    parser = parse_args_dataset(parser, net_type)
    parser = parse_args_network(parser, net_type)
    parser = parse_args_training(parser)
    parser = parse_args_program(parser)
    
    args = parser.parse_args()
    if not val:
        if args.model_name is None:
            model_name = make_model_name(args, net_type)        
            return args, model_name
        return args, args.model_name
    return args
    
def make_model_name(args, net_type):
    if net_type == 'resnet':
        model_name = "res{}_{}_{}_{}".format(args.resnet_version, args.channels,
                         args.batch_size, args.dropout, args.max_epochs)
        if args.pretrained:
            model_name = model_name + "_pt"
        if args.feature_extraction:
            model_name = model_name + "_ft"
        
        model_name = model_name + "_{}".format(args.inter)
        if args.bin_img:
            model_name = model_name + "_bin"
        else:
            model_name = model_name + "_smooth"
        if args.pad:
            model_name = model_name + "_pad"
        else:
            model_name = model_name + "_nopad"
        
    elif net_type == 'lstm':
        model_name = "lstm"
        model_name = model_name + "_{}_{}_{}".format(args.batch_size, 
                                    args.dropout, args.max_epochs)
        if args.lstm_dual:
            model_name = model_name + "_dual"
        model_name = model_name + "_{}_{}_{}".format(args.lstm_input,
                                    args.lstm_hidden, args.lstm_layers)
        model_name = model_name + "_seq{}".format(args.lstm_seq_size if args.lstm_seq_size != 0 else "full")
        model_name = model_name + "_{}".format("coords" if args.lstm_feature.startswith('coord') else "vec")
        if args.lstm_clamped:
            model_name = model_name + "_clamped"
        if args.no_norm_input:
            model_name = model_name + "_no_norm"
    
    model_name = model_name + "_{}".format(args.lr_type)
    if args.lr_type == "clr":
        clr_type = "tri" if args.lr_steps[4] == "triangular" else "tri2" if args.lr_steps[4] == "triangular2" else "exp"
        model_name = model_name + "_{}".format(clr_type)
    if args.verb_classes != 120:
        model_name = model_name + "_sel{}".format(args.verb_classes)
    if args.mixup_a != 1.:
        model_name = model_name + "_mixup"   
    
    return model_name

#def parse_args():
#    parser = argparse.ArgumentParser(description='Hand activity recognition')
#    
#    parser.add_argument('model_name', type=str, 
#                        help="Naming convention:"\
#                        +"resXXX_channels_batchsize_dropout_maxepochs"\
#                        + "_pt if pretrained is used, _ft if only the last linear is trained"\
#                        + "_interpolation_method, _bin or _smooth and _pad or nopad")
#    # Load the necessary paths
#    parser.add_argument('train_list', type=str)
#    parser.add_argument('test_list', type=str)
#    parser.add_argument('--base_output_dir', type=str, default=r'outputs/')
#    
#    # Dataset parameters
#    parser.add_argument('--verb_classes', type=int, default=120)
#    parser.add_argument('--no_resize', default=False, action='store_true')
#    parser.add_argument('--inter', type=str, default='cubic',
#                        choices=['linear', 'cubic', 'nn', 'area', 'lanc', 'linext'])
#    parser.add_argument('--bin_img', default=False, action='store_true')
#    parser.add_argument('--pad', default=False, action='store_true')    
#    parser.add_argument('--no_norm_input', default=False, action='store_true', help='lstm')
#    
#    # Network configuration
#    parser.add_argument('--resnet_version', type=str, default='18', 
#                        choices=['18','34','50','101','152'], 
#                        help="One of 18, 34, 50, 101, 152")
#    parser.add_argument('--lstm_input', type=int, default=4)
#    parser.add_argument('--lstm_hidden', type=int, default=8)
#    parser.add_argument('--lstm_layers', type=int, default=2)
#    parser.add_argument('--pretrained', default=False, action='store_true')
#    parser.add_argument('--feature_extraction', default=False, action='store_true')
#    parser.add_argument('--channels', default='RGB', choices=['RGB', 'G'], help="optional to train on one input channel with binary inputs.")
#    
#    # Training parameters
#    parser.add_argument('--dropout', type=float, default=0.5)
#    parser.add_argument('--mixup_a', type=float, default=1)
#    parser.add_argument('--batch_size', type=int, default=1)
#    parser.add_argument('--lr', type=float, default=0.001)
#    parser.add_argument('--lr_type', type=str, default='step',
#                        choices=['step', 'multistep', 'clr'])
#    parser.add_argument('--lr_steps', nargs='+', type=str, default=[7],
#                        help="The value of lr_steps depends on lr_type. If lr_type is:"\
#                            +"'step' then lr_steps is a list of size 2 that contains the number of epochs needed to reduce the lr at lr_steps[0] and the gamma to reduce by, at lr_steps[1]."\
#                            +"'multistep' then lr_steps is a list of size n+1 for n number of learning rate decreases and the gamma to reduce by at lr_steps[-1]."\
#                            +"'clr' then lr_steps is a list of size 6: [base_lr, max_lr, num_epochs_up, num_epochs_down, mode, gamma]. In the clr case, argument 'lr' is ignored.")
#    parser.add_argument('--momentum', type=float, default=0.9)
#    parser.add_argument('--decay', type=float, default=0.0005)
#    parser.add_argument('--max_epochs', type=int, default=20)
#    
#    # Program parameters
#    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
#    parser.add_argument('--num_workers', type=int, default=8)
#    parser.add_argument('--eval_freq', type=int, default=1)
#    parser.add_argument('--eval_on_train', default=False, action='store_true')
#    parser.add_argument('--save_all_weights', default=False, action='store_true')
#    parser.add_argument('--logging', default=False, action='store_true')
#    
#    args = parser.parse_args()
#    return args

def parse_args_train_log_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log_path", type=str, help="The file to be analyzed", required=True)
    parser.add_argument("--no_lr_graph", default=False, action='store_true')
    return parser.parse_args()

def parse_args_train_log_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log_dir", type=str, help="Dir containing the logs to be analyzed", required=True)
    parser.add_argument("--walk", default=False, action='store_true')
    return parser.parse_args()