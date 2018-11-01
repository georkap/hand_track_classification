# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:01:06 2018

argparse utils

@author: GEO
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Hand activity recognition')
    
    parser.add_argument('model_name', type=str, 
                        help="Naming convention:"\
                        +"resXXX_channels_batchsize_dropout_maxepochs"\
                        + "_pt if pretrained is used, _ft if only the last linear is trained"\
                        + "_interpolation_method, _bin or _smooth and _pad or nopad")
    # Load the necessary paths
    parser.add_argument('train_list', type=str)
    parser.add_argument('test_list', type=str)
    parser.add_argument('--base_output_dir', type=str, default=r'outputs/')
    
    # Dataset parameters
    parser.add_argument('--inter', type=str, default='cubic',
                        choices=['linear', 'cubic', 'nn', 'area', 'lanc', 'linext'])
    parser.add_argument('--bin_img', default=False, action='store_true')
    parser.add_argument('--pad', default=False, action='store_true')    
    
    # Model parameters
    parser.add_argument('--resnet_version', type=str, default='18', 
                        choices=['18','34','50','101','152'], 
                        help="One of 18, 34, 50, 101, 152")
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--feature_extraction', default=False, action='store_true')
    parser.add_argument('--channels', default='RGB', choices=['RGB', 'G'], help="optional to train on one input channel with binary inputs.")
    
    # Training parameters
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_steps', nargs='+', type=int, default=[7])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--max_epochs', type=int, default=20)
    
    # Program parameters
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--eval_on_train', default=False, action='store_true')
    parser.add_argument('--logging', default=False, action='store_true')
    
    args = parser.parse_args()
    return args

def parse_args_train_log_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log_path", type=str, help="The file to be analyzed", required=True)
    return parser.parse_args()

def parse_args_train_log_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log_dir", type=str, help="Dir containing the logs to be analyzed", required=True)
    return parser.parse_args()