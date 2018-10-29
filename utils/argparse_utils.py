# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:01:06 2018

argparse utils

@author: GEO
"""

import argparse
from utils.file_utils import print_and_save

def parse_args():
    parser = argparse.ArgumentParser(description='Hand activity recognition')
    
    # First load the necessary paths
    parser.add_argument('--output_dir', type=str, default=r'backup/')
    
    # Model parameters
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--finetune', default=False, action='store_true')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--max_epochs', type=int, default=None)
    
    # Program parameters
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval_freq', type=int, default=2)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--logging', default=False, action='store_true')
    
    args = parser.parse_args()
    return args