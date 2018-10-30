# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:58:26 2018

file_utils

@author: GEO
"""
import os

def print_and_save(text, path):
    print(text)
    if path is not None:
        with open(path, 'a') as f:
            print(text, file=f)
            
def print_model_config(args, log_file):
    to_print = "Model config {}\n".format(args.channels)
    to_print += "Resnet {}, Using pretrained weights {}, Only train last linear {}\n".format(args.resnet_version, args.pretrained, args.feature_extraction)
    to_print += "Parameters: Batch size {}, Learning rate {}, LR decrease steps {}, Momentum {}, Weight decay {}\n".format(args.batch_size, args.lr, args.lr_steps, args.momentum, args.decay)
    to_print += "Stop after {} epochs, evaluate every {} epoch(s), save weights every {} epoch(s)\n".format(args.max_epochs, args.eval_freq, args.eval_freq)
    to_print += "Name {}\nTo train on {}\nTo test on {}\n".format(args.model_name, args.train_list, args.test_list)
    to_print += "Output will be saved at {}\n".format(os.path.join(args.base_output_dir, args.model_name))
    print_and_save(to_print, log_file)