# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:58:26 2018

file_utils

@author: GEO
"""
import os
import pandas as pd
from matplotlib import pyplot as plt

def print_and_save(text, path):
    print(text)
    if path is not None:
        with open(path, 'a') as f:
            print(text, file=f)
            
def print_model_config(args, log_file):
    to_print = "Model config {}\n".format(args.channels)
    to_print += "Resnet {}, Using pretrained weights {}, Only train last linear {}\n".format(args.resnet_version, args.pretrained, args.feature_extraction)
    to_print += "Parameters: Batch size {}, Learning rate {}, LR scheduler {}, LR options {}, Momentum {}, Weight decay {}\n".format(args.batch_size, args.lr, args.lr_type, args.lr_steps, args.momentum, args.decay)
    to_print += "Stop after {} epochs, evaluate every {} epoch(s), save weights every {} epoch(s)\n".format(args.max_epochs, args.eval_freq, args.eval_freq)
    to_print += "Name {}\nTo train on {}\nTo test on {}\n".format(args.model_name, args.train_list, args.test_list)
    to_print += "Output will be saved at {}\n".format(os.path.join(args.base_output_dir, args.model_name))
    print_and_save(to_print, log_file)
    
def get_eval_results(lines, test_set):
    epochs = [int(line.strip().split(":")[1]) for line in lines if line.startswith("Beginning")]
    res = [line.strip() for line in lines if line.startswith(test_set)]
    if len(res)==0:
        return epochs, [], [], []
    loss, top1, top5 = [], [], []    
    for r in res:
        l = float(r.split(":")[1].split(",")[0].split()[1])
        t1 = float(r.split(":")[1].split(",")[1].split()[1])
        t5 = float(r.split(":")[1].split(",")[2].split()[1])
        loss.append(l)
        top1.append(t1)
        top5.append(t5)
    
    assert len(epochs) == len(loss) == len(top1) == len(top5)
    
    return epochs, loss, top1, top5

def parse_train_line(line):
    epoch = int(line.split("Epoch:")[1].split(",")[0])
    batch = line.split("Batch ")[1].split()[0]
    b0 = int(batch.split("/")[0])
    b1 = int(batch.split("/")[1])

    loss_val = float(line.split("[avg:")[0].split()[-1])
    top1_val = float(line.split("[avg:")[1].split()[-1])
    top5_val = float(line.split("[avg:")[2].split()[-1])
    loss_avg = float(line.split("avg:")[1].split("]")[0])
    t1_avg = float(line.split("avg:")[2].split("]")[0])
    t5_avg = float(line.split("avg:")[3].split("]")[0])
    
    return loss_val, top1_val, top5_val, loss_avg, t1_avg, t5_avg

def get_train_results(lines):
    epochs = [int(line.strip().split(":")[1]) for line in lines if line.startswith("Beginning")]
    avg_loss, avg_t1, avg_t5 = [], [], []
    train_start = False
    for i, line in enumerate(lines):
        if line.startswith("Beginning"):
            train_start = True
            continue
        if train_start and line.startswith("Evaluating"):
            loss_val, top1_val, top5_val, loss_avg, t1_avg, t5_avg = parse_train_line(lines[i-1])
            avg_loss.append(loss_avg)
            avg_t1.append(t1_avg)
            avg_t5.append(t5_avg)
            train_start = False
    
    return epochs, avg_loss, avg_t1, avg_t5

def parse_train_line_lr(line):
    loss_avg = float(line.split("avg:")[1].split("]")[0])
    lr = float(line.split()[-1])

    return loss_avg, lr

def get_loss_over_lr(lines):
    avg_loss, lrs = [], []
    train_start = False
    for i, line in enumerate(lines):
        if line.startswith("Beginning"):
            train_start=True
            continue
        if train_start and line.startswith("Evaluating"):
            train_start=False
            continue
        if train_start:
            loss_avg, lr = parse_train_line_lr(line)
            avg_loss.append(loss_avg)
            lrs.append(lr)
    
    return avg_loss, lrs

def make_plot_dataframe(np_columns, str_columns, title, file):
    df = pd.DataFrame(data=np_columns, columns=str_columns)
    plot = df.plot(title=title).legend(bbox_to_anchor=(0, -0.06), loc='upper left')
    plt.tight_layout()
    fig=plot.get_figure()
    fig.savefig(file)
    
    return df