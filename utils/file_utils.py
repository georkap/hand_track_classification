# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:58:26 2018

file_utils

@author: GEO
"""
import os
import sys
import torch
import shutil
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt

def init_folders(base_output_dir, model_name, resume, logging):
    output_dir = os.path.join(base_output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if not resume:
            sys.exit("Attempted to overwrite previous folder, exiting..")
    
    log_file = os.path.join(base_output_dir, model_name, model_name+".txt") if logging else None
    return output_dir, log_file

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

def get_train_results_hands(lines):
    epochs = [int(line.strip().split(":")[1]) for line in lines if line.startswith("Beginning")]
    avg_loss, avg_loss_cls, avg_loss_coo, avg_t1, avg_t5 = [], [], [], [], []
    train_start = False
    for i, line in enumerate(lines):
        if line.startswith("Beginning"):
            train_start = True
            continue
        if train_start and line.startswith("Evaluating"):
            loss_avg, loss_cls_avg, loss_coo_avg, t1_avg, t5_avg = parse_train_line_hands(lines[i-1])
            avg_loss.append(loss_avg)
            avg_loss_cls.append(loss_cls_avg)
            avg_loss_coo.append(loss_coo_avg)
            avg_t1.append(t1_avg)
            avg_t5.append(t5_avg)
            train_start = False
    return epochs, avg_loss, avg_loss_cls, avg_loss_coo, avg_t1, avg_t5

def parse_train_line_hands(line):
    loss = line.split("avg:")[1].split("]")[0]
    loss_avg = float(loss.split(' ')[0])
    loss_avg_cls = float(loss.split(' ')[2])
    loss_avg_coo = float(loss.split(' ')[4])

    t1_avg = float(line.split("avg:")[2].split("]")[0])
    t5_avg = float(line.split("avg:")[3].split("]")[0])

    return loss_avg, loss_avg_cls, loss_avg_coo, t1_avg, t5_avg

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

def save_best_checkpoint(top1, new_top1, output_dir, model_name, weight_file):
    isbest = True if new_top1 >= top1 else False
    if isbest:
        best = os.path.join(output_dir, model_name+'_best.pth')
        shutil.copyfile(weight_file, best)
        top1 = new_top1
    return top1


def save_checkpoints(model_ft, optimizer, top1, new_top1,
                     save_all_weights, output_dir, model_name, epoch, log_file):
    if save_all_weights:
        weight_file = os.path.join(output_dir, model_name + '_{:03d}.pth'.format(epoch))
    else:
        weight_file = os.path.join(output_dir, model_name + '_ckpt.pth')
    print_and_save('Saving weights to {}'.format(weight_file), log_file)
    if not isinstance(top1, tuple):
        torch.save({'epoch': epoch,
                    'state_dict': model_ft.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'top1': new_top1}, weight_file)
        top1 = save_best_checkpoint(top1, new_top1, output_dir, model_name, weight_file)
    else:
        torch.save({'epoch':epoch,
                    'state_dict': model_ft.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'top1_a': new_top1[0],
                    'top1_b': new_top1[1]}, weight_file)
        top1_a = save_best_checkpoint(top1[0], new_top1[0], output_dir, 
                                      model_name + "_verb", weight_file)
        top1_b = save_best_checkpoint(top1[1], new_top1[1], output_dir, 
                                      model_name + "_noun", weight_file)
        top1 = (top1_a, top1_b)
    return top1

def save_prev_checkpoint(pth_path):
    ckpt_name_parts = os.path.basename(pth_path).split(".")
    old_ckpt_name = ""
    for part in ckpt_name_parts[:-1]:
        old_ckpt_name += part
    dtm = datetime.fromtimestamp(os.path.getmtime(pth_path))
    old_ckpt = os.path.join(os.path.dirname(pth_path), old_ckpt_name + "_{}{}_{}{}.pth".format(dtm.day, dtm.month, dtm.hour, dtm.minute))
    shutil.copyfile(pth_path, old_ckpt)

def prepare_resume_latest(output_dir, model_name, resume_from):
    old_ckpt_path = os.path.join(output_dir, model_name + '_ckpt.pth')
    save_prev_checkpoint(old_ckpt_path)
    old_best_path = os.path.join(output_dir, model_name + '_best.pth')
    save_prev_checkpoint(old_best_path)
    
    ckpt_path = old_ckpt_path if resume_from == "ckpt" else old_best_path
    return ckpt_path

def load_checkpoint(ckpt_path, model_ft):
    checkpoint = torch.load(ckpt_path)    
    model_ft.load_state_dict(checkpoint['state_dict'])
    return model_ft

def resume_checkpoint(model_ft, output_dir, model_name, resume_from):
    if resume_from in ["ckpt", "best"]: 
        # resume from ckpt or best and rename the previous weights
        ckpt_path = prepare_resume_latest(output_dir, model_name, resume_from)        
    else: # using a new model structure so the old weights are not overwritten or we just dont care
        ckpt_path = resume_from
    
    return load_checkpoint(ckpt_path, model_ft), ckpt_path

def get_log_files(LOG_DIR, pattern, walk):
    log_files = []
    if walk:
        for root, dirs, files in os.walk(LOG_DIR):
            for file in files:
                if file.startswith(pattern) and file.endswith('.txt'):
                    log_files.append(os.path.join(root, file))
    else:
        log_file_names = [x for x in os.listdir(LOG_DIR) if x.endswith('.txt')]
        log_files = [os.path.join(LOG_DIR, x) for x in log_file_names]
    
    return log_files

def parse_log_file_name(name_parts):
    batch_size = int(name_parts[0].split('_')[1])
    dropout = float(name_parts[1].split('_')[0])/10
    epochs = int(name_parts[1].split('_')[1])
    input_size = int(name_parts[1].split('_')[2])
    hidden_size = int(name_parts[1].split('_')[3])
    num_layers = int(name_parts[1].split('_')[4])
    seq_size = int(name_parts[1].split('_')[5][-2:])
    feature = name_parts[1].split('_')[6]
    lr_type = name_parts[1].split('_')[7]
    dataset = name_parts[1].split('_')[-1]
    return batch_size, dropout, epochs, input_size, hidden_size, num_layers, seq_size, feature, lr_type, dataset