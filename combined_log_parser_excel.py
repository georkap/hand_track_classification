# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:32:37 2018

combined log parser only excel output

@author: Γιώργος
"""

import os
import pandas
import numpy as np
from utils.argparse_utils import parse_args_train_log_dir
from utils.file_utils import get_eval_results, get_log_files, parse_log_file_name

args = parse_args_train_log_dir() #--train_log_dir
LOG_DIR = args.train_log_dir
output_dir = os.path.join(LOG_DIR, "_excel")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

excel_writer = pandas.ExcelWriter(os.path.join(output_dir, "best_test_results.xlsx"))

log_files = get_log_files(LOG_DIR, "lstm_", args.walk)
    
columns=["file","top1-1","top1-2","top1-3","top1-4","top1-5","epoch","loss","top5",
         "batch_size", "max_epochs", "hidden", "layers", "seq_size"]
df_all = pandas.DataFrame(columns=columns)
for LOG_FILE in log_files:
    with open(LOG_FILE) as f:
        lines = f.readlines()
        
    name_parts = os.path.basename(LOG_FILE).split(".")
    file_name = ""
    for part in name_parts[:-1]:
        file_name += part
    batch_size, dropout, max_epochs, input_size, hidden_size, num_layers, seq_size, feature, lr_type, dataset = parse_log_file_name(name_parts)
    
    test_epochs, test_loss, test_top1, test_top5 = get_eval_results(lines, "Test")
    sort_inds = np.argsort(test_top1)
    
    data = []
    data.append(file_name)
    for i in range(1,6):
        data.append(test_top1[sort_inds[-i]])
    data.append(test_epochs[sort_inds[-1]])
    data.append(test_loss[sort_inds[-1]])
    data.append(test_top5[sort_inds[-1]])
    data.append(batch_size)
    data.append(max_epochs)
    data.append(hidden_size)
    data.append(num_layers)
    data.append(seq_size)
    df_to_append=pandas.DataFrame(data=[data], columns=columns)
    df_all = df_all.append(df_to_append)

df_all.to_excel(excel_writer)

excel_writer.save()