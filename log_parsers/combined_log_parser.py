# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:22:09 2018

parser for multiple train logs combined
given a folder that contains the .txt logs
it will create a folder 'parsed_logs' with the graphs

1) combined test top1
2) combined test top5
3) combined test losses
4) combined avg train top1
5) combined avg train top5
6) combined avg train losses
and when possible
7) combined train top1
8) combined train top5
9) combined train losses

@author: Γιώργος
"""

import os
import numpy as np
from utils.argparse_utils import parse_args_train_log_dir
from utils.file_utils import get_eval_results, get_train_results, make_plot_dataframe

args = parse_args_train_log_dir() #--train_log_dir
LOG_DIR = args.train_log_dir
output_dir = os.path.join(LOG_DIR, "parsed_logs")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

log_file_names = [x for x in os.listdir(LOG_DIR) if x.endswith('.txt')]
log_files = [os.path.join(LOG_DIR, x) for x in log_file_names]

all_names = ["Test acc@1", "Test acc@5", "Test losses", "Train acc@1", "Train acc@5", "Train losses",
             "Avg train acc@1", "Avg train acc@5", "Avg train loss"]

test_losses = {}
test_top1s = {}
test_top5s = {}
train_losses = {}
train_top1s = {}
train_top5s = {}
avg_train_losses = {}
avg_train_top1s = {}
avg_train_top5s = {}
filenames = []
max_epochs = 0
for LOG_FILE in log_files:
    with open(LOG_FILE) as f:
        lines = f.readlines()
        
    name_parts = os.path.basename(LOG_FILE).split(".")
    file_name = ""
    for part in name_parts[:-1]:
        file_name += part
    filenames.append(file_name)
    
    test_epochs, test_loss, test_top1, test_top5 = get_eval_results(lines, "Test")
    train_epochs, train_loss, train_top1, train_top5 = get_eval_results(lines, "Train")
    epochs, avg_loss, avg_t1, avg_t5 = get_train_results(lines)
    
    if len(epochs) > max_epochs:
        max_epochs = len(epochs)
    assert test_epochs == epochs # should work
    
    test_losses[file_name] = test_loss
    test_top1s[file_name] = test_top1
    test_top5s[file_name] = test_top5
    train_losses[file_name] = train_loss
    train_top1s[file_name] = train_top1
    train_top5s[file_name] = train_top5
    avg_train_losses[file_name] = avg_loss
    avg_train_top1s[file_name] = avg_t1
    avg_train_top5s[file_name] = avg_t5


np_columns = np.column_stack([test_top1s[fname] + [0]*(max_epochs - len(test_top1s[fname])) for fname in filenames])
df_test_top1 = make_plot_dataframe(np_columns, filenames, all_names[0],
                                   os.path.join(output_dir, "results_test_top1.png"))

np_columns = np.column_stack([test_top5s[fname] + [0]*(max_epochs - len(test_top5s[fname])) for fname in filenames])
df_test_top5 = make_plot_dataframe(np_columns, filenames, all_names[1],
                                   os.path.join(output_dir, "results_test_top5.png"))

np_columns = np.column_stack([test_losses[fname] + [0]*(max_epochs - len(test_losses[fname])) for fname in filenames])
df_test_loss = make_plot_dataframe(np_columns, filenames, all_names[2], 
                                os.path.join(output_dir, "results_test_loss.png"))
    
np_columns = np.column_stack([train_top1s[fname] + [0]*(max_epochs - len(train_top1s[fname])) for fname in filenames])
df_train_top1 = make_plot_dataframe(np_columns, filenames, all_names[3], 
                                os.path.join(output_dir, "results_train_top1.png"))

np_columns = np.column_stack([train_top5s[fname] + [0]*(max_epochs - len(train_top5s[fname])) for fname in filenames])
df_train_top5 = make_plot_dataframe(np_columns, filenames, all_names[4], 
                                    os.path.join(output_dir, "results_train_top5.png"))

np_columns = np.column_stack([train_losses[fname] + [0]*(max_epochs - len(train_losses[fname])) for fname in filenames])
df_train_loss = make_plot_dataframe(np_columns, filenames, all_names[5], 
                                    os.path.join(output_dir, "results_train_loss.png"))

np_columns = np.column_stack([avg_train_top1s[fname] + [0]*(max_epochs - len(avg_train_top1s[fname])) for fname in filenames])
df_avg_train_top1 = make_plot_dataframe(np_columns, filenames, all_names[6], 
                                os.path.join(output_dir, "results_avg_train_top1.png"))

np_columns = np.column_stack([avg_train_top5s[fname] + [0]*(max_epochs - len(avg_train_top5s[fname])) for fname in filenames])
df_avg_train_top5 = make_plot_dataframe(np_columns, filenames, all_names[7], 
                                        os.path.join(output_dir, "results_avg_train_top5.png"))
np_columns = np.column_stack([avg_train_losses[fname] + [0]*(max_epochs - len(avg_train_losses[fname])) for fname in filenames])
df_avg_train_losses = make_plot_dataframe(np_columns, filenames, all_names[8], 
                                          os.path.join(output_dir, "results_avg_train_loss.png"))



    
    
    
    