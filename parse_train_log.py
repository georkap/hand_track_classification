# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:22:46 2018

train_log parser to graphs 

@author: Γιώργος
"""

import os
import numpy as np

from utils.argparse_utils import parse_args_train_log_file
from utils.file_utils import get_eval_results, get_train_results, make_plot_dataframe, get_loss_over_lr

args = parse_args_train_log_file() #--train_log_path
LOG_FILE = args.train_log_path
file_name = os.path.basename(LOG_FILE).split(".")[0]
LOG_DIR = os.path.dirname(LOG_FILE)
output_dir = os.path.join(LOG_DIR, "parsed_log")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(LOG_FILE) as f:
    lines = f.readlines()


test_epochs, test_loss, test_top1, test_top5 = get_eval_results(lines, "Test")
train_epochs, train_loss, train_top1, train_top5 = get_eval_results(lines, "Train")

epochs, avg_loss, avg_t1, avg_t5 = get_train_results(lines)

all_names = ["test acc@1", "test acc@5", "test loss", "train acc@1", "train acc@5", "train loss",
             "avg train acc@1", "avg train acc@5", "avg train loss"]

test_columns = np.column_stack([test_top1, test_top5, test_loss])
train_columns = np.column_stack([train_top1, train_top5, train_loss])
avg_train_columns = np.column_stack([avg_t1, avg_t5, avg_loss])
test_names = all_names[:3]
train_names = all_names[3:6]
avg_train_names = all_names[6:]

if not args.no_lr_graph:
    avg_train_loss, lrs = get_loss_over_lr(lines)
#    from matplotlib import pyplot as plt
#    title="{}\n Avg Train loss versus learning rate per batch".format(file_name)
#    file = os.path.join(output_dir, "results_lr.png")
#    plot = plt.plot(avg_train_loss, lrs)
#    plt.title(title)
#    plt.xlabel("learning rate")
#    plt.ylabel("loss")
#    plt.legend(bbox_to_anchor=(0, -0.06), loc="upper left")
#    plt.tight_layout()
#    plt.savefig(file)
    
    df_lr = make_plot_dataframe(np.column_stack([avg_train_loss, lrs]), 
                                ["avg train loss", "learning_rate"],
                                "{}\n Avg Train loss versus learning rate per batch".format(file_name),
                                os.path.join(output_dir, "results_lr.png"))

# all results
df_all = make_plot_dataframe(np.column_stack([test_columns, avg_train_columns]), 
                             test_names+avg_train_names, 
                             "{}\n Test, Avg Train".format(file_name), 
                             os.path.join(output_dir, "results_all.png"))

# text for google docs accumulation
with open(os.path.join(output_dir, "text_for_google_docs.txt"), 'a') as f:
    for i, val in enumerate(df_all.values):
        print(df_all.index.tolist()[i], end=',', file=f)
        for j in val:
            print(j, end=',',file=f)
        print(file=f)

# test only
make_plot_dataframe(test_columns, test_names,
                    "{}\n Eval on test set".format(file_name),
                    os.path.join(output_dir, "results_test.png"))

# avg train only
make_plot_dataframe(avg_train_columns, avg_train_names,
                    "{}\n Avg from training".format(file_name),
                    os.path.join(output_dir, "results_avg_train.png"))

# eval on train set if exists
if len(train_loss)>0:
    make_plot_dataframe(train_columns, train_names,
                        "{}\n Eval on train set".format(file_name),
                        os.path.join(output_dir, "results_train.png"))
    
# losses
make_plot_dataframe(np.column_stack([test_loss, avg_loss]),
                    [all_names[2], all_names[8]],
                    "{}\n Test and Avg Train losses".format(file_name),
                    os.path.join(output_dir, "results_loss.png"))

