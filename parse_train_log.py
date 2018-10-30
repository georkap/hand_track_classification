# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:22:46 2018

train_log parser to graphs 

@author: Γιώργος
"""

import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log_path", type=str, help="The file to be analyzed", required=True)
    return parser.parse_args()

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

def make_plot_dataframe(np_columns, str_columns, title, file):
    df = pd.DataFrame(data=np_columns, columns=str_columns)
    plot = df.plot(title=title)
    fig=plot.get_figure()
    fig.savefig(file)
    
    return df

args = parse_args()
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

