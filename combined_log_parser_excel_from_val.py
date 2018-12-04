# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:42:11 2018

combined log parser only excel output from results_accuracy_validation.txt
(Temp file to avoid rerunning all previous experiments)

@author: Γιώργος
"""

import os
import pandas
import numpy as np
from utils.argparse_utils import parse_args_train_log_dir

args = parse_args_train_log_dir() #--train_log_dir
LOG_DIR = args.train_log_dir
output_dir = os.path.join(LOG_DIR, "_combined_results", "parsed_logs")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

excel_writer = pandas.ExcelWriter(os.path.join(output_dir, "best_eval_results.xlsx"))

columns=["file","top1","epoch","loss","top5"]
df_all = pandas.DataFrame(columns=columns)
for root, dirs, files in os.walk(LOG_DIR):
    for file in files:
        if file=="results-accuracy-validation.txt":
            file_name = os.path.join(root, file)
            with open(file_name) as f:
                lines = f.readlines()
            epoch = int(lines[2].split()[3])
            res_line = lines[-30].strip().split()
            loss = float(res_line[3].rstrip(","))
            top1 = float(res_line[5].rstrip(","))
            top5 = float(res_line[7])
            name = os.path.basename(os.path.dirname(file_name))
            
            data = []
            data.append(name)
            data.append(top1)
            data.append(epoch)
            data.append(loss)
            data.append(top5)
            df_to_append=pandas.DataFrame(data=[data], columns=columns)
            df_all = df_all.append(df_to_append)

df_all.to_excel(excel_writer)

excel_writer.save()