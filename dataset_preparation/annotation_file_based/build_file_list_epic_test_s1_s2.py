# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:12:50 2019

Make unseen kitchen split epic

@author: Γιώργος
"""

import os, pandas

BASE_DIR = r"frames_rgb_flow\rgb\test"
SPLITS_DIR = r"..\splits\epic_rgb\test"
os.makedirs(SPLITS_DIR, exist_ok=True)

s1_test_name = os.path.join(SPLITS_DIR, "epic_rgb_s1.txt")
s2_test_name = os.path.join(SPLITS_DIR, "epic_rgb_s2.txt")

test_files = []
test_files.append(open(s1_test_name, 'a'))
test_files.append(open(s2_test_name, 'a'))

ANNOTS_FILE_S1 = r"D:\Datasets\egocentric\EPIC_KITCHENS\EPIC_test_s1_timestamps.csv"
ANNOTS_FILE_S2 = r"D:\Datasets\egocentric\EPIC_KITCHENS\EPIC_test_s2_timestamps.csv"
ANNOT_FILES = [ANNOTS_FILE_S1, ANNOTS_FILE_S2]

for (split_file, ANNOTATION_FILE) in zip(test_files, ANNOT_FILES):
    annotations = pandas.read_csv(ANNOTATION_FILE)
    for index, row in annotations.iterrows():
        start_frame = row.start_frame
        stop_frame = row.stop_frame
        num_frames = stop_frame - start_frame
        pid = row.participant_id
        uid = row.uid       
        videoid = row.video_id
        action_dir = os.path.join(BASE_DIR, pid, videoid)
        line = "{} {} {} {} {} {}\n".format(action_dir, num_frames, -1, -1, uid, start_frame)
        split_file.write(line)
    split_file.close()