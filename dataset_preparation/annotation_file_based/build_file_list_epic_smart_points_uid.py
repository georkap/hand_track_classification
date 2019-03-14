# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:01:53 2019

Make splits on the train set of epic kitchens for hand track activity recognition

Similar to build_file_list_epic_smart_points.py with the difference that the track
files are now based on the uid so I can make the train/test lists from the annotatation file
of epic kitchens

@author: Γιώργος
"""
import os, pandas

unavailable = [9, 11, 18]
available_pids = ["P{:02d}".format(i) for i in range(1,32) if i not in unavailable]

split_1 = {}
for i in range(28):
    split_1[available_pids[i]] = "train" if i < 21 else "val"

split_dicts = [split_1]

BASE_DIR = r"hand_detection_tracks_fd_train"
SPLITS_DIR = r"..\..\splits\hand_tracks_fd_train"
os.makedirs(SPLITS_DIR, exist_ok=True)

train_names = os.path.join(SPLITS_DIR, "hand_locs_train_{}.txt".format(1))
val_names = os.path.join(SPLITS_DIR, "hand_locs_val_{}.txt".format(1))

train_files = []
val_files = []
train_files.append(open(train_names, 'a'))
val_files.append(open(val_names, 'a'))
    
ANNOTATION_FILE = r"D:\Datasets\egocentric\EPIC_KITCHENS\EPIC_train_action_labels.csv"
annotations = pandas.read_csv(ANNOTATION_FILE)
for index, row in annotations.iterrows():
    start_frame = row.start_frame
    stop_frame = row.stop_frame
    num_frames = stop_frame - start_frame
    verb_class = row.verb_class
    noun_class = row.noun_class
    pid = row.participant_id
    uid = row.uid       
    videoid = row.video_id
    filename = "{}_{}.pkl".format(uid, start_frame)
    action_dir = os.path.join(BASE_DIR, pid, videoid, filename)
    line = "{} {} {} {} {} {}\n".format(action_dir, num_frames, verb_class, noun_class, uid, start_frame)
    split = split_dicts[0][pid]
    if split == 'train':
        train_files[0].write(line)
    else:
        val_files[0].write(line)

train_files[0].close()
val_files[0].close()
