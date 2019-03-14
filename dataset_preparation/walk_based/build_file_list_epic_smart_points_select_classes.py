# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 09:18:01 2018

Make lists for hand activity classification based on the epic kitchens splits for track points

only select classes for debugging purposes

@author: Γιώργος
"""

import os

get_track_class = lambda x: int(x.split('_')[1])

#selected_classes = [5,6]
selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 32]

unavailable = [9, 11, 18]
available_pids = ["P{:02d}".format(i) for i in range(1,32) if i not in unavailable]

split_1, split_2, split_3, split_4 = {}, {}, {}, {}
for i in range(28):
    split_1[available_pids[i]] = "train" if i < 21 else "val"
    split_2[available_pids[i]] = "train" if i < 14 or i > 20 else "val"
    split_3[available_pids[i]] = "train" if i < 7 or i > 13 else "val"
    split_4[available_pids[i]] = "train" if i > 6 else "val"
split_dicts = [split_1, split_2, split_3, split_4]

BASE_DIR = r"..\hand_detection_tracks"
SPLITS_DIR = r"..\splits\hand_tracks_select24"
os.makedirs(SPLITS_DIR, exist_ok=True)

train_names = [os.path.join(SPLITS_DIR, "hand_locs_train_{}.txt".format(i)) for i in range(1,5)]
val_names = [os.path.join(SPLITS_DIR, "hand_locs_val_{}.txt".format(i)) for i in range(1,5)]

train_files = []
val_files = []
for i in range(4):
    train_files.append(open(train_names[i], 'a'))
    val_files.append(open(val_names[i], 'a'))

for base_dir, sub_dirs, files in os.walk(BASE_DIR):
    if len(files) == 0: continue
    path_split = base_dir.split('\\')
    pid = path_split[-2]
    for file in files:
        verb_class = get_track_class(file)
        if verb_class not in selected_classes:
            continue
        file_split = file.split('.')[0].split('_')
        im_path = os.path.join(path_split[-3],path_split[-2],path_split[-1],file)
        line = "{} {} {} {}\n".format(im_path, file_split[0], file_split[1], file_split[2])
        for i in range(4): # in range(num_splits)
            split = split_dicts[i][pid]
            if split == 'train':
                train_files[i].write(line)
            else:
                val_files[i].write(line)


for i in range(4):
    train_files[i].close()
    val_files[i].close()