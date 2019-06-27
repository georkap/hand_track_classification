# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:34:58 2018

calc_utils

@author: Γιώργος
"""

import os
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.file_utils import print_and_save

def init_annot_and_splits(annotations_file, brd_splits = False):
    # init annotation file and splits
    annotations = pandas.read_csv(annotations_file)

    # get number of verb and noun classes for the full "train" dataset as discussed in the paper
    verb_classes = len(np.unique(annotations.verb_class.values))
    noun_classes = len(np.unique(annotations.noun_class.values))
    print("Classes in original train dataset, verbs:{}, nouns:{}".format(verb_classes, noun_classes))

    unavailable = [9, 11, 18]
    available_pids = ["P{:02d}".format(i) for i in range(1, 32) if i not in unavailable]

    if brd_splits:
        return annotations, init_splits_brd(available_pids)
    else:
        return annotations, init_splits_custom(available_pids)

def init_splits_brd(available_pids):
    split_1 = {}
    for i in range(len(available_pids)):
        split_1[available_pids[i]] = "train" if i < 26 else "val"
    split_dicts = [split_1]
    return split_dicts

def init_splits_custom(available_pids):
    split_1, split_2, split_3, split_4 = {}, {}, {}, {}
    for i in range(28):
        split_1[available_pids[i]] = "train" if i < 21 else "val"
        split_2[available_pids[i]] = "train" if i < 14 or i > 20 else "val"
        split_3[available_pids[i]] = "train" if i < 7 or i > 13 else "val"
        split_4[available_pids[i]] = "train" if i > 6 else "val"
    split_dicts = [split_1, split_2, split_3, split_4]
    return split_dicts


def get_classes(annotations_file, split_path, brd_splits, num_instances):
    annotations, split_dicts = init_annot_and_splits(annotations_file, brd_splits=brd_splits)
    # export verb and noun classes with more than 100 instances in training
    split_id = int(os.path.basename(split_path).split('.')[0].split('_')[-1]) - 1
    split = split_dicts[split_id]
    train = [k for (k, i) in split.items() if i=='train']
    val = [k for (k, i) in split.items() if i=='val']
    
    verbs_t_un, verbs_t_count = np.unique(annotations.loc[annotations['participant_id'].isin(train)].
                                                          verb_class.values, return_counts=True)
    nouns_t_un, nouns_t_count = np.unique(annotations.loc[annotations['participant_id'].isin(train)].
                                                          noun_class.values, return_counts=True)
    verbs_v_un, verbs_v_count = np.unique(annotations.loc[annotations['participant_id'].isin(val)].
                                                          verb_class.values, return_counts=True)
    nouns_v_un, nouns_v_count = np.unique(annotations.loc[annotations['participant_id'].isin(val)].
                                                          noun_class.values, return_counts=True)
    
    print("Classes in train split: verbs:{}, nouns:{}".format(len(verbs_t_un), len(nouns_t_un)))
    print("Classes in val split: verbs:{}, nouns:{}".format(len(verbs_v_un), len(nouns_v_un)))
    verbs_training = dict(zip(verbs_t_un, verbs_t_count))
    nouns_training = dict(zip(nouns_t_un, nouns_t_count))
#    verbs_val = dict(zip(verbs_v_un, verbs_v_count))
#    nouns_val = dict(zip(nouns_v_un, nouns_v_count))
    verb_ids_sorted = list(reversed(np.argsort(verbs_t_count)))
    noun_ids_sorted = list(reversed(np.argsort(nouns_t_count)))
    
    verbs_t_instances, nouns_t_instances = {}, {}
    for key, item in verbs_training.items():
        if int(item) >= num_instances:
            verbs_t_instances[key] = item
    for key, item in nouns_training.items():
        if int(item) >= num_instances:
            nouns_t_instances[key] = item    
    return list(verbs_t_instances.keys()), verb_ids_sorted, list(nouns_t_instances.keys()), noun_ids_sorted

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    if maxk > output.shape[1]:
        maxk = output.shape[1]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def rec_prec_per_class(confusion_matrix):
    # cm is inversed from the wikipedia example on 3/8/18

    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP
    
    with np.errstate(divide='warn'):
        precision = np.nan_to_num(TP/(TP+FP))
        recall = np.nan_to_num(TP/(TP+FN))
    
    return np.around(100*recall, 2), np.around(100*precision, 2)

def analyze_preds_labels(preds, labels, all_class_indices):
    cf = confusion_matrix(labels, preds, all_class_indices).astype(int)
    recall, precision = rec_prec_per_class(cf)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    with np.errstate(divide='warn'):
        cls_acc = np.around(100*np.nan_to_num(cls_hit / cls_cnt), 2)
    mean_cls_acc = np.mean(cls_acc)
    top1_acc = np.around(100*(np.sum(cls_hit)/np.sum(cf)), 3)
    
    return cf, recall, precision, cls_acc, mean_cls_acc, top1_acc

def avg_rec_prec_trimmed(pred, labels, valid_class_indices, all_class_indices):
    cm = confusion_matrix(labels, pred, all_class_indices).astype(float)
    
    cm_trimmed_cols = cm[:, valid_class_indices]
    cm_trimmed_rows = cm_trimmed_cols[valid_class_indices, :]
    
    recall, precision = rec_prec_per_class(cm_trimmed_rows)
    
    return np.sum(precision)/len(precision), np.sum(recall)/len(recall), cm_trimmed_rows.astype(int)

def eval_final_print_mt(video_preds, video_labels, task_id, current_classes, log_file, annotations_path=None,
                        val_list=None, task_type = 'None'):
    cf, recall, precision, cls_acc, mean_cls_acc, top1_acc = analyze_preds_labels(video_preds, video_labels,
                                                                                  all_class_indices=list(range(int(current_classes))))
    print_and_save("Task {}".format(task_id), log_file)
    print_and_save(cf, log_file)

    if annotations_path:
        brd_splits = '_brd' in val_list
        valid_verb_indices, verb_ids_sorted, valid_noun_indices, noun_ids_sorted = get_classes(annotations_path,
                                                                                               val_list, brd_splits, 100)
        if task_type == 'EpicVerbs': # 'Verbs': error prone if I ever train nouns on their own
            valid_indices, ids_sorted = valid_verb_indices, verb_ids_sorted
            all_indices = list(range(int(125))) # manually set verb classes to avoid loading the verb names file that loads 125...
        elif task_type == 'EpicNouns':
            valid_indices, ids_sorted = valid_noun_indices, noun_ids_sorted
            all_indices = list(range(int(352)))
        ave_pre, ave_rec, _ = avg_rec_prec_trimmed(video_preds, video_labels, valid_indices, all_indices)
        print_and_save("{} > 100 instances at training:".format(task_type), log_file)
        print_and_save("Classes are {}".format(valid_indices), log_file)
        print_and_save("average precision {0:02f}%, average recall {1:02f}%".format(ave_pre, ave_rec), log_file)
        print_and_save("Most common {} in training".format(task_type), log_file)
        print_and_save("15 {} rec {}".format(task_type, recall[ids_sorted[:15]]), log_file)
        print_and_save("15 {} pre {}".format(task_type, precision[ids_sorted[:15]]), log_file)

    print_and_save("Cls Rec {}".format(recall), log_file)
    print_and_save("Cls Pre {}".format(precision), log_file)
    print_and_save("Cls Acc {}".format(cls_acc), log_file)
    print_and_save("Mean Cls Acc {:.02f}%".format(mean_cls_acc), log_file)
    print_and_save("Dataset Acc {}".format(top1_acc), log_file)
    return mean_cls_acc, top1_acc

def eval_final_print(video_preds, video_labels, cls_type, annotations_path, val_list, max_classes, log_file):
    cf, recall, precision, cls_acc, mean_cls_acc, top1_acc = analyze_preds_labels(video_preds, video_labels,
                                                                                  all_class_indices=list(range(int(max_classes))))
    print_and_save(cls_type, log_file)
    print_and_save(cf, log_file)
    
    if annotations_path:
        brd_splits = '_brd' in val_list
        valid_verb_indices, verb_ids_sorted, valid_noun_indices, noun_ids_sorted = get_classes(annotations_path,
                                                                                               val_list, brd_splits, 100)
        if cls_type == 'Verbs':
            valid_indices, ids_sorted = valid_verb_indices, verb_ids_sorted  
            all_indices = list(range(int(125))) # manually set verb classes to avoid loading the verb names file that loads 125...
        else:
            valid_indices, ids_sorted = valid_noun_indices, noun_ids_sorted
            all_indices = list(range(int(352)))
        
        ave_pre, ave_rec, _ = avg_rec_prec_trimmed(video_preds, video_labels, valid_indices, all_indices)
        print_and_save("{} > 100 instances at training:".format(cls_type), log_file)
        print_and_save("Classes are {}".format(valid_indices), log_file)
        print_and_save("average precision {0:02f}%, average recall {1:02f}%".format(ave_pre, ave_rec), log_file)
        print_and_save("Most common {} in training".format(cls_type), log_file)
        print_and_save("15 {} rec {}".format(cls_type, recall[ids_sorted[:15]]), log_file)
        print_and_save("15 {} pre {}".format(cls_type, precision[ids_sorted[:15]]), log_file)
    
    print_and_save("Cls Rec {}".format(recall), log_file)
    print_and_save("Cls Pre {}".format(precision), log_file)
    print_and_save("Cls Acc {}".format(cls_acc), log_file)
    print_and_save("Mean Cls Acc {:.02f}%".format(mean_cls_acc), log_file)
    print_and_save("Dataset Acc {}".format(top1_acc), log_file)
    return mean_cls_acc, top1_acc