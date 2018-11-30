# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:34:58 2018

calc_utils

@author: Γιώργος
"""

import numpy as np
from sklearn.metrics import confusion_matrix

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

def analyze_preds_labels(preds, labels):
    cf = confusion_matrix(labels, preds).astype(int)
    recall, precision = rec_prec_per_class(cf)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    with np.errstate(divide='warn'):
        cls_acc = np.around(100*np.nan_to_num(cls_hit / cls_cnt), 2)
    mean_cls_acc = np.mean(cls_acc)
    top1_acc = np.around(100*(np.sum(cls_hit)/np.sum(cf)), 3)
    
    return cf, recall, precision, cls_acc, mean_cls_acc, top1_acc