# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:30:54 2018

Used for evaluation of a model and outputs a confusion matrix, top1, top5 and per class accuracy metrics

for lstm models

@author: Γιώργος
"""

import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from models.lstm_hands import LSTM_Hands, LSTM_per_hand
from utils.dataset_loader import PointDatasetLoader, PointVectorSummedDatasetLoader
from utils.dataset_loader_utils import lstm_collate
from utils.calc_utils import AverageMeter, accuracy
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
torch.set_printoptions(linewidth=1000000, threshold=1000000)

def validate_lstm(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    outputs = []
    
    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, seq_lengths, targets, video_names) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            inputs = inputs.transpose(1,0)
            output = model(inputs, seq_lengths)            
            loss = criterion(output, targets)
            
            batch_preds = []
            for j in range(output.size(0)):
                res = np.argmax(output[j].detach().cpu().numpy())
                label = targets[j].cpu().numpy()
                outputs.append([res, label])
                batch_preds.append("{}, P-L:{}-{}".format(video_names[j], res, label))
                
                t1, t5 = accuracy(output[j].unsqueeze_(0).detach().cpu(), 
                                  targets[j].unsqueeze_(0).detach().cpu(), topk=(1,5))
                top1.update(t1.item(), 1)
                top5.update(t5.item(), 1)
            losses.update(loss.item(), output.size(0))

            print_and_save('[Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]\n\t{}'.format(
                    batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg, batch_preds), log_file)
        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg, outputs

norm_val = [456., 256., 456., 256.]

def main():
    args = parse_args('lstm', val=True)
    
    verb_classes = args.verb_classes
    
    ckpt_path = args.ckpt_path
    val_list = args.val_list
    
    output_dir = os.path.dirname(ckpt_path)

    if args.logging:
        log_file = os.path.join(output_dir, "results-accuracy-validation.txt")
    else:
        log_file = None
    
    print_and_save(args, log_file)
    
    if args.no_norm_input:
        norm_val = [1., 1., 1., 1.]
    else:
        norm_val = [456., 256., 456., 256.]
        
    if args.lstm_dual:
        lstm_model = LSTM_per_hand
    else:
        lstm_model = LSTM_Hands    
    
    model_ft = lstm_model(args.lstm_input, args.lstm_hidden, args.lstm_layers, verb_classes, args.dropout)
    collate_fn = lstm_collate
    validate = validate_lstm    
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    print_and_save("Model loaded to gpu", log_file)
    cudnn.benchmark = True
    checkpoint = torch.load(ckpt_path)    
    model_ft.load_state_dict(checkpoint['state_dict'])


    if args.lstm_feature == "coords" or args.lstm_feature == "coords_dual":
        if args.lstm_clamped and (not args.lstm_dual or args.lstm_seq_size == 0):
            sys.exit("Clamped tracks require dual lstms and a fixed lstm sequence size.")
        dataset_loader = PointDatasetLoader(val_list, max_seq_length=args.lstm_seq_size,
                                            norm_val=norm_val, dual=args.lstm_dual,
                                            clamp=args.lstm_clamped, validation=True)
    elif args.lstm_feature == "vec_sum" or args.lstm_feature == "vec_sum_dual":
        dataset_loader = PointVectorSummedDatasetLoader(val_list,
                                                        max_seq_length=args.lstm_seq_size,
                                                        dual=args.lstm_dual, validation=True)
    else:
        sys.exit("Unsupported lstm feature")

    dataset_iterator = torch.utils.data.DataLoader(dataset_loader, 
                                                   batch_size=args.batch_size, 
                                                   num_workers=args.num_workers, 
                                                   collate_fn=collate_fn, 
                                                   pin_memory=True)
    
    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    top1, outputs = validate(model_ft, ce_loss, dataset_iterator, checkpoint['epoch'], val_list.split("\\")[-1], log_file)

    #video_pred = [np.argmax(x[0].detach().cpu().numpy()) for x in outputs]
    #video_labels = [x[1].cpu().numpy() for x in outputs]
    video_pred = [x[0] for x in outputs]
    video_labels = [x[1] for x in outputs]

    cf = confusion_matrix(video_labels, video_pred).astype(int)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    with np.errstate(divide='warn'):
        cls_acc = np.nan_to_num(cls_hit / cls_cnt)

    print_and_save(cf, log_file)
    print_and_save(cls_acc, log_file)
    print_and_save('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100), log_file)

if __name__ == '__main__':
    main()