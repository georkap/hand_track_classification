# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:24:00 2018

main_eval.py 

Used for evaluation of a model and outputs a confusion matrix, top1, top5 and per class accuracy metrics

@author: Γιώργος
"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from models.lstm_hands import LSTM_Hands
from utils.dataset_loader import PointDatasetLoader
from utils.dataset_loader_utils import lstm_collate
from utils.calc_utils import AverageMeter, accuracy
from utils.argparse_utils import parse_args_val
from utils.file_utils import print_and_save


np.set_printoptions(linewidth=np.inf, threshold=np.inf)

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
                
                t1, t5 = accuracy(output[j].detach().cpu(), targets[j].detach().cpu(), topk=(1,5))
                top1.update(t1.item(), 1)
                top5.update(t5.item(), 1)
                losses.update(loss.item()/inputs.size(0), 1) # approximate the loss over the batch

            print_and_save('[Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]\n\t{}'.format(
                    batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg, batch_preds), log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg, outputs

norm_val = [456., 256., 456., 256.]
def main():
    args = parse_args_val()
    verb_classes = 120
    
    ckpt_path = args.ckpt_path
    val_list = args.val_list
    output_dir = os.path.dirname(ckpt_path)

    if args.logging:
        log_file = os.path.join(output_dir, "results-accuracy-validation.txt")
    else:
        log_file = None
    
    print_and_save(args, log_file)

    model_ft = LSTM_Hands(4, args.lstm_hidden, args.lstm_layers, verb_classes)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    print_and_save("Model loaded to gpu", log_file)
    cudnn.benchmark = True

    checkpoint = torch.load(ckpt_path)

    # below line is needed if network is trained with DataParallel
#    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
#    model_ft.load_state_dict(base_dict) 
    model_ft.load_state_dict(checkpoint['state_dict'])

    dataset_loader = PointDatasetLoader(val_list, norm_val=norm_val, validation=True)
    dataset_iterator = torch.utils.data.DataLoader(dataset_loader, 
                                                   batch_size=args.batch_size, 
                                                   num_workers=args.num_workers, 
                                                   collate_fn=lstm_collate, 
                                                   pin_memory=True)

    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    top1, outputs = validate_lstm(model_ft, ce_loss, dataset_iterator, checkpoint['epoch'], val_list.split("\\")[-1], log_file)

    #video_pred = [np.argmax(x[0].detach().cpu().numpy()) for x in outputs]
    #video_labels = [x[1].cpu().numpy() for x in outputs]
    video_pred = [x[0] for x in outputs]
    video_labels = [x[1] for x in outputs]

    cf = confusion_matrix(video_labels, video_pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    with np.errstate(divide='warn'):
        cls_acc = np.nan_to_num(cls_hit / cls_cnt)

    print_and_save(cf, log_file)
    print_and_save(cls_acc, log_file)
    print_and_save('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100), log_file)

if __name__ == '__main__':
    main()