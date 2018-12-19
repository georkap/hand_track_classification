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

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from models.lstm_hands import LSTM_Hands, LSTM_per_hand, LSTM_Hands_attn
from utils.dataset_loader import PointDatasetLoader, PointVectorSummedDatasetLoader
from utils.dataset_loader_utils import lstm_collate
from utils.calc_utils import AverageMeter, accuracy, analyze_preds_labels
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save

from matplotlib import pyplot as plt

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
torch.set_printoptions(linewidth=1000000, threshold=1000000)

def validate_lstm(model, criterion, test_iterator, cur_epoch, dataset, log_file, args):
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

            t1, t5 = accuracy(output.detach().cpu(), 
                              targets.detach().cpu(), topk=(1,5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))

            print_and_save('[Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]\n\t{}'.format(
                    batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg, batch_preds), log_file)
        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg, outputs

def validate_lstm_attn(model, criterion, test_iterator, cur_epoch, dataset, log_file, args):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    predictions = []
    all_predictions = torch.zeros((0, args.lstm_seq_size, 1))
    all_attentions = torch.zeros((0, args.lstm_seq_size, args.lstm_seq_size))
    all_targets = torch.zeros((0, 1))
    all_video_names = []
    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, seq_lengths, targets, video_names) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            inputs = inputs.transpose(1,0)
            outputs, attn_weights = model(inputs, seq_lengths)         
            
            loss = 0
            for output in outputs:
                loss += criterion(output, targets)
            loss /= len(outputs)
            
            outputs = torch.stack(outputs)
            outputs = torch.argmax(outputs,dim=2).detach().cpu()
            all_predictions = torch.cat((all_predictions, torch.transpose(outputs, 0, 1).float()), dim=0)
            outputs = outputs.numpy()
            outputs = [np.bincount(outputs[:,kk]).argmax() for kk in range(len(outputs[0]))]
            attn_weights = torch.stack(attn_weights)
            attn_weights = torch.transpose(attn_weights, 0, 1).detach().cpu()
            all_attentions = torch.cat((all_attentions, attn_weights), dim=0)
            all_targets = torch.cat((all_targets, targets.detach().cpu().float()), dim=0)
            all_video_names = all_video_names + video_names
            
            batch_preds = []
            for j in range(len(outputs)):
                res = outputs[j]
                label = targets[j].cpu().numpy()
                predictions.append([res, label])
                batch_preds.append("{}, P-L:{}-{}".format(video_names[j], res, label))

            t1, t5 = accuracy(output.detach().cpu(), 
                              targets.detach().cpu(), topk=(1,5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))

            print_and_save('[Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]\n\t{}'.format(
                    batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg, batch_preds), log_file)
        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
        
        if args.save_attentions:
            all_predictions = all_predictions.numpy().astype(np.int)
            all_targets = all_targets.numpy().astype(np.int)
            output_dir = os.path.join(os.path.dirname(log_file), "figures")
            os.makedirs(output_dir, exist_ok=True)
            for i in range(len(all_targets)):
                name_parts = all_video_names[i].split("\\")[-2:]
                output_file = os.path.join(output_dir, "{}_{}.png".format(name_parts[0], name_parts[1].split('.')[0]))
                showAttention(args.lstm_seq_size, all_predictions[i], all_targets[i], all_attentions[i], output_file)
     
    return top1.avg, predictions

import matplotlib.ticker as ticker
def showAttention(sequence_size, predictions, target, attentions, output_file=None):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_title("class of sequence {}".format(target))
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(list(range(sequence_size+1)))
    ax.set_yticklabels(list(predictions))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.tight_layout()
    if output_file == None:
        plt.show()
    else:
        fig.savefig(output_file)
    plt.close()
norm_val = [456., 256., 456., 256.]
def main():
    args = parse_args('lstm', val=True)
    
    output_dir = os.path.dirname(args.ckpt_path)
    log_file = os.path.join(output_dir, "results-accuracy-validation.txt") if args.logging else None
    print_and_save(args, log_file)
    cudnn.benchmark = True
        
    lstm_model = LSTM_per_hand if args.lstm_dual else LSTM_Hands_attn if args.lstm_attn else LSTM_Hands
    kwargs = {'dropout':args.dropout, 'max_seq_len':args.lstm_seq_size}    
    model_ft = lstm_model(args.lstm_input, args.lstm_hidden, args.lstm_layers, args.verb_classes, **kwargs)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.ckpt_path)    
    model_ft.load_state_dict(checkpoint['state_dict'])
    print_and_save("Model loaded to gpu", log_file)

    norm_val = [1., 1., 1., 1.] if args.no_norm_input else [456., 256., 456., 256.]
    if args.lstm_feature == "coords" or args.lstm_feature == "coords_dual":
        if args.lstm_clamped and (not args.lstm_dual or args.lstm_seq_size == 0):
            sys.exit("Clamped tracks require dual lstms and a fixed lstm sequence size.")
        dataset_loader = PointDatasetLoader(args.val_list, max_seq_length=args.lstm_seq_size,
                                            num_classes=args.verb_classes, norm_val=norm_val,
                                            dual=args.lstm_dual, clamp=args.lstm_clamped,
                                            validation=True)
    elif args.lstm_feature == "vec_sum" or args.lstm_feature == "vec_sum_dual":
        dataset_loader = PointVectorSummedDatasetLoader(args.val_list,
                                                        max_seq_length=args.lstm_seq_size,
                                                        num_classes=args.verb_classes,
                                                        dual=args.lstm_dual, 
                                                        validation=True)
    else:
        sys.exit("Unsupported lstm feature")

    collate_fn = lstm_collate
#    collate_fn = torch.utils.data.dataloader.default_collate
    dataset_iterator = torch.utils.data.DataLoader(dataset_loader, 
                                                   batch_size=args.batch_size, 
                                                   num_workers=args.num_workers, 
                                                   collate_fn=collate_fn, 
                                                   pin_memory=True)
    
    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    validate = validate_lstm_attn if args.lstm_attn else validate_lstm
    top1, outputs = validate(model_ft, ce_loss, dataset_iterator, checkpoint['epoch'], args.val_list.split("\\")[-1], log_file, args)

    #video_pred = [np.argmax(x[0].detach().cpu().numpy()) for x in outputs]
    #video_labels = [x[1].cpu().numpy() for x in outputs]
    video_preds = [x[0] for x in outputs]
    video_labels = [x[1] for x in outputs]

    cf, recall, precision, cls_acc, mean_cls_acc, top1_acc = analyze_preds_labels(video_preds, video_labels)

    print_and_save(cf, log_file)
    print_and_save("Cls Rec {}".format(recall), log_file)
    print_and_save("Cls Pre {}".format(precision), log_file)
    print_and_save("Cls Acc {}".format(cls_acc), log_file)
    print_and_save("Mean Cls Acc {:.02f}%".format(mean_cls_acc), log_file)
    print_and_save("Dataset Acc {}".format(top1_acc), log_file)

if __name__ == '__main__':
    main()