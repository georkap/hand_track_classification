# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:49:11 2019

Used for evaluation of a model and outputs a confusion matrix, top1, top5 and per class accuracy metrics

for lstm polar models

@author: Γιώργος
"""
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.lstm_hands import LSTM_Hands
from utils.dataset_loader import PointPolarDatasetLoader, AnglesDatasetLoader, PointDiffDatasetLoader
from utils.dataset_loader_utils import lstm_collate
from utils.calc_utils import AverageMeter, accuracy, analyze_preds_labels
from utils.argparse_utils import parse_args
from utils.file_utils import print_and_save

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

norm_val = [456., 256., 456., 256.]
def main():
    args = parse_args('lstm_diffs', val=True)
    
    output_dir = os.path.dirname(args.ckpt_path)
    log_file = os.path.join(output_dir, "results-accuracy-validation.txt") if args.logging else None
    print_and_save(args, log_file)
    cudnn.benchmark = True
        
    lstm_model = LSTM_Hands
    kwargs = {'dropout': 0, 'bidir':args.lstm_bidir}    
    model_ft = lstm_model(args.lstm_input, args.lstm_hidden, args.lstm_layers, args.verb_classes, **kwargs)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.ckpt_path)    
    model_ft.load_state_dict(checkpoint['state_dict'])
    print_and_save("Model loaded to gpu", log_file)

    norm_val = [1., 1., 1., 1.] if args.no_norm_input else [456., 256., 456., 256.]
#    dataset_loader = PointPolarDatasetLoader(args.val_list, max_seq_length=args.lstm_seq_size,
#                                             norm_val=norm_val, validation=True)
    dataset_loader = PointDiffDatasetLoader(args.val_list, max_seq_length=args.lstm_seq_size,
                                            norm_val=norm_val, validation=True)

    collate_fn = lstm_collate
#    collate_fn = torch.utils.data.dataloader.default_collate
    dataset_iterator = torch.utils.data.DataLoader(dataset_loader, 
                                                   batch_size=args.batch_size, 
                                                   num_workers=args.num_workers, 
                                                   collate_fn=collate_fn, 
                                                   pin_memory=True)
    
    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    validate = validate_lstm
    top1, outputs = validate(model_ft, ce_loss, dataset_iterator, checkpoint['epoch'], args.val_list.split("\\")[-1], log_file, args)

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