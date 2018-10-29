# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:57:09 2018

training on the hand images using resnet

@author: Γιώργος
"""

import os
import time
import shutil
import torch
from models.resnet18_zoo import resnet101loader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils.dataset_loader import DatasetLoader, RandomHorizontalFlip, Resize
from utils.calc_utils import AverageMeter, accuracy

def train(model, optimizer, criterion, device, train_iterator, cur_epoch):
    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.train()

    print('Beginning of epoch: {}'.format(cur_epoch))
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        inputs = torch.tensor(inputs, requires_grad=True).cuda()
        targets = torch.tensor(targets).cuda()

        output = model(inputs)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1,5))
        top1.update(t1.item(), inputs.size(0))
        top5.update(t5.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        print('[Epoch:{}, Batch {}/{} in {:.3f} s][Loss {:.4f}[avg:{:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg))

def test(model, device, test_iterator, cur_epoch):
    top1, top5 = AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print('Evaluating after epoch: {}'.format(cur_epoch))
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets)

            output = model(inputs)

            t1, t5 = accuracy(output.detach().cpu(), targets, topk=(1,5))
            top1.update(t1.item(), inputs.size(0))
            top5.update(t5.item(), inputs.size(0))

            print('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                    cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg))

        print('Test Results: Top1 {:.3f} Top5 {:.3f}'.format(top1.avg, top5.avg))
    return top1.avg

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    verb_classes = 120
    max_epochs = 200
    eval_freq = 1
    save_freq = 5
    pretrained = True
    feature_extraction = False
    num_workers = 8
    batch_size = 64
    output_dir = 'outputs'
    model_name = 'resnet101_ft_200ep_64bs'
    output_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   

    train_list = r'splits\hand_track_train_1.txt'
    test_list = r'splits\hand_track_val_1.txt'

    model_ft = resnet101loader(verb_classes, pretrained, feature_extraction)
    model_ft = torch.nn.DataParallel(model_ft).cuda()

    cudnn.benchmark = True

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extraction:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([Resize((224,224)), RandomHorizontalFlip(),
                                           transforms.ToTensor(), normalize])
    train_loader = DatasetLoader(train_list, train_transforms)
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    test_transforms = transforms.Compose([Resize((224,224)),
                                          transforms.ToTensor(), normalize])
    test_loader = DatasetLoader(test_list, test_transforms)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    new_top1, top1 = 0.0, 0.0
    isbest = False
    for epoch in range(max_epochs):
        lr_scheduler.step()
        train(model_ft, optimizer, ce_loss, device, train_iterator, epoch)
        if (epoch+1) % eval_freq == 0:
            new_top1 = test(model_ft, device, test_iterator, epoch)
            isbest = new_top1 >= top1
        if (epoch+1) % save_freq == 0:
            weight_file = os.path.join(output_dir, model_name + '_{:03d}.pth'.format(epoch))
            print('Saving weights to {}'.format(weight_file))
            torch.save({'epoch': epoch + 1,
                        'state_dict': model_ft.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'top1': new_top1}, weight_file)
            if isbest:
                best = os.path.join(output_dir, model_name+'_best.pth')
                shutil.copyfile(weight_file, best)

if __name__=='__main__':
    main()