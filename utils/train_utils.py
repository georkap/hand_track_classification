# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:53:37 2018

Functions that are used in training the network.

1) Cyclic learning rate scheduler from:
    https://github.com/thomasjpfan/pytorch/blob/master/torch/optim/lr_scheduler.py


@author: Γιώργος
"""
import sys
import time
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from utils.calc_utils import AverageMeter, accuracy
from utils.file_utils import print_and_save

class CyclicLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle.
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up.
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_idx (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 optimizer,
                 base_lr=1e-3,
                 max_lr=6e-3,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 last_batch_idx=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        base_lrs = self._format_lr('base_lr', optimizer, base_lr)
        if last_batch_idx == -1:
            for base_lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = base_lr

        self.max_lrs = self._format_lr('max_lr', optimizer, max_lr)

        step_size_down = step_size_down or step_size_up
        self.total_size = float(step_size_up + step_size_down)
        self.step_ratio = float(step_size_up) / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        super(CyclicLR, self).__init__(optimizer, last_batch_idx)

    def _format_lr(self, name, optimizer, lr):
        """Return correctly formatted lr for each param group."""
        if isinstance(lr, (list, tuple)):
            if len(lr) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(lr)))
            return np.array(lr)
        else:
            return lr * np.ones(len(optimizer.param_groups))

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.
        """
        cycle = np.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)
        return lrs

class LRRangeTest(_LRScheduler):
    def __init__(self, optimizer, lr_steps, step_size, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.lr_steps = lr_steps
        self.step_size = step_size
        super(LRRangeTest, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        lrs = []
        lr_ind = self.last_epoch//self.step_size
        lr = self.lr_steps[lr_ind]
        lrs.append(lr)
        return lrs
    
class GroupMultistep(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.epoch = 0
        super(GroupMultistep, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        lr_decay = 0.1 ** sum(self.epoch >= np.array(self.milestones))
        lr = self.base_lrs[0] * lr_decay #self.base_lrs is a list created from the optimizer, in the constructor of _LRScheduler
        return lr
    
    def step(self, epoch=None):
        if epoch is None:
            self.epoch = self.last_epoch + 1
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            param_group['lr'] = self.get_lr() * lr_mult  
    
def load_lr_scheduler(lr_type, lr_steps, optimizer, train_iterator_length):
    lr_scheduler = None
    if lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=int(lr_steps[0]),
                                                       gamma=float(lr_steps[1]))
    elif lr_type == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(x) for x in lr_steps[:-1]],
                                                            gamma=float(lr_steps[-1]))
    elif lr_type == 'clr':
        lr_scheduler = CyclicLR(optimizer, base_lr=float(lr_steps[0]), 
                                max_lr=float(lr_steps[1]), step_size_up=int(lr_steps[2])*train_iterator_length,
                                step_size_down=int(lr_steps[3])*train_iterator_length, mode=str(lr_steps[4]),
                                gamma=float(lr_steps[5]))
    elif lr_type == 'groupmultistep':
        lr_scheduler = GroupMultistep(optimizer,
                                      milestones=[int(x) for x in lr_steps[:-1]],
                                      gamma=float(lr_steps[-1]))
    else:
        sys.exit("Unsupported lr type")
    return lr_scheduler

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda(device=0)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    pred=pred.to(torch.device("cuda:{}".format(0)))
    y_a=y_a.to(torch.device("cuda:{}".format(0)))
    y_b=y_b.to(torch.device("cuda:{}".format(0)))
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_attn_lstm(model, optimizer, criterion, train_iterator, cur_epoch, 
                    log_file, lr_scheduler):
    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    
    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()
    
    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, seq_lengths, targets) in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()
        
        inputs = torch.tensor(inputs, requires_grad=True).cuda()
        targets = torch.tensor(targets).cuda()

        inputs = inputs.transpose(1, 0)
        outputs, attn_weights = model(inputs, seq_lengths)

        loss = 0
        for output in outputs:
            loss += criterion(output, targets)
        loss /= len(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1, t5 = accuracy(outputs[-1].detach().cpu(), targets.cpu(), topk=(1,5))
        top1.update(t1.item(), outputs[-1].size(0))
        top5.update(t5.item(), outputs[-1].size(0))
        losses.update(loss.item(), outputs[-1].size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        print_and_save('[Epoch:{}, Batch {}/{} in {:.3f} s][Loss {:.4f}[avg:{:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]], LR {:.6f}'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg, 
                lr_scheduler.get_lr()[0]), log_file)

def test_attn_lstm(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, seq_lengths, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            inputs = inputs.transpose(1, 0)
            outputs, attn_weights = model(inputs, seq_lengths)
    
            loss = 0
            for output in outputs:
                loss += criterion(output, targets)
            loss /= len(outputs)

            t1, t5 = accuracy(outputs[-1].detach().cpu(), targets.cpu(), topk=(1,5))
            top1.update(t1.item(), outputs[-1].size(0))
            top5.update(t5.item(), outputs[-1].size(0))                
            losses.update(loss.item(), outputs[-1].size(0))

            print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                    cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg

def train_lstm(model, optimizer, criterion, train_iterator, cur_epoch, log_file, lr_scheduler):
    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    
    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()
    
    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, seq_lengths, targets) in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()    
        
        # for multilstm test
#        inputs.squeeze_(0)
#        inputs.transpose_(1,2)
#        inputs.transpose_(0,1)
        
        inputs = torch.tensor(inputs, requires_grad=True).cuda()
        targets = torch.tensor(targets).cuda()

        inputs = inputs.transpose(1, 0)
        output = model(inputs, seq_lengths)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1,5))
        top1.update(t1.item(), output.size(0))
        top5.update(t5.item(), output.size(0))
        losses.update(loss.item(), output.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        print_and_save('[Epoch:{}, Batch {}/{} in {:.3f} s][Loss {:.4f}[avg:{:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]], LR {:.6f}'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg, 
                lr_scheduler.get_lr()[0]), log_file)

def test_lstm(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, seq_lengths, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            inputs = inputs.transpose(1, 0)
            output = model(inputs, seq_lengths)
            
            loss = criterion(output, targets)

            t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1,5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))                
            losses.update(loss.item(), output.size(0))

            print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                    cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg

#def train_cnn(model, optimizer, criterion, train_iterator, mixup_alpha, cur_epoch, log_file, lr_scheduler=None, clip_gradient=False):
def train_cnn(model, optimizer, criterion, train_iterator, mixup_alpha, cur_epoch, log_file, lr_scheduler=None):
    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    
    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()
    
    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()
            
        inputs = torch.tensor(inputs, requires_grad=True).cuda()
        targets = torch.tensor(targets).cuda()

        # TODO: Fix mixup and cuda integration, especially for mfnet
        if mixup_alpha != 1:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
        
        output = model(inputs)

        if mixup_alpha != 1:
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        
#        if clip_gradient is not None:
#            total_norm = torch.nn.clip_grad_norm_(model.parameters(), clip_gradient)
#            if total_norm > clip_gradient:
#                to_print = "clipping gradient: {} with coef {}".format(total_norm, clip_gradient / total_norm)
#                print_and_save(to_print, log_file)
        
        optimizer.step()

        t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1,5))
        top1.update(t1.item(), output.size(0))
        top5.update(t5.item(), output.size(0))
        losses.update(loss.item(), output.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        print_and_save('[Epoch:{}, Batch {}/{} in {:.3f} s][Loss {:.4f}[avg:{:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]], LR {:.6f}'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg, 
                lr_scheduler.get_lr()[0]), log_file)

def test_cnn(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            output = model(inputs)
            loss = criterion(output, targets)

            t1, t5 = accuracy(output.detach().cpu(), targets.detach().cpu(), topk=(1,5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))

            print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                    cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg