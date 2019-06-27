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
import dsntnn
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from utils.calc_utils import AverageMeter, accuracy
from utils.file_utils import print_and_save

class CustomLRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            param_group['lr'] = lr * lr_mult # Added lr mult to facilitate finetuning in mfnet

class CyclicLR(CustomLRScheduler):
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
        return [lr]
    
    def step(self, epoch=None):
        if epoch is None:
            self.epoch = self.last_epoch + 1
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            param_group['lr'] = self.get_lr()[0] * lr_mult  
    
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

def train_lstm_do(model, optimizer, criterion, train_iterator, cur_epoch, log_file, lr_scheduler):
    batch_time, losses_a, losses_b, losses, top1_a, top5_a, top1_b, top5_b = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
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
        inputs = inputs.transpose(1, 0)
        output_a, output_b = model(inputs, seq_lengths)
        
        targets_a = torch.tensor(targets[0]).cuda()
        targets_b = torch.tensor(targets[1]).cuda()        
        loss_a = criterion(output_a, targets_a)
        loss_b = criterion(output_b, targets_b)
        loss = 0.75*loss_a + 0.25*loss_b
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t1_a, t5_a = accuracy(output_a.detach().cpu(), targets_a.detach().cpu(), topk=(1,5))
        t1_b, t5_b = accuracy(output_b.detach().cpu(), targets_b.detach().cpu(), topk=(1,5))
        top1_a.update(t1_a.item(), output_a.size(0))
        top5_a.update(t5_a.item(), output_a.size(0))
        top1_b.update(t1_b.item(), output_b.size(0))
        top5_b.update(t5_b.item(), output_b.size(0))
        losses_a.update(loss_a.item(), output_a.size(0))
        losses_b.update(loss_b.item(), output_b.size(0))
        losses.update(loss.item(), output_a.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        to_print = '[Epoch:{}, Batch {}/{} in {:.3f} s]'\
                   '[Losses {:.4f}[avg:{:.4f}], loss_a {:.4f}[avg:{:.4f}], loss_b {:.4f}[avg:{:.4f}],' \
                   'Top1_a {:.3f}[avg:{:.3f}], Top5_a {:.3f}[avg:{:.3f}],' \
                   'Top1_b {:.3f}[avg:{:.3f}], Top5_b {:.3f}[avg:{:.3f}]],' \
                   'LR {:.6f}'.format(
                           cur_epoch, batch_idx, len(train_iterator), batch_time.val,
                           losses_a.val, losses_a.avg, losses_b.val, losses_b.avg, losses.val, losses.avg,
                           top1_a.val, top1_a.avg, top5_a.val, top5_a.avg,
                           top1_b.val, top1_b.avg, top5_b.val, top5_b.avg,
                           lr_scheduler.get_lr()[0])
        print_and_save(to_print, log_file)

def test_lstm_do(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, losses_a, losses_b, top1_a, top5_a, top1_b, top5_b = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, seq_lengths, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            inputs = inputs.transpose(1, 0)

            output_a, output_b = model(inputs, seq_lengths)
            
            targets_a = torch.tensor(targets[0]).cuda()
            targets_b = torch.tensor(targets[1]).cuda()
            loss_a = criterion(output_a, targets_a)
            loss_b = criterion(output_b, targets_b)
            loss = 0.75*loss_a + 0.25*loss_b
            
            t1_a, t5_a = accuracy(output_a.detach().cpu(), targets_a.detach().cpu(), topk=(1,5))
            t1_b, t5_b = accuracy(output_b.detach().cpu(), targets_b.detach().cpu(), topk=(1,5))
            top1_a.update(t1_a.item(), output_a.size(0))
            top5_a.update(t5_a.item(), output_a.size(0))
            top1_b.update(t1_b.item(), output_b.size(0))
            top5_b.update(t5_b.item(), output_b.size(0))
            losses_a.update(loss_a.item(), output_a.size(0))
            losses_b.update(loss_b.item(), output_b.size(0))
            losses.update(loss.item(), output_a.size(0))
            
            to_print = '[Epoch:{}, Batch {}/{}]' \
                       '[Top1_a {:.3f}[avg:{:.3f}], Top5_a {:.3f}[avg:{:.3f}],' \
                       'Top1_b {:.3f}[avg:{:.3f}], Top5_b {:.3f}[avg:{:.3f}]]'.format(
                       cur_epoch, batch_idx, len(test_iterator),
                       top1_a.val, top1_a.avg, top5_a.val, top5_a.avg,
                       top1_b.val, top1_b.avg, top5_b.val, top5_b.avg)
            print_and_save(to_print, log_file)

        print_and_save('{} Results: Loss {:.3f}, Top1_a {:.3f}, Top5_a {:.3f}, Top1_b {:.3f}, Top5_b {:.3f}'.format(dataset, losses.avg, top1_a.avg, top5_a.avg, top1_b.avg, top5_b.avg), log_file)
    return top1_a.avg, top1_b.avg
        

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


def calc_coord_loss(coords, heatmaps, target_var):
    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(coords, target_var)  # shape:[B, D, L, 2] batch, depth, locations, feature
    # Per-location regularization losses

    reg_losses = []
    for i in range(heatmaps.shape[1]):
        hms = heatmaps[:, i]
        target = target_var[:, i]
        reg_loss = dsntnn.js_reg_losses(hms, target, sigma_t=1.0)
        reg_losses.append(reg_loss)
    reg_losses = torch.stack(reg_losses, 1)
    # reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0) # shape: [B, D, L, 7, 7]
    # Combine losses into an overall loss
    coord_loss = dsntnn.average_loss(euc_losses + reg_losses)
    return coord_loss


def train_mfnet_mo(model, optimizer, criterion, train_iterator, num_outputs, use_gaze, use_hands, cur_epoch, log_file,
                 gpus, lr_scheduler=None):
    batch_time = AverageMeter()
    loss_meters = [AverageMeter() for _ in range(num_outputs)]
    losses = AverageMeter()
    loss_hands, loss_gaze = AverageMeter(), AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_outputs)]

    model.train()

    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()

    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()

        inputs = inputs.cuda(gpus[0])
        outputs, coords, heatmaps = model(inputs)
        targets = targets.cuda(gpus[0]).transpose(0, 1)  # needs transpose to get the first dim to be the task and the second dim to be the batch

        if use_gaze or use_hands:
            cls_targets = targets[:num_outputs, :].long()
        else:
            cls_targets = targets
        assert len(cls_targets) == num_outputs

        losses_per_task = []
        for output, target in zip(outputs, cls_targets):
            loss_for_task = criterion(output, target)
            losses_per_task.append(loss_for_task)

        loss = sum(losses_per_task)

        gaze_coord_loss, hand_coord_loss = 0, 0
        if use_gaze:  # need some debugging for the gaze targets
            gaze_targets = targets[num_outputs:num_outputs + 16, :].transpose(1,0).reshape(-1, 8, 1, 2)
            # for a single shared layer representation of the two signals
            # for gaze slice the first element
            gaze_coords = coords[:, :, 0, :]
            gaze_coords.unsqueeze_(2) # unsqueeze to add the extra dimension for consistency
            gaze_heatmaps = heatmaps[:, :, 0, :]
            gaze_heatmaps.unsqueeze_(2)
            gaze_coord_loss = calc_coord_loss(gaze_coords, gaze_heatmaps, gaze_targets)
            loss = loss + gaze_coord_loss
        if use_hands:
            hand_targets = targets[-32:, :].transpose(1,0).reshape(-1, 8, 2, 2)
            # for hands slice the last two elements, first is left, second is right hand
            hand_coords = coords[:, :, -2:, :]
            hand_heatmaps = heatmaps[:, :, -2:, :]
            hand_coord_loss = calc_coord_loss(hand_coords, hand_heatmaps, hand_targets)
            loss = loss + hand_coord_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metrics
        batch_size = outputs[0].size(0)
        losses.update(loss.item(), batch_size)

        for ind in range(num_outputs):
            t1, t5 = accuracy(outputs[ind].detach().cpu(), cls_targets[ind].detach().cpu(), topk=(1, 5))
            top1_meters[ind].update(t1.item(), batch_size)
            top5_meters[ind].update(t5.item(), batch_size)
            loss_meters[ind].update(losses_per_task[ind].item(), batch_size)

        if use_gaze:
            loss_gaze.update(gaze_coord_loss.item(), batch_size)
        if use_hands:
            loss_hands.update(hand_coord_loss.item(), batch_size)

        batch_time.update(time.time() - t0)
        t0 = time.time()
        to_print = '[Epoch:{}, Batch {}/{} in {:.3f} s]'.format(cur_epoch, batch_idx, len(train_iterator),
                                                                batch_time.val)
        to_print += '[Losses {:.4f}[avg:{:.4f}], '.format(losses.val, losses.avg)
        if use_gaze:
            to_print += '[l_gcoo {:.4f}[avg:{:.4f}], '.format(loss_gaze.val, loss_gaze.avg)
        if use_hands:
            to_print += '[l_hcoo {:.4f}[avg:{:.4f}], '.format(loss_hands.val, loss_hands.avg)
        for ind in range(num_outputs):
            to_print += 'T{}::loss {:.4f}[avg:{:.4f}], '.format(ind, loss_meters[ind].val, loss_meters[ind].avg)
        for ind in range(num_outputs):
            to_print += 'T{}::Top1 {:.3f}[avg:{:.3f}],Top5 {:.3f}[avg:{:.3f}],'.format(
                ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)
        to_print += 'LR {:.6f}'.format(lr_scheduler.get_lr()[0])
        print_and_save(to_print, log_file)
    print_and_save("Epoch train time: {}".format(batch_time.sum), log_file)


def test_mfnet_mo(model, criterion, test_iterator, num_outputs, use_gaze, use_hands, cur_epoch, dataset, log_file, gpus):
    loss_meters = [AverageMeter() for _ in range(num_outputs)]
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_outputs)]
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = inputs.cuda(gpus[0])
            outputs, coords, heatmaps = model(inputs)
            targets = targets.cuda(gpus[0]).transpose(0, 1)

            if use_gaze or use_hands:
                cls_targets = targets[:num_outputs, :].long()
            else:
                cls_targets = targets
            assert len(cls_targets) == num_outputs

            losses_per_task = []
            for output, target in zip(outputs, cls_targets):
                loss_for_task = criterion(output, target)
                losses_per_task.append(loss_for_task)

            loss = sum(losses_per_task)

            if use_gaze:  # need some debugging for the gaze targets
                gaze_targets = targets[num_outputs:num_outputs + 16, :].transpose(1, 0).reshape(-1, 8, 1, 2)
                # for a single shared layer representation of the two signals
                # for gaze slice the first element
                gaze_coords = coords[:, :, 0, :]
                gaze_coords.unsqueeze_(2)  # unsqueeze to add the extra dimension for consistency
                gaze_heatmaps = heatmaps[:, :, 0, :]
                gaze_heatmaps.unsqueeze_(2)
                gaze_coord_loss = calc_coord_loss(gaze_coords, gaze_heatmaps, gaze_targets)
                loss = loss + gaze_coord_loss
            if use_hands:
                hand_targets = targets[-32:, :].transpose(1,0).reshape(-1, 8, 2, 2)
                # for hands slice the last two elements, first is left, second is right hand
                hand_coords = coords[:, :, -2:, :]
                hand_heatmaps = heatmaps[:, :, -2:, :]
                hand_coord_loss = calc_coord_loss(hand_coords, hand_heatmaps, hand_targets)
                loss = loss + hand_coord_loss

            # update metrics
            batch_size = outputs[0].size(0)
            losses.update(loss.item(), batch_size)

            for ind in range(num_outputs):
                t1, t5 = accuracy(outputs[ind].detach().cpu(), cls_targets[ind].detach().cpu(), topk=(1, 5))
                top1_meters[ind].update(t1.item(), batch_size)
                top5_meters[ind].update(t5.item(), batch_size)
                loss_meters[ind].update(losses_per_task[ind].item(), batch_size)

            to_print = '[Epoch:{}, Batch {}/{}] '.format(cur_epoch, batch_idx, len(test_iterator))
            for ind in range(num_outputs):
                to_print += 'T{}::Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}],'.format(
                    ind, top1_meters[ind].val, top1_meters[ind].avg, top5_meters[ind].val, top5_meters[ind].avg)

            print_and_save(to_print, log_file)

        final_print = '{} Results: Loss {:.3f},'.format(dataset, losses.avg)
        for ind in range(num_outputs):
            final_print += 'T{}::Top1 {:.3f}, Top5 {:.3f},'.format(ind, top1_meters[ind].avg, top5_meters[ind].avg)
        print_and_save(final_print, log_file)
    return [tasktop1.avg for tasktop1 in top1_meters]

import math

def unnorm_gaze_coords(_coords):  # expecting values in [-1, 1]
    return ((_coords + 1) * 224 - 1) / 2

def calc_aae(pred, gt, avg='no'):
    # input should be [2] with modalities=1
    d = 112/math.tan(math.pi/6)
    pred = pred - 112
    gt = gt - 112
    r1 = np.array([pred[0], pred[1], d])  # x, y are inverted in numpy but it doesn't change results
    r2 = np.array([gt[0], gt[1], d])
    # angles needs to be of dimension batch*temporal*modalities*1
    angles = math.atan2(np.linalg.norm(np.cross(r1, r2)), np.dot(r1, r2))

    # angles_deg = math.degrees(angles)
    angles_deg = np.rad2deg(angles)
    return angles_deg
    # aae = None
    # if avg == 'no':  # use each coordinate pair for error calculation
    #     aae = [deg for deg in angles_deg.flatten()]
    # elif avg == 'temporal':  # average the angles for one video segment
    #     angles_deg = np.mean(angles_deg, 1)
    #     aae = [deg for deg in angles_deg.flatten()]
    #
    # return aae

from scipy import ndimage
def calc_auc(pred, gt):
    z = np.zeros((224, 224))
    z[int(pred[0])][int(pred[1])] = 1
    z = ndimage.filters.gaussian_filter(z, 14)
    z = z - np.min(z)
    z = z / np.max(z)
    atgt = z[int(gt[0])][int(gt[1])]  # z[i][j]
    fpbool = z > atgt
    auc = (1 - float(fpbool.sum()) / (224 * 224))
    return auc


def validate_mfnet_mo_gaze(model, criterion, test_iterator, num_outputs, use_gaze, use_hands, cur_epoch, dataset, log_file):
    auc_frame, auc_temporal = AverageMeter(), AverageMeter()
    aae_frame, aae_temporal = AverageMeter(), AverageMeter()
    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)

    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, video_names) in enumerate(test_iterator):
            inputs = inputs.cuda()
            outputs, coords, heatmaps = model(inputs)
            targets = targets.cuda().transpose(0, 1)

            if use_gaze or use_hands:
                cls_targets = targets[:num_outputs, :].long()
            else:
                cls_targets = targets
            assert len(cls_targets) == num_outputs

            gaze_targets = targets[num_outputs:num_outputs + 16, :].transpose(1, 0).reshape(-1, 8, 1, 2)
            gaze_targets.squeeze_(2)
            gaze_targets = unnorm_gaze_coords(gaze_targets).cpu().numpy()

            gaze_coords = coords[:, :, 0, :]
            gaze_coords = unnorm_gaze_coords(gaze_coords).cpu().numpy()

            batch_size, temporal_size, _ = gaze_targets.shape
            for b in range(batch_size):
                angles_temp = []
                auc_temp = []
                for t in range(temporal_size):
                    angle_deg = calc_aae(gaze_coords[b,t], gaze_targets[b,t])
                    angles_temp.append(angle_deg)
                    aae_frame.update(angle_deg) # per frame

                    auc_once = calc_auc(gaze_coords[b,t], gaze_targets[b,t])
                    auc_temp.append(auc_once)
                    auc_frame.update(auc_once)
                aae_temporal.update(np.mean(angles_temp)) # per video segment
                auc_temporal.update(np.mean(auc_temp))

            to_print = '[Batch {}/{}]'.format(batch_idx, len(test_iterator))
            to_print += '[Gaze::aae_frame {:.3f}[avg:{:.3f}], aae_temporal {:.3f}[avg:{:.3f}],'.format(aae_frame.val,
                                                                                                       aae_frame.avg,
                                                                                                       aae_temporal.val,
                                                                                                       aae_temporal.avg)
            to_print += '::auc_frame {:.3f}[avg:{:.3f}], auc_temporal {:.3f}[avg:{:.3f}]]'.format(auc_frame.val,
                                                                                                  auc_frame.avg,
                                                                                                  auc_temporal.val,
                                                                                                  auc_temporal.avg)
            print_and_save(to_print, log_file)



def validate_mfnet_mo(model, criterion, test_iterator, num_outputs, use_gaze, use_hands, cur_epoch, dataset, log_file):
    loss_meters = [AverageMeter() for _ in range(num_outputs)]
    losses = AverageMeter()
    top1_meters = [AverageMeter() for _ in range(num_outputs)]
    top5_meters = [AverageMeter() for _ in range(num_outputs)]
    task_outputs = [[] for _ in range(num_outputs)]

    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, video_names) in enumerate(test_iterator):
            inputs = inputs.cuda()
            outputs, coords, heatmaps = model(inputs)
            targets = targets.cuda().transpose(0, 1)

            if use_gaze or use_hands:
                cls_targets = targets[:num_outputs, :].long()
            else:
                cls_targets = targets
            assert len(cls_targets) == num_outputs

            losses_per_task = []
            for output, target in zip(outputs, cls_targets):
                loss_for_task = criterion(output, target)
                losses_per_task.append(loss_for_task)

            loss = sum(losses_per_task)

            gaze_coord_loss, hand_coord_loss = 0, 0
            if use_gaze:
                gaze_targets = targets[num_outputs:num_outputs + 16, :].transpose(1, 0).reshape(-1, 8, 1, 2)
                # for a single shared layer representation of the two signals
                # for gaze slice the first element
                gaze_coords = coords[:, :, 0, :]
                gaze_coords.unsqueeze_(2)  # unsqueeze to add the extra dimension for consistency
                gaze_heatmaps = heatmaps[:, :, 0, :]
                gaze_heatmaps.unsqueeze_(2)
                gaze_coord_loss = calc_coord_loss(gaze_coords, gaze_heatmaps, gaze_targets)
                loss = loss + gaze_coord_loss
            if use_hands:
                hand_targets = targets[-32:, :].reshape(-1, 8, 2, 2)
                # for hands slice the last two elements, first is left, second is right hand
                hand_coords = coords[:, :, -2:, :]
                hand_heatmaps = heatmaps[:, :, -2:, :]
                hand_coord_loss = calc_coord_loss(hand_coords, hand_heatmaps, hand_targets)
                loss = loss + hand_coord_loss

            batch_size = outputs[0].size(0)

            batch_preds = []
            for j in range(batch_size):
                txt_batch_preds = "{}".format(video_names[j])
                for ind in range(num_outputs):
                    txt_batch_preds += ", "
                    res = np.argmax(outputs[ind][j].detach().cpu().numpy())
                    label = cls_targets[ind][j].detach().cpu().numpy()
                    task_outputs[ind].append([res, label])
                    txt_batch_preds += "T{} P-L:{}-{}".format(ind, res, label)
                batch_preds.append(txt_batch_preds)

            losses.update(loss.item(), batch_size)
            for ind in range(num_outputs):
                t1, t5 = accuracy(outputs[ind].detach().cpu(), cls_targets[ind].detach().cpu(), topk=(1, 5))
                top1_meters[ind].update(t1.item(), batch_size)
                top5_meters[ind].update(t5.item(), batch_size)
                loss_meters[ind].update(losses_per_task[ind].item(), batch_size)

            to_print = '[Batch {}/{}]'.format(batch_idx, len(test_iterator))
            for ind in range(num_outputs):
                to_print += '[T{}::Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]],'.format(ind,
                                                                                 top1_meters[ind].val, top1_meters[ind].avg,
                                                                                 top5_meters[ind].val, top5_meters[ind].avg)
            to_print+= '\n\t{}'.format(batch_preds)
            print_and_save(to_print, log_file)

        to_print = '{} Results: Loss {:.3f}'.format(dataset, losses.avg)
        for ind in range(num_outputs):
            to_print += ', T{}::Top1 {:.3f}, Top5 {:.3f}'.format(ind, top1_meters[ind].avg, top5_meters[ind].avg)
        print_and_save(to_print, log_file)
    return [tasktop1.avg for tasktop1 in top1_meters], task_outputs

def train_mfnet_h(model, optimizer, criterion, train_iterator, mixup_alpha, cur_epoch, log_file, gpus,
              lr_scheduler=None):
    batch_time, losses, cls_losses, c_losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(),\
                                                           AverageMeter(), AverageMeter()
    model.train()

    if not isinstance(lr_scheduler, CyclicLR):
        lr_scheduler.step()

    print_and_save('*********', log_file)
    print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets, points) in enumerate(train_iterator): # left_track.shape = [batch, 8, 2]
        if isinstance(lr_scheduler, CyclicLR):
            lr_scheduler.step()

        inputs = torch.tensor(inputs, requires_grad=True).cuda(gpus[0])
        target_class = torch.tensor(targets).cuda(gpus[0])
        target_var = torch.tensor(points).cuda(gpus[0])

        output, coords, heatmaps = model(inputs)

        cls_loss = criterion(output, target_class)
        coord_loss = calc_coord_loss(coords, heatmaps, target_var)
        loss = cls_loss + coord_loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        t1, t5 = accuracy(output.detach().cpu(), target_class.cpu(), topk=(1, 5))
        top1.update(t1.item(), output.size(0))
        top5.update(t5.item(), output.size(0))
        cls_losses.update(cls_loss.item(), output.size(0))
        c_losses.update(coord_loss.item(), output.size(0))
        losses.update(loss.item(), output.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        print_and_save(
            '[Epoch:{}, Batch {}/{} in {:.3f} s][Loss(f|cls|coo) {:.4f} | {:.4f} | {:.4f} [avg:{:.4f} | {:.4f} | {:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]], LR {:.6f}'.format(
                cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, cls_losses.val, c_losses.val,
                losses.avg, cls_losses.avg, c_losses.avg, top1.val, top1.avg, top5.val, top5.avg, lr_scheduler.get_lr()[0]),
            log_file)
        print_and_save("Epoch train time: {}".format(batch_time.sum), log_file)


def test_mfnet_h(model, criterion, test_iterator, cur_epoch, dataset, log_file, gpus):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        model.eval()
        print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets, points) in enumerate(test_iterator):
            inputs = inputs.cuda(gpus[0])
            target_class = targets.cuda(gpus[0])
            target_var = points.cuda(gpus[0])

            output, coords, heatmaps = model(inputs)

            cls_loss = criterion(output, target_class)
            coord_loss = calc_coord_loss(coords, heatmaps, target_var)
            loss = cls_loss + coord_loss

            t1, t5 = accuracy(output.detach().cpu(), target_class.detach().cpu(), topk=(1, 5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))

            print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        print_and_save(
            '{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg),
            log_file)
    return top1.avg

def validate_mfnet_hands(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, cls_losses, coo_losses,  top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    outputs = []

    print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, points, video_names) in enumerate(test_iterator):
            inputs = inputs.cuda()
            target_class = targets.cuda()
            target_var = points.cuda()

            output, coords, heatmaps = model(inputs)

            cls_loss = criterion(output, target_class)
            coord_loss = calc_coord_loss(coords, heatmaps, target_var)
            loss = cls_loss + coord_loss

            batch_preds = []
            for j in range(output.size(0)):
                res = np.argmax(output[j].detach().cpu().numpy())
                label = targets[j].cpu().numpy()
                outputs.append([res, label])
                batch_preds.append("{}, P-L:{}-{}".format(video_names[j], res, label))

            t1, t5 = accuracy(output.detach().cpu(), targets.cpu(), topk=(1, 5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))
            cls_losses.update(cls_loss.item(), output.size(0))
            coo_losses.update(coord_loss.item(), output.size(0))

            print_and_save('[Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]\n\t{}'.format(
                batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg, batch_preds), log_file)
        print_and_save(
            '{} Results: Loss(f|cls|coo) {:.4f} | {:.4f} | {:.4f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg,
                                                                                                    cls_losses.avg,
                                                                                                    coo_losses.avg,
                                                                                                    top1.avg, top5.avg),
            log_file)
    return top1.avg, outputs
