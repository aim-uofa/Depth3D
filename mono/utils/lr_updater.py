import os
import numpy as np
from PIL import Image

from torch import nn
import torch.nn.init as initer

class LrUpdater():
    """Refer to LR Scheduler in MMCV.
    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_lr(self, iter, max_iters, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, iter, max_iters, optimizer):
        if isinstance(optimizer, dict):
            lr_groups = {}
            for k in optimizer.keys():
                _lr_group = [
                    self.get_lr(iter, max_iters,  _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(iter, max_iters, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, optimizer):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(optimizer, dict):
            self.base_lr = {}
            for k, optim in optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr'] for group in optimizer.param_groups
            ]

    # def before_train_epoch(self, runner):
    #     if self.warmup_iters is None:
    #         epoch_len = len(runner.data_loader)
    #         self.warmup_iters = self.warmup_epochs * epoch_len
    #
    #     if not self.by_epoch:
    #         return
    #
    #     self.regular_lr = self.get_regular_lr(runner)
    #     self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, optimizer, cur_iter, max_iters):
        self.regular_lr = self.get_regular_lr(cur_iter, max_iters, optimizer)
        if self.warmup is None or cur_iter >= self.warmup_iters:
            self._set_lr(optimizer, self.regular_lr)
        else:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(optimizer, warmup_lr)

    def get_curr_lr(self, cur_iter):
        if self.warmup is None or cur_iter >= self.warmup_iters:
            return self.regular_lr
        else:
            return self.get_warmup_lr(cur_iter)


class PolyLrUpdater(LrUpdater):

    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = 0.0
        super(PolyLrUpdater, self).__init__(**kwargs)

    def get_lr(self, iter, max_iters, base_lr):
        progress = iter
        max_progress = max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr

def build_lr_schedule_with_cfg(cfg):
  

def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr
