# ------------------------------------------------------------------------------------------------
# Copyright (c) MonoDETR
# ------------------------------------------------------------------------------------------------
import torch.optim.lr_scheduler as lr_sched
import math


def build_lr_scheduler(cfg, optimizer, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg["decay_list"]:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg["decay_rate"]
        return cur_decay

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    warmup_lr_scheduler = None
    if cfg["warmup"]:
        warmup_lr_scheduler = CosineWarmupLR(optimizer, num_epoch=5, init_lr=0.00001)
    return lr_scheduler, warmup_lr_scheduler


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.init_lr
            + (base_lr - self.init_lr)
            * (1 - math.cos(math.pi * self.last_epoch / self.num_epoch))
            / 2
            for base_lr in self.base_lrs
        ]
