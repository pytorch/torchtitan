# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from torchtrain.logging_utils import rank0_log


class LinearScheduler:
    def __init__(self, args):
        self.lr_max = args.lr
        self.lr_min = args.lr / 10
        self.lr_warmup_pct = 0.10
        # enforce min of 2 steps for warmup
        self.warmup_steps = max(int(args.steps * self.lr_warmup_pct), 2)

        rank0_log(
            f"LR Warmup Schedule: {self.lr_min} -> {self.lr_max} with {self.warmup_steps} warmup steps"
        )
        self.decay_steps = args.steps - self.warmup_steps
        self.curr_lr = 0

    def set_lr(self, optimizer, step):
        """Set the learning rate for the optimizer"""
        if step < self.warmup_steps:
            self.curr_lr = self.lr_max * (step / self.warmup_steps)
        else:
            self.curr_lr = self.lr_min + (
                (self.lr_max - self.lr_min)
                * (1 - (step - self.warmup_steps) / self.decay_steps)
            )
        # apply across all optim groups
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.curr_lr
        rank0_log(f"Optimizer LR Update: {step=}, lr = {round(self.curr_lr,6)}")
