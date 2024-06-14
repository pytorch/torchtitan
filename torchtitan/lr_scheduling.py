# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim.lr_scheduler import LambdaLR
from torchtitan.config_manager import JobConfig

# global states for scheduling
# these are needed as LambdaLR does not support argument passing
_warmup_steps = 200
_decay_steps = 0


def linear_warmup_linear_decay(current_step: int) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < _warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (_warmup_steps + 1))

    else:
        # linear decay
        normalized_step = _decay_steps - (current_step - _warmup_steps)
        curr_adjustment = 1 - (_decay_steps - normalized_step) / _decay_steps

    return curr_adjustment


def get_lr_schedulers(optimizers, job_config: JobConfig):
    def _get_lr_scheduler(optimizer):
        """Build a linear warmup and linear decay scheduler"""
        global _warmup_steps, _decay_steps
        _warmup_steps = int(job_config.training.warmup_steps)
        _decay_steps = float(max(1, job_config.training.steps - _warmup_steps))

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup_linear_decay)
        return warmup_scheduler

    class SchedulersContainer:
        """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

        def __init__(self, schedulers):
            self.schedulers = schedulers

        def step(self):
            for schedulers in self.schedulers:
                schedulers.step()

    return SchedulersContainer(
        [_get_lr_scheduler(optimizer) for optimizer in optimizers]
    )
