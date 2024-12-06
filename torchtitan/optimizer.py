# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torch.optim.lr_scheduler import LambdaLR
from torchtitan.config_manager import JobConfig


class OptimizersContainer:
    """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

    def __init__(self, model_parts, optimizer_kwargs, name):
        self.optimizers = []
        for model in model_parts:
            if name == "Adam":
                # TODO: make the optimizer options configurable by toml/cmd args
                optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
            elif name == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            else:
                raise NotImplementedError(f"Optimizer {name} not added.")
            self.optimizers.append(optimizer)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()


class OptimizersInBackwardContainer(OptimizersContainer):
    """Optimiers in backward to skip .step() and .zero_grad()"""

    def __init__(self, model_parts, optimizer_kwargs, name):
        self.optimizers = []
        for model in model_parts:
            if name == "Adam":
                # TODO: make the optimizer options configurable by toml/cmd args
                optim_dict = {
                    param: torch.optim.Adam([param], **optimizer_kwargs)
                    for param in model.parameters()
                }
            elif name == "AdamW":
                optim_dict = {
                    param: torch.optim.AdamW([param], **optimizer_kwargs)
                    for param in model.parameters()
                }
            else:
                raise NotImplementedError(f"Optimizer {name} not added.")

            def optim_hook(param) -> None:
                optim_dict[param].step()
                optim_dict[param].zero_grad()

            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

            self.optimizers.append([optim_dict[param] for param in model.parameters()])

    def step(self):
        pass

    def zero_grad(self):
        pass


# consider split between PP and non-PP
def build_optimizers(model_parts, job_config: JobConfig):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    fused = job_config.optimizer.fused
    optimizer_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": fused,
        "foreach": not fused,
    }

    return (
        OptimizersContainer(model_parts, optimizer_kwargs, name)
        if not optim_in_bwd
        else OptimizersInBackwardContainer(model_parts, optimizer_kwargs, name)
    )


def linear_warmup_linear_decay(
    warmup_steps: int, decay_steps: int, current_step: int
) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_steps + 1))

    else:
        # linear decay
        normalized_step = decay_steps - (current_step - warmup_steps)
        curr_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

    return curr_adjustment


class SchedulersContainer:
    """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

    def __init__(self, optimizers, lr_lambda):
        self.schedulers = []
        for optimizer in optimizers:
            self.schedulers.append(LambdaLR(optimizer, lr_lambda=lr_lambda))

    def step(self):
        for schedulers in self.schedulers:
            schedulers.step()


class SchedulersInBackwardContainer(SchedulersContainer):
    """Util for calling step on multiple learning rate schedulers when optimizers are in backward"""

    def __init__(self, optimizers, lr_lambda):
        # all the schedulers for each optimizer group are the same, here we only store the first one
        # to self.schedulers follow the same structure as SchedulersContainer, but store all of them
        # to self.all_schedulers for container.step() to call
        self.schedulers = []
        for optim_group in optimizers:
            scheduler_group = []
            for sub_optim in optim_group:
                scheduler_group.append(LambdaLR(sub_optim, lr_lambda=lr_lambda))
            self.schedulers.append(scheduler_group)

    def step(self):
        for scheduler_group in self.schedulers:
            for scheduler in scheduler_group:
                scheduler.step()


def build_lr_schedulers(optimizers, job_config: JobConfig):
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    warmup_steps = int(job_config.training.warmup_steps)
    decay_steps = float(max(1, job_config.training.steps - warmup_steps))
    lr_lambda = functools.partial(linear_warmup_linear_decay, warmup_steps, decay_steps)

    return (
        SchedulersContainer(optimizers, lr_lambda)
        if not optim_in_bwd
        else SchedulersInBackwardContainer(optimizers, lr_lambda)
    )
