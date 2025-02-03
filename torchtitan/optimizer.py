# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR
from torchtitan.config_manager import JobConfig


class OptimizersContainer(Stateful):
    """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages
    and saving/loading optimizer state_dict at checkpoint.
    """

    def __init__(
        self, model_parts: List[nn.Module], optimizer_kwargs: Dict[str, Any], name: str
    ) -> None:
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            if name == "Adam":
                # TODO: make the optimizer options configurable by toml/cmd args
                optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
            elif name == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            else:
                raise NotImplementedError(f"Optimizer {name} not added.")
            self.optimizers.append(optimizer)
        self._validate_length(len(self.model_parts))

    def _validate_length(self, expected_length) -> None:
        assert expected_length == len(
            self.optimizers
        ), "Must pass one optimizer per model part or per param if using OptimizersInBackwardContainer"

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))


class OptimizersInBackwardContainer(OptimizersContainer):
    """Optimiers in backward to skip .step() and .zero_grad()"""

    def __init__(
        self, model_parts: List[nn.Module], optimizer_kwargs: Dict[str, Any], name: str
    ) -> None:
        self.optimizers = []
        self.model_parts = model_parts
        optim_dict = {}
        for model in self.model_parts:
            if name == "Adam":
                # TODO: make the optimizer options configurable by toml/cmd args
                optim_dict.update(
                    {
                        param: torch.optim.Adam([param], **optimizer_kwargs)
                        for param in model.parameters()
                    }
                )
            elif name == "AdamW":
                optim_dict.update(
                    {
                        param: torch.optim.AdamW([param], **optimizer_kwargs)
                        for param in model.parameters()
                    }
                )
            else:
                raise NotImplementedError(f"Optimizer {name} not added.")

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

            self.optimizers.extend([optim_dict[param] for param in model.parameters()])

        self._validate_length(
            sum(
                len([param for param in model.parameters()])
                for model in self.model_parts
            )
        )

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


# consider split between PP and non-PP
def build_optimizers(
    model_parts: List[nn.Module], job_config: JobConfig
) -> OptimizersContainer:
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.experimental.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Optimizers in backward is not supported with pipeline parallelism."
        )
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


class SchedulersContainer(Stateful):
    """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

    def __init__(self, optimizers, lr_lambda) -> None:
        self.schedulers = []
        for optimizer in optimizers:
            self.schedulers.append(LambdaLR(optimizer, lr_lambda=lr_lambda))

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        # Currently, we have one scheduler per optimizer. However, when using MultiSchedule PP or optimizer-in-backward,
        # there are multiple optimizers and schedulers, but the scheduler state_dict remains the same for all.
        # Therefore, we only save the first one and later load it for all.
        assert (
            len(self.schedulers) > 0
        ), "Must have at least one scheduler to save state_dict"
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned with in scheduler.state_dict() is `last_epoch`,
        # which is an integer that will be automatically copied. As long as `training.steps` and `training.warmup_steps` remain
        # unchanged when resuming from a checkpoint, this approach is safe. We call `.copy()` here to ensure extra safety.
        for scheduler in self.schedulers:
            scheduler.load_state_dict(state_dict.copy())


def build_lr_schedulers(optimizers, job_config: JobConfig) -> SchedulersContainer:
    warmup_steps = int(job_config.training.warmup_steps)
    decay_steps = float(max(1, job_config.training.steps - warmup_steps))
    lr_lambda = functools.partial(linear_warmup_linear_decay, warmup_steps, decay_steps)

    return SchedulersContainer(optimizers, lr_lambda)
