# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
from typing import Callable, TypeAlias

import torch

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def build_cross_entropy_loss(job_config: JobConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = cross_entropy_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
    return loss_fn


class RescaleAccumulatedLoss:
    def __init__(self, unwrapped_loss_fn, accumulation_steps):
        self.unwrapped_loss_fn = unwrapped_loss_fn
        self.accumulation_steps = accumulation_steps
        self.skip_rescale = False

        # Copy over attributes from the original function, but don't
        # copy the dict, which interferes with nested wrapping.
        functools.update_wrapper(self, unwrapped_loss_fn, updated=tuple())

    def __call__(self, *args, **kwargs):
        loss = self.unwrapped_loss_fn(*args, **kwargs)
        if self.skip_rescale:
            return loss
        return loss / self.accumulation_steps

    @contextlib.contextmanager
    def no_rescale(self):
        """Context manager for disabling rescaling"""
        previous = self.skip_rescale
        self.skip_rescale = True
        try:
            yield
        finally:
            self.skip_rescale = previous


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """
    return RescaleAccumulatedLoss(unwrapped_loss_fn, accumulation_steps)


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common MSE loss function for Transformer models training."""
    return torch.nn.functional.mse_loss(pred.float(), labels.float().detach())


def build_mse_loss(job_config: JobConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = mse_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
    return loss_fn
