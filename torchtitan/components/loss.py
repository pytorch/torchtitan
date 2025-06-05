# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, TypeAlias

import torch

from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def build_cross_entropy_loss(job_config: JobConfig):
    loss_fn = cross_entropy_loss
    if job_config.training.compile:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """

    @functools.wraps(unwrapped_loss_fn)
    def accumulated_loss_fn(*args, **kwargs):
        loss = unwrapped_loss_fn(*args, **kwargs)
        return loss / accumulation_steps

    return accumulated_loss_fn
