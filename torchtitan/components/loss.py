# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import TypeAlias

import torch

from torchtitan.config import Configurable
from torchtitan.tools.logging import logger
from torchtitan.distributed import ParallelDims
from functools import partial

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

@dataclass
class LossOutput:
    """
    A wrapper for the loss output, 
    contains a main loss from which gradients will be computed, 
    and auxiliary losses that can be used for logging 
    but will not contribute to the gradient computation.
    """
    main: torch.Tensor
    aux: dict[str, torch.Tensor] = field(default_factory=dict)

LossFunction: TypeAlias = Callable[..., LossOutput]

class Loss(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        type: str = "cross_entropy"
        compile: bool = False
        extra_kwargs: dict = field(default_factory=dict)
        
    def __init__(self, config: Config) -> None:
        self.config = config
        match self.config.type, self.config.compile:
            case "cross_entropy", True:
                logger.info("Using cross-entropy loss with compilation.")
                self.loss_fn = torch.compile(partial(cross_entropy_loss, **self.config.extra_kwargs))
            case "cross_entropy", False:
                logger.info("Using cross-entropy loss without compilation.")
                self.loss_fn = partial(cross_entropy_loss, **self.config.extra_kwargs)
            case "mse", True:
                logger.info("Using MSE loss with compilation.")
                self.loss_fn = torch.compile(partial(mse_loss, **self.config.extra_kwargs))
            case "mse", False:
                logger.info("Using MSE loss without compilation.")
                self.loss_fn = partial(mse_loss, **self.config.extra_kwargs)
            case _, _:
                raise ValueError(f"Unsupported loss type: {self.config.type}")

    def __call__(self, pred: torch.Tensor, labels: torch.Tensor) -> LossOutput:

def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor, z_loss_weight: float = 0) -> LossOutput:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    logits = pred.flatten(0, 1).float()
    cross_entropy_loss = torch.nn.functional.cross_entropy(
        logits,
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )

    if not z_loss_weight == 0.0:
        return LossOutput(
            main = cross_entropy_loss,
        )

    z_squared = logits.logsumexp(-1).pow(2)
    z_loss = (z_squared * (labels != IGNORE_INDEX)).sum()

    return LossOutput(
        main = cross_entropy_loss + z_loss_weight * z_loss,
        aux = {"z_loss": z_loss},
    )

def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> LossOutput:
    """Common MSE loss function with sum reduction for Transformer models training."""
    return LossOutput(
        main = torch.nn.functional.mse_loss(
            pred.float(), labels.float().detach(), reduction="sum"
        )
    )
