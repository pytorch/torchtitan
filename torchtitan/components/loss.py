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

class CrossEntropyLoss(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = True
        compile: bool = False
        z_loss_weight: float = 0.0
        ignore_index: int = IGNORE_INDEX
        
    def __init__(self, config: Config) -> None:
        self.config = config
        self.loss_fn = torch.compile(self.cross_entropy_loss) if self.config.compile else self.cross_entropy_loss

    def __call__(self, pred: torch.Tensor, labels: torch.Tensor) -> LossOutput:
        return self.loss_fn(pred, labels)

    def cross_entropy_loss(self, pred: torch.Tensor, labels: torch.Tensor) -> LossOutput:
        """Cross-entropy loss with sum reduction for token-based normalization."""
        labels = labels.flatten(0, 1)
        logits = pred.flatten(0, 1).float()
        cross_entropy_loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction="sum",
            ignore_index=self.config.ignore_index,
        )

        if not self.config.z_loss_weight == 0.0:
            return LossOutput(
                main = cross_entropy_loss,
            )

        z_squared = logits.logsumexp(-1).pow(2)
        z_loss = (z_squared * (labels != self.config.ignore_index)).sum()

        return LossOutput(
            main = cross_entropy_loss + self.config.z_loss_weight * z_loss,
            aux = {
                "z_loss": z_loss,
                "npt_loss": cross_entropy_loss,
            },
        )

class MSELoss(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        compile: bool = False
        enable: bool = False
        
    def __init__(self, config: Config) -> None:
        self.config = config
        self.loss_fn = torch.compile(self.mse_loss) if self.config.compile else self.mse_loss

    def __call__(self, pred: torch.Tensor, labels: torch.Tensor) -> LossOutput:
        return self.loss_fn(pred, labels)

    @staticmethod
    def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> LossOutput:
        """Common MSE loss function with sum reduction for Transformer models training."""
        return LossOutput(
            main = torch.nn.functional.mse_loss(
                pred.float(), labels.float().detach(), reduction="sum"
            )
        )

class Loss(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        cross_entropy: CrossEntropyLoss.Config = field(default_factory=CrossEntropyLoss.Config)
        mse: MSELoss.Config = field(default_factory=MSELoss.Config)
        
    def __init__(self, config: Config) -> None:
        """ check only one loss is enabled """
        self.config = config
        match (self.config.cross_entropy.enable, self.config.mse.enable):
            case (True, True):
                raise ValueError("Only one loss can be enabled at a time.")
            case (False, False):
                logger.warning("No loss is enabled. Loss will return zero.")
            case (True, False):
                self.loss = CrossEntropyLoss(self.config.cross_entropy)
            case (False, True):
                self.loss = MSELoss(self.config.mse)
            case _:
                raise ValueError("Invalid loss configuration.")

    def __call__(self, pred: torch.Tensor, labels: torch.Tensor) -> LossOutput:
        return self.loss(pred, labels)
        

