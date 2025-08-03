# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, TypeAlias

import torch

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

try:
    from liger_kernel.transformers.cross_entropy import LigerFusedLinearCrossEntropyLoss
    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def is_liger_kernel_enabled(job_config: JobConfig) -> bool:
    """Check if Liger-Kernel fused linear cross entropy loss is enabled."""
    return job_config.liger_kernel.enable_fused_linear_cross_entropy


def liger_fused_linear_cross_entropy_loss(
    weight: torch.Tensor, input: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Compute fused linear cross entropy loss using Liger-Kernel.
    
    Args:
        weight: Linear layer weight tensor [vocab_size, hidden_dim]
        input: Hidden states tensor [batch_size, seq_len, hidden_dim]
        target: Target labels tensor [batch_size, seq_len]
        
    Returns:
        Loss tensor
    """
    if not LIGER_KERNEL_AVAILABLE:
        raise ImportError(
            "Liger-Kernel is not installed. Please install it with: pip install liger-kernel"
        )
    
    # Reshape input from 3D to 2D for the fused operation
    batch_size, seq_len, hidden_dim = input.shape
    input_2d = input.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
    target_1d = target.view(-1)  # [batch_size * seq_len]
    
    # Initialize the fused loss function
    liger_loss_fn = LigerFusedLinearCrossEntropyLoss()
    
    # Compute fused linear + cross entropy loss
    loss = liger_loss_fn(input_2d, weight, target_1d)
    
    return loss


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
