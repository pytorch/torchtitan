# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


@dataclass
class BaseModelArgs:
    """All ModelArgs should inherit from this class.

    The only usage of this class is type checking but allows us to extend common
    arguments to all models in the future.
    """

    _enforced: str = "This field is used to enforce all fields have defaults."

    @abstractmethod
    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        pass

    @abstractmethod
    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        pass


def get_dense_nparams_and_flops(
    model_args: BaseModelArgs, model: nn.Module, seq_len: int
) -> tuple[int, float]:
    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.children()
        if isinstance(m, nn.Embedding)
    )

    l, h, q, t = (
        model_args.n_layers,
        model_args.n_heads,
        model_args.dim // model_args.n_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    return nparams, num_flops_per_token


def get_moe_nparams_and_flops(
    model_args: BaseModelArgs, model: nn.Module, seq_len: int
) -> tuple[int, float]:
    """
    Calculate nparams and nflops for MoE model
    """
    nparams_embedding = 0
    nparams_moe_router = 0
    nparams_shared_experts = 0
    nparams_experts = 0
    nparams_dense = 0

    for name, p in model.named_parameters():
        if "embedding" in name:
            nparams_embedding += p.numel()
            nparams_dense += p.numel()
        elif "moe.shared_experts" in name:
            nparams_shared_experts += p.numel()
        elif "moe.router" in name:
            nparams_moe_router += p.numel()
        elif "moe.experts" in name:
            nparams_experts += p.numel()
        else:
            nparams_dense += p.numel()

    nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
    nparams = nparams_dense + nparams_sparse
    nparams_sparse_active = (
        nparams_moe_router
        + nparams_shared_experts
        + nparams_experts * model_args.moe_args.top_k // model_args.moe_args.num_experts
    )

    logger.info(
        f"Total parameter count: dense {nparams_dense:,}, "
        f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
    )

    l, h, q, t = (
        model_args.n_layers,
        model_args.n_heads,
        model_args.dim // model_args.n_heads,
        seq_len,
    )

    num_flops_per_token = (
        6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
        + 12 * l * h * q * t
    )

    return nparams, num_flops_per_token


class ModelProtocol(Protocol):
    """Defines the interface for a model class.

    This is used to enforce that all model classes have some methods that are
    required by the trainer.
    """

    def __init__(self, model_args: BaseModelArgs) -> None:
        pass

    @abstractmethod
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights.

        Args:
            buffer_device: Optional device to place buffers on during initialization.
        """
        pass
