# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable wrappers around standard ``torch.nn`` modules.

Each class uses diamond inheritance (``nn.X`` + ``Module``) so that:
- The module hierarchy stays flat (no extra wrapper layer).
- All ``nn.X`` logic (forward, state_dict, etc.) is reused as-is.
- The ``Module`` protocol is satisfied and ``build()`` is inherited
  from ``Configurable.Config``.

Each ``Config`` only exposes the fields that current callsites set;
add more if a new callsite needs them.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.protocols.module import Module


class Conv2d(nn.Conv2d, Module):
    """Configurable nn.Conv2d."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        padding: int = 0
        # Matches the upstream ``nn.Conv2d`` default (differs from
        # ``Linear.Config.bias``, which defaults to False).
        bias: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=config.bias,
        )


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int
        enable_sp: bool | None = None

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)
        self.enable_sp = config.enable_sp
        self.tp_pg: dist.ProcessGroup | None = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Runs vocab-parallel embedding when a local TP process group is set."""
        weight = self.weight
        if self.tp_pg is None or isinstance(weight, DTensor):
            return F.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        assert self.enable_sp is not None
        chunk_size = (
            self.num_embeddings + dist.get_world_size(self.tp_pg) - 1
        ) // dist.get_world_size(self.tp_pg)
        offset = dist.get_rank(self.tp_pg) * chunk_size
        mask = (input >= offset) & (input < offset + weight.shape[0])
        local_input = (input - offset).clamp(0, weight.shape[0] - 1)
        out = F.embedding(
            local_input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        out = out * mask.unsqueeze(-1).to(out.dtype)
        tp_out_type = spmd.S(1) if self.enable_sp else spmd.I
        return spmd.redistribute(
            out,
            self.tp_pg,
            src=spmd.P,
            dst=tp_out_type,
        )


class GELU(nn.GELU, Module):
    """Configurable nn.GELU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        approximate: str = "none"

    def __init__(self, config: Config):
        super().__init__(approximate=config.approximate)


class GroupNorm(nn.GroupNorm, Module):
    """Configurable nn.GroupNorm."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_groups: int
        num_channels: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__(
            config.num_groups,
            config.num_channels,
            eps=config.eps,
        )


class Identity(nn.Identity, Module):
    """Configurable nn.Identity."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()


class LayerNorm(nn.LayerNorm, Module):
    """Configurable nn.LayerNorm."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int
        eps: float = 1e-5
        elementwise_affine: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )


class Linear(nn.Linear, Module):
    """Configurable nn.Linear."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        bias: bool = False

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.out_features,
            bias=config.bias,
        )


class RMSNorm(nn.RMSNorm, Module):
    """Configurable nn.RMSNorm."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int
        eps: float = 1e-5
        elementwise_affine: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )


class SiLU(nn.SiLU, Module):
    """Configurable nn.SiLU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()


__all__ = [
    "Conv2d",
    "Embedding",
    "GELU",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "Linear",
    "RMSNorm",
    "SiLU",
]
