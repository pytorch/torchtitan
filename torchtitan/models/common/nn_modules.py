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

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols.module import Module


class Conv1d(nn.Conv1d, Module):
    """Configurable nn.Conv1d."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        padding: int = 0
        groups: int = 1
        # Matches the upstream ``nn.Conv1d`` default (differs from
        # ``Linear.Config.bias``, which defaults to False).
        bias: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            groups=config.groups,
            bias=config.bias,
        )


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


class GatedRMSNorm(Module):
    """Apply RMSNorm followed by an elementwise unary gate activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float
        activation_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.activation_fn = config.activation_fn
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = F.rms_norm(
            x.float(),
            (x.shape[-1],),
            self.weight.float(),
            self.eps,
        )
        return (normalized * self.activation_fn(gate.float())).to(input_dtype)


class SiLU(nn.SiLU, Module):
    """Configurable nn.SiLU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()


__all__ = [
    "Conv1d",
    "Conv2d",
    "GELU",
    "GatedRMSNorm",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "RMSNorm",
    "SiLU",
]
