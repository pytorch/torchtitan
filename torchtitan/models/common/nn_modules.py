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

import torch
import torch.nn as nn

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.base.module import Module


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
    """Configurable nn.RMSNorm with the same input name as LayerNorm."""

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

    # ShardingConfig keys use "input" for both RMSNorm and LayerNorm; some
    # PyTorch versions name the RMSNorm argument "x".
    # pyrefly: ignore[bad-param-name-override]
    def forward(
        self, input: torch.Tensor, *, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Normalize ``input``. With a residual, add it first and also return the sum.

        Example:
            norm(h)                     -> rms_norm(h)
            norm(ffn_out, residual=h)   -> (rms_norm(h + ffn_out), h + ffn_out)
        """
        if residual is None:
            return super().forward(input)
        return self._add_rmsnorm(input, residual)

    # Only calls with a residual compile, so the add runs inside the norm kernel.
    # Compiled q/k head norms would be slower (2.6 vs 1.7 us, 64 tokens, GB300).
    # emulate_precision_casts rounds the sum to bf16 before the norm, as eager
    # does. Without it, the trainer's (grad) and generator's (no_grad) kernels
    # round differently: up to 22% of outputs differ by one bf16 step.
    @local_compile(
        "add_rmsnorm",
        batch_invariant=False,
        options={"emulate_precision_casts": True},
    )
    def _add_rmsnorm(
        self, input: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = residual + input
        return super().forward(residual), residual


__all__ = [
    "Conv1d",
    "Conv2d",
    "GELU",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "RMSNorm",
]
