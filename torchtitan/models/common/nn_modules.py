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
import torch_remat as remat

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
    """Configurable nn.LayerNorm. The norm is the remat region ``<fqn>.norm``."""

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return remat.region(
            super().forward,
            self.remat_region_name("norm"),
            recompute=self.remat_should_recompute("norm"),
        )(input)


class RMSNorm(nn.RMSNorm, Module):
    """Configurable nn.RMSNorm.

    The norm is the remat region ``<fqn>.norm``. Its consumers are usually
    other regions (projections, rope, attention), so a saved producer feeding
    it needs no ``recompute_needs_tensor``. Subclasses override ``_norm``.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return remat.region(
            self._norm,
            self.remat_region_name("norm"),
            recompute=self.remat_should_recompute("norm"),
        )(x)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return nn.RMSNorm.forward(self, x)


class SiLU(nn.SiLU, Module):
    """Configurable nn.SiLU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()


def residual_add(
    module: Module, x: torch.Tensor, branch: torch.Tensor, name: str
) -> torch.Tensor:
    """Add a residual branch as the remat region ``<module fqn>.<name>``.

    The add saves nothing for backward. As a region, a saved add lets the
    branch's saved producer (e.g. an output projection) skip persisting its
    output, and a recomputed add makes torch_remat persist it automatically.
    """
    return remat.region(
        torch.add,
        module.remat_region_name(name),
        recompute=module.remat_should_recompute(name),
    )(x, branch)


__all__ = [
    "Conv1d",
    "Conv2d",
    "GELU",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "residual_add",
    "RMSNorm",
    "SiLU",
]
