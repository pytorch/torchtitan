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
            # Always recomputed: the norm is cheap, and its consumers then
            # re-derive their saved input instead of keeping it.
            recompute=True,
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
            # Always recomputed: the norm is cheap, and its consumers then
            # re-derive their saved input instead of keeping it.
            recompute=True,
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
    module: Module,
    x: torch.Tensor,
    branch: torch.Tensor,
    name: str,
    *,
    recompute: bool,
) -> torch.Tensor:
    """Add a residual branch as the remat region ``<module fqn>.<name>``.

    The add saves nothing for backward, so its choice is fixed by where its
    output goes rather than by the save policy:

    - ``recompute=True`` when the block reads the sum again (e.g. the
      mid-block residual feeding the FFN norm). Replay is cheap, and a saved
      add would persist the sum even under full recomputation.
    - ``recompute=False`` when the sum only leaves the block. The add is
      skipped during replay, so the branch's saved producer (e.g. the output
      projection) does not persist its output for it.
    """
    return remat.region(
        torch.add,
        module.remat_region_name(name),
        recompute=recompute,
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
