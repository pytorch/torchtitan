# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch_remat as remat

from torchtitan.models.common.activation import ActivationFn, SwiGLU
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

# Shape suffix legend:
#   T = token dimensions, D = model dimension, F = feed-forward hidden dimension

__all__ = ["FeedForward", "SigmoidGatedFeedForward", "compute_ffn_hidden_dim"]


def _make_fused_gate_up_init(
    gate_init: Callable,
    up_init: Callable,
    *,
    gate_up_axis: int,
) -> Callable:
    """Build an initializer for a fused gate/up weight from per-half initializers.

    The fused weight has a size-2 ``gate_up_axis`` (index 0 = gate / stock w1,
    index 1 = up / stock w3). Each half is initialized with its own initializer
    because the gate and up projections differ (e.g. up shares w2's depth-scaled
    init), so initializing the whole tensor at once would mis-init the up half.
    Used through logical views of dense and grouped fused linear weights.
    """

    def _init(t: torch.Tensor) -> None:
        gate_idx: list[int | slice] = [slice(None)] * t.ndim
        up_idx: list[int | slice] = [slice(None)] * t.ndim
        gate_idx[gate_up_axis] = 0
        up_idx[gate_up_axis] = 1
        gate_init(t[tuple(gate_idx)])  # gate (stock w1)
        up_init(t[tuple(up_idx)])  # up (stock w3)

    return _init


def _make_fused_linear_init(
    gate_init: Callable,
    up_init: Callable,
    *,
    output_axis: int = 0,
) -> Callable:
    """Build an initializer for an interleaved gate/up linear weight."""
    init_logical_weight = _make_fused_gate_up_init(
        gate_init,
        up_init,
        gate_up_axis=output_axis + 1,
    )

    def _init(t: torch.Tensor) -> None:
        init_logical_weight(t.unflatten(output_axis, (-1, 2)))

    return _init


def _split_dense_gate_up_state_dict(state_dict: dict[str, Any]) -> None:
    """Split dense ``w13`` Linear parameters into logical w1/w3 entries."""
    for param_name in ("weight", "bias"):
        suffix = f"w13.{param_name}"
        for key in tuple(state_dict):
            if not key.endswith(suffix):
                continue
            prefix = key.removesuffix(suffix)
            gate_up = state_dict.pop(key).unflatten(0, (-1, 2))
            state_dict[f"{prefix}w1.{param_name}"] = gate_up[:, 0].contiguous()
            state_dict[f"{prefix}w3.{param_name}"] = gate_up[:, 1].contiguous()


def _split_grouped_gate_up_state_dict(state_dict: dict[str, Any]) -> None:
    """Split grouped ``w13`` parameters into logical w1_EFD/w3_EFD entries."""
    for key in tuple(state_dict):
        if not key.endswith("w13"):
            continue
        prefix = key.removesuffix("w13")
        gate_up = state_dict.pop(key)
        state_dict[f"{prefix}w1_EFD"] = gate_up[:, :, 0, :].contiguous()
        state_dict[f"{prefix}w3_EFD"] = gate_up[:, :, 1, :].contiguous()


def split_fused_gate_up_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Translate native fused gate/up state into logical adapter keys.

    Dense ``FeedForward`` parameters use ``w13.{weight,bias}`` with shape
    ``(2F, ...)`` and become ``w1.{weight,bias}`` / ``w3.{weight,bias}``.
    ``GroupedExperts.w13`` has shape ``(E, F, 2, D)`` and becomes
    ``w1_EFD`` / ``w3_EFD``.
    """
    result = dict(state_dict)
    _split_dense_gate_up_state_dict(result)
    _split_grouped_gate_up_state_dict(result)
    return result


def _fuse_dense_gate_up_state_dict(state_dict: dict[str, Any]) -> None:
    """Pack logical dense w1/w3 entries into a physical ``w13`` Linear."""
    for param_name in ("weight", "bias"):
        suffix = f"w1.{param_name}"
        for gate_key in tuple(state_dict):
            if not gate_key.endswith(suffix):
                continue
            prefix = gate_key.removesuffix(suffix)
            up_key = f"{prefix}w3.{param_name}"
            if up_key not in state_dict:
                continue
            state_dict[f"{prefix}w13.{param_name}"] = torch.stack(
                [state_dict.pop(gate_key), state_dict.pop(up_key)], dim=1
            ).flatten(0, 1)


def _fuse_grouped_gate_up_state_dict(state_dict: dict[str, Any]) -> None:
    """Pack logical grouped w1_EFD/w3_EFD entries into physical ``w13``."""
    for gate_key in tuple(state_dict):
        if not gate_key.endswith("w1_EFD"):
            continue
        prefix = gate_key.removesuffix("w1_EFD")
        up_key = f"{prefix}w3_EFD"
        if up_key not in state_dict:
            continue
        state_dict[f"{prefix}w13"] = torch.stack(
            [state_dict.pop(gate_key), state_dict.pop(up_key)], dim=2
        )


def fuse_gate_up_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Translate logical adapter keys into native fused gate/up state."""
    result = dict(state_dict)
    _fuse_dense_gate_up_state_dict(result)
    _fuse_grouped_gate_up_state_dict(result)
    return result


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(Module):
    """SwiGLU feed-forward with one physical gate-and-up projection.

    ``w13`` stores the interleaved gate and up projections. Config takes the
    **final** hidden_dim (no internal 2/3 scaling). Use
    compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w13: Linear.Config
        w2: Linear.Config
        activation_fn: ActivationFn.Config = field(default_factory=SwiGLU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.activation_fn = config.activation_fn.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up_TF = remat.region(
            self.w13,
            self.remat_region_name("w13"),
            recompute=self.remat_should_recompute("w13"),
        )(x)
        gate_TF, up_TF = gate_up_TF.unflatten(-1, (-1, 2)).unbind(-1)
        remat.recompute_needs_tensor(gate_TF, up_TF)
        out_TD = remat.region(
            self.w2,
            self.remat_region_name("w2"),
            recompute=self.remat_should_recompute("w2"),
        )(self.activation_fn(gate_TF, up_TF))
        remat.recompute_needs_tensor(out_TD)
        return out_TD


class SigmoidGatedFeedForward(FeedForward):
    """SwiGLU feed-forward with a per-token sigmoid gate.

    The output is ``sigmoid(gate(x)) * ffn(x)``. It uses FeedForward's fused
    ``w13`` and ``w2`` projections and adds a separate ``gate`` projection.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_TD = super().forward(x)
        gate_out_TD = remat.region(
            self.gate,
            self.remat_region_name("gate"),
            recompute=self.remat_should_recompute("gate"),
        )(x)
        remat.recompute_needs_tensor(out_TD, gate_out_TD)
        return torch.sigmoid(gate_out_TD) * out_TD
