# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.distributed.spmd_types import (
    spmd_mesh_group,
    spmd_type_for_axis,
    spmd_validate_redistributions,
)
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


def split_fused_gate_up_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Return a state dict with ``w13`` parameters split into logical w1/w3."""
    result = dict(state_dict)
    for param_name in ("weight", "bias"):
        suffix = f"w13.{param_name}"
        for key in tuple(result):
            if not key.endswith(suffix):
                continue
            prefix = key[: -len(suffix)]
            gate_up = result.pop(key).unflatten(0, (-1, 2))
            result[f"{prefix}w1.{param_name}"] = gate_up[:, 0].contiguous()
            result[f"{prefix}w3.{param_name}"] = gate_up[:, 1].contiguous()

    for key in tuple(result):
        if not key.endswith("w13"):
            continue
        prefix = key[: -len("w13")]
        gate_up = result.pop(key)
        result[f"{prefix}w1_EFD"] = gate_up[:, :, 0, :].contiguous()
        result[f"{prefix}w3_EFD"] = gate_up[:, :, 1, :].contiguous()
    return result


def fuse_gate_up_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a state dict with logical w1/w3 parameters packed into ``w13``."""
    result = dict(state_dict)
    for param_name in ("weight", "bias"):
        suffix = f"w1.{param_name}"
        for gate_key in tuple(result):
            if not gate_key.endswith(suffix):
                continue
            prefix = gate_key[: -len(suffix)]
            up_key = f"{prefix}w3.{param_name}"
            if up_key not in result:
                continue
            result[f"{prefix}w13.{param_name}"] = torch.stack(
                [result.pop(gate_key), result.pop(up_key)], dim=1
            ).flatten(0, 1)

    for gate_key in tuple(result):
        if not gate_key.endswith("w1_EFD"):
            continue
        prefix = gate_key[: -len("w1_EFD")]
        up_key = f"{prefix}w3_EFD"
        if up_key not in result:
            continue
        result[f"{prefix}w13"] = torch.stack(
            [result.pop(gate_key), result.pop(up_key)], dim=2
        )
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

    _tp_input_redistribution: (
        tuple[spmd.PerMeshAxisSpmdType, spmd.PerMeshAxisSpmdType] | None
    ) = None
    _tp_output_redistribution: (
        tuple[spmd.PerMeshAxisSpmdType, spmd.PerMeshAxisSpmdType] | None
    ) = None

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

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        sharding_config = self._sharding_config
        if (
            type(self) is FeedForward
            and parallel_dims.spmd_backend == "spmd_types"
            and sharding_config is not None
        ):
            in_src = sharding_config.in_src_shardings or {}
            in_dst = sharding_config.in_dst_shardings or {}
            out_src = sharding_config.out_src_shardings
            out_dst = sharding_config.out_dst_shardings
            if (
                "x" in in_src
                and "x" in in_dst
                and out_src is not None
                and not isinstance(out_src, tuple)
                and out_dst is not None
            ):
                spmd_validate_redistributions(sharding_config)
                self._tp_input_redistribution = (
                    spmd_type_for_axis(in_src["x"], MeshAxisName.TP),
                    spmd_type_for_axis(in_dst["x"], MeshAxisName.TP),
                )
                self._tp_output_redistribution = (
                    spmd_type_for_axis(out_src, MeshAxisName.TP),
                    spmd_type_for_axis(out_dst, MeshAxisName.TP),
                )
                self._sharding_config = replace(
                    sharding_config,
                    in_dst_shardings=None,
                    out_src_shardings=out_dst,
                    out_dst_shardings=None,
                )
        super().parallelize(parallel_dims)

    @staticmethod
    def _redistribute_tp(
        x: torch.Tensor,
        redistribution: (
            tuple[spmd.PerMeshAxisSpmdType, spmd.PerMeshAxisSpmdType] | None
        ),
    ) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is None or redistribution is None:
            return x
        src, dst = redistribution
        return spmd.redistribute(
            x,
            tp_group,
            src=src,
            dst=dst,
            backward_options={"op_dtype": x.dtype},
        )

    def _gate_up_projection(self, x_TD: torch.Tensor) -> torch.Tensor:
        if not getattr(self.w13, "performs_tp_input_all_gather", False):
            x_TD = self._redistribute_tp(x_TD, self._tp_input_redistribution)
        return self.w13(x_TD)

    def _output_projection(self, h_TF: torch.Tensor) -> torch.Tensor:
        out_TD = self.w2(h_TF)
        if not getattr(self.w2, "performs_tp_output_reduce_scatter", False):
            out_TD = self._redistribute_tp(out_TD, self._tp_output_redistribution)
        return out_TD

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up_TF = remat.region(
            self._gate_up_projection,
            self.remat_region_name("w13"),
            recompute=self.remat_should_recompute("w13"),
        )(x)
        gate_TF, up_TF = gate_up_TF.unflatten(-1, (-1, 2)).unbind(-1)
        remat.recompute_needs_tensor(gate_TF, up_TF)
        out_TD = remat.region(
            self._output_projection,
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
