# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, replace

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

    ``w1`` and ``w3`` remain separate logical configs and checkpoint keys, but
    build into one interleaved ``w13`` Linear. Config takes the **final**
    hidden_dim (no internal 2/3 scaling). Use compute_ffn_hidden_dim() for
    Llama3/4-style dim computation.
    """

    _tp_input_redistribution: (
        tuple[spmd.PerMeshAxisSpmdType, spmd.PerMeshAxisSpmdType] | None
    ) = None
    _tp_output_redistribution: (
        tuple[spmd.PerMeshAxisSpmdType, spmd.PerMeshAxisSpmdType] | None
    ) = None

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config
        activation_fn: ActivationFn.Config = field(
            default_factory=lambda: ActivationFn.Config(
                fn=SwiGLU()  # pyrefly: ignore[bad-argument-type]
            )
        )

    def __init__(self, config: Config):
        super().__init__()
        w1_init = (config.w1.param_init or {}).get("weight")
        w3_init = (config.w3.param_init or {}).get("weight")
        w13_param_init = None
        if w1_init is not None and w3_init is not None:
            w13_param_init = {"weight": _make_fused_linear_init(w1_init, w3_init)}

        w13_config = replace(
            config.w1,
            out_features=2 * config.w1.out_features,
            bias=False,
            param_init=w13_param_init,
        )
        self.w13 = w13_config.build()
        self.w13._logical_output_slices = (
            ("w1", config.w1.out_features),
            ("w3", config.w3.out_features),
        )
        self.w2 = config.w2.build()
        self.activation_fn = config.activation_fn.build()
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

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
        )(self._activation(gate_TF, up_TF))
        remat.recompute_needs_tensor(out_TD)
        return out_TD

    def _activation(self, gate_TF: torch.Tensor, up_TF: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(gate_TF, up_TF)

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Expose the physical w13 parameter as logical w1 and w3 keys."""
        for param_name in ("weight", "bias"):
            key = f"{prefix}w13.{param_name}"
            if key not in state_dict:
                continue
            param = state_dict.pop(key).unflatten(0, (-1, 2))
            state_dict[f"{prefix}w1.{param_name}"] = param[:, 0].contiguous()
            state_dict[f"{prefix}w3.{param_name}"] = param[:, 1].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Merge logical w1 and w3 checkpoint keys into the physical w13."""
        for param_name in ("weight", "bias"):
            w1_key = f"{prefix}w1.{param_name}"
            w3_key = f"{prefix}w3.{param_name}"
            if w1_key in state_dict and w3_key in state_dict:
                state_dict[f"{prefix}w13.{param_name}"] = torch.stack(
                    [state_dict.pop(w1_key), state_dict.pop(w3_key)], dim=1
                ).flatten(0, 1)

        native_key = f"{prefix}w13"
        if native_key in state_dict:
            state_dict[f"{prefix}w13.weight"] = state_dict.pop(native_key).flatten(0, 1)


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
