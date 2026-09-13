# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Expose fused parameters under the logical w1/w3 checkpoint keys."""
        for param_name in ("weight", "bias"):
            fused_key = f"{prefix}w13.{param_name}"
            if fused_key not in state_dict:
                continue
            gate_up = state_dict.pop(fused_key).unflatten(0, (-1, 2))
            state_dict[f"{prefix}w1.{param_name}"] = gate_up[:, 0].contiguous()
            state_dict[f"{prefix}w3.{param_name}"] = gate_up[:, 1].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Pack logical w1/w3 checkpoint entries into the fused parameter."""
        for param_name in ("weight", "bias"):
            gate_key = f"{prefix}w1.{param_name}"
            up_key = f"{prefix}w3.{param_name}"
            if gate_key not in state_dict or up_key not in state_dict:
                continue
            state_dict[f"{prefix}w13.{param_name}"] = torch.stack(
                [state_dict.pop(gate_key), state_dict.pop(up_key)], dim=1
            ).flatten(0, 1)

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        w13_sharding_config = self.w13._sharding_config
        if parallel_dims.tp_enabled and w13_sharding_config is not None:
            weight_layout = w13_sharding_config.state_shardings.get("weight")
            if weight_layout is not None:
                tp_type = spmd_type_for_axis(weight_layout, MeshAxisName.TP)
                if (
                    isinstance(tp_type, spmd.Shard)
                    and tp_type.dim in (0, -self.w13.weight.ndim)
                    and self.w2.in_features % parallel_dims.tp
                ):
                    raise ValueError(
                        "FeedForward hidden dimension "
                        f"({self.w2.in_features}) must be divisible by TP degree "
                        f"({parallel_dims.tp}) when w13 is sharded colwise. "
                        "Checking only the fused w13 output dimension would allow "
                        "TP to split an interleaved gate/up pair."
                    )
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
