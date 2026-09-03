# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for common vision encoder components."""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import set_gqa_inner_attention_local_map
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.common.vision_encoder import VisionTransformerBlock


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def _vision_state_placement(
    *,
    tp: spmd.PerMeshAxisSpmdType,
    include_cp_axis: bool = False,
) -> SpmdType:
    if include_cp_axis:
        return SpmdType({DP: spmd.R, CP: spmd.R, TP: tp})
    return SpmdType({DP: spmd.R, TP: tp})


def _vision_activation_placement(
    *,
    dp: spmd.PerMeshAxisSpmdType = spmd.V,
    tp: spmd.PerMeshAxisSpmdType = spmd.I,
    include_cp_axis: bool = False,
) -> SpmdType:
    if include_cp_axis:
        return SpmdType({DP: dp, CP: spmd.R, TP: tp})
    return SpmdType({DP: dp, TP: tp})


def multimodal_input_sharding(*, include_cp_axis: bool = False) -> dict[str, SpmdType]:
    """SPMD layouts for VLM vision inputs (folded into a model's input_sharding).

    The vision tensors are DP-local (``V@DP``) -- each DP rank owns its own
    images -- and TP-invariant (``I@TP``): the model consumes them inside
    ``multimodal_context`` (a DP-local mesh) and the vision encoder runs per-rank.
    Shared by every VLM decoder (Qwen3.5, Kimi K2.5, Muse Glimmer).
    """
    layout = _vision_activation_placement(include_cp_axis=include_cp_axis)
    return {
        "pixel_values": layout,
        "pixel_values_videos": layout,
        "grid_thw": layout,
        "grid_thw_videos": layout,
    }


def invariant_norm_config(*, include_cp_axis: bool = False) -> ShardingConfig:
    """Norm whose state and activations are invariant across TP ranks."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(
                tp=spmd.I, include_cp_axis=include_cp_axis
            ),
            "bias": _vision_state_placement(tp=spmd.I, include_cp_axis=include_cp_axis),
        },
        in_src_shardings={
            "input": _vision_activation_placement(include_cp_axis=include_cp_axis),
        },
        in_dst_shardings={
            "input": _vision_activation_placement(include_cp_axis=include_cp_axis),
        },
        out_src_shardings=_vision_activation_placement(include_cp_axis=include_cp_axis),
        out_dst_shardings=_vision_activation_placement(include_cp_axis=include_cp_axis),
    )


def vision_invariant_linear_config(*, include_cp_axis: bool = False) -> ShardingConfig:
    """Unsharded linear whose state and activations are invariant at TP."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(
                tp=spmd.I, include_cp_axis=include_cp_axis
            ),
            "bias": _vision_state_placement(tp=spmd.I, include_cp_axis=include_cp_axis),
        },
        in_src_shardings={
            "input": _vision_activation_placement(include_cp_axis=include_cp_axis),
        },
        in_dst_shardings={
            "input": _vision_activation_placement(include_cp_axis=include_cp_axis),
        },
        out_src_shardings=_vision_activation_placement(include_cp_axis=include_cp_axis),
        out_dst_shardings=_vision_activation_placement(include_cp_axis=include_cp_axis),
    )


def vision_colwise_config(
    *,
    input_tp: spmd.PerMeshAxisSpmdType = spmd.I,
    include_cp_axis: bool = False,
) -> ShardingConfig:
    """Colwise vision linear with a TP-replicated local matmul input."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(
                tp=spmd.S(0), include_cp_axis=include_cp_axis
            ),
            "bias": _vision_state_placement(
                tp=spmd.S(0), include_cp_axis=include_cp_axis
            ),
        },
        in_src_shardings={
            "input": _vision_activation_placement(
                tp=input_tp, include_cp_axis=include_cp_axis
            ),
        },
        in_dst_shardings={
            "input": _vision_activation_placement(
                tp=spmd.R, include_cp_axis=include_cp_axis
            ),
        },
        out_src_shardings=_vision_activation_placement(
            tp=spmd.S(-1), include_cp_axis=include_cp_axis
        ),
    )


def vision_scaled_bias_rowwise_config(
    *, include_cp_axis: bool = False
) -> ShardingConfig:
    """Scaled-bias rowwise vision linear returning a TP-invariant activation."""
    input_layout = _vision_activation_placement(
        tp=spmd.S(1), include_cp_axis=include_cp_axis
    )
    input_grad_layout = (
        SpmdType({DP: spmd.V, CP: spmd.P, TP: spmd.S(1)})
        if include_cp_axis
        else input_layout
    )
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(
                tp=spmd.S(1), include_cp_axis=include_cp_axis
            ),
            "bias": _vision_state_placement(tp=spmd.R, include_cp_axis=include_cp_axis),
        },
        in_src_shardings={
            "input": input_layout,
        },
        in_dst_shardings={
            "input": input_layout,
        },
        out_src_shardings=_vision_activation_placement(
            tp=spmd.P, include_cp_axis=include_cp_axis
        ),
        out_dst_shardings=_vision_activation_placement(include_cp_axis=include_cp_axis),
        local_map=LocalMapConfig(in_grad_placements=(input_grad_layout,)),
    )


def set_vision_transformer_block_sharding_config(
    block: "VisionTransformerBlock.Config",
    *,
    rope_cache_dp: spmd.PerMeshAxisSpmdType,
    include_cp_axis: bool = False,
) -> None:
    """Set TP sharding for the common vision transformer block."""
    block.norm1.sharding_config = invariant_norm_config(include_cp_axis=include_cp_axis)
    block.norm2.sharding_config = invariant_norm_config(include_cp_axis=include_cp_axis)

    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": _vision_activation_placement(include_cp_axis=include_cp_axis),
            "rope_cache": _vision_activation_placement(
                dp=rope_cache_dp, include_cp_axis=include_cp_axis
            ),
        },
        in_dst_shardings={
            "x": _vision_activation_placement(
                tp=spmd.R, include_cp_axis=include_cp_axis
            ),
            "rope_cache": _vision_activation_placement(
                dp=rope_cache_dp,
                tp=spmd.R,
                include_cp_axis=include_cp_axis,
            ),
        },
    )
    block.attn.wq.sharding_config = vision_colwise_config(
        input_tp=spmd.R, include_cp_axis=include_cp_axis
    )
    block.attn.wk.sharding_config = vision_colwise_config(
        input_tp=spmd.R, include_cp_axis=include_cp_axis
    )
    block.attn.wv.sharding_config = vision_colwise_config(
        input_tp=spmd.R, include_cp_axis=include_cp_axis
    )
    block.attn.proj.sharding_config = vision_scaled_bias_rowwise_config(
        include_cp_axis=include_cp_axis
    )
    if include_cp_axis:
        attention_layout = _vision_activation_placement(
            tp=spmd.S(1), include_cp_axis=True
        )
        attention_grad_layout = SpmdType({DP: spmd.V, CP: spmd.P, TP: spmd.S(1)})
        block.attn.inner_attention.sharding_config = ShardingConfig(
            in_src_shardings={
                "q_THK": attention_layout,
                "k_THK": attention_layout,
                "v_THV": attention_layout,
            },
            in_dst_shardings={
                "q_THK": attention_layout,
                "k_THK": attention_layout,
                "v_THV": attention_layout,
            },
            out_src_shardings=attention_layout,
            local_map=LocalMapConfig(
                in_grad_placements=(attention_grad_layout,) * 3,
            ),
        )
    else:
        set_gqa_inner_attention_local_map(block.attn.inner_attention)

    block.mlp.fc1.sharding_config = vision_colwise_config(
        include_cp_axis=include_cp_axis
    )
    block.mlp.fc2.sharding_config = vision_scaled_bias_rowwise_config(
        include_cp_axis=include_cp_axis
    )
