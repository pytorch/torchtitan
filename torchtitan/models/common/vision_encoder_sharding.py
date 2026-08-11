# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for common vision encoder components."""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import set_gqa_inner_attention_local_map
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

if TYPE_CHECKING:
    from torchtitan.models.common.vision_encoder import VisionTransformerBlock


DP = MeshAxisName.DP
TP = MeshAxisName.TP


def invariant_norm_config() -> ShardingConfig:
    """Norm whose state and activations are invariant across TP ranks."""
    return ShardingConfig(
        state_shardings={
            "weight": SpmdLayout({DP: spmd.R, TP: spmd.I}),
            "bias": SpmdLayout({DP: spmd.R, TP: spmd.I}),
        },
        in_src_shardings={
            "input": SpmdLayout({DP: spmd.V, TP: spmd.I}),
        },
        in_dst_shardings={
            "input": SpmdLayout({DP: spmd.V, TP: spmd.I}),
        },
        out_src_shardings=SpmdLayout({DP: spmd.V, TP: spmd.I}),
        out_dst_shardings=SpmdLayout({DP: spmd.V, TP: spmd.I}),
    )


def vision_invariant_linear_config() -> ShardingConfig:
    """Unsharded linear whose state and activations are invariant at TP."""
    return ShardingConfig(
        state_shardings={
            "weight": SpmdLayout({DP: spmd.R, TP: spmd.I}),
            "bias": SpmdLayout({DP: spmd.R, TP: spmd.I}),
        },
        in_src_shardings={
            "input": SpmdLayout({DP: spmd.V, TP: spmd.I}),
        },
        in_dst_shardings={
            "input": SpmdLayout({DP: spmd.V, TP: spmd.I}),
        },
        out_src_shardings=SpmdLayout({DP: spmd.V, TP: spmd.I}),
        out_dst_shardings=SpmdLayout({DP: spmd.V, TP: spmd.I}),
    )


def vision_colwise_config(
    *, input_tp: spmd.PerMeshAxisSpmdType = spmd.I
) -> ShardingConfig:
    """Colwise vision linear with a TP-replicated local matmul input."""
    return ShardingConfig(
        state_shardings={
            "weight": SpmdLayout({DP: spmd.R, TP: spmd.S(0)}),
            "bias": SpmdLayout({DP: spmd.R, TP: spmd.S(0)}),
        },
        in_src_shardings={
            "input": SpmdLayout({DP: spmd.V, TP: input_tp}),
        },
        in_dst_shardings={
            "input": SpmdLayout({DP: spmd.V, TP: spmd.R}),
        },
        out_src_shardings=SpmdLayout({DP: spmd.V, TP: spmd.S(-1)}),
    )


def vision_scaled_bias_rowwise_config() -> ShardingConfig:
    """Scaled-bias rowwise vision linear returning a TP-invariant activation."""
    return ShardingConfig(
        state_shardings={
            "weight": SpmdLayout({DP: spmd.R, TP: spmd.S(1)}),
            "bias": SpmdLayout({DP: spmd.R, TP: spmd.R}),
        },
        in_src_shardings={
            "input": SpmdLayout({DP: spmd.V, TP: spmd.S(2)}),
        },
        in_dst_shardings={
            "input": SpmdLayout({DP: spmd.V, TP: spmd.S(2)}),
        },
        out_src_shardings=SpmdLayout({DP: spmd.V, TP: spmd.P}),
        out_dst_shardings=SpmdLayout({DP: spmd.V, TP: spmd.I}),
        local_map=LocalMapConfig(
            in_grad_placements=(SpmdLayout({DP: spmd.V, TP: spmd.S(2)}),)
        ),
    )


def set_vision_transformer_block_sharding_config(
    block: "VisionTransformerBlock.Config",
    *,
    rope_cache_dp: spmd.PerMeshAxisSpmdType,
) -> None:
    """Set TP sharding for the common vision transformer block."""
    block.norm1.sharding_config = invariant_norm_config()
    block.norm2.sharding_config = invariant_norm_config()

    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": SpmdLayout({DP: spmd.V, TP: spmd.I}),
            "rope_cache": SpmdLayout({DP: rope_cache_dp, TP: spmd.I}),
        },
        in_dst_shardings={
            "x": SpmdLayout({DP: spmd.V, TP: spmd.R}),
            "rope_cache": SpmdLayout({DP: rope_cache_dp, TP: spmd.R}),
        },
    )
    block.attn.wq.sharding_config = vision_colwise_config(input_tp=spmd.R)
    block.attn.wk.sharding_config = vision_colwise_config(input_tp=spmd.R)
    block.attn.wv.sharding_config = vision_colwise_config(input_tp=spmd.R)
    block.attn.proj.sharding_config = vision_scaled_bias_rowwise_config()
    set_gqa_inner_attention_local_map(block.attn.inner_attention)

    block.mlp.fc1.sharding_config = vision_colwise_config()
    block.mlp.fc2.sharding_config = vision_scaled_bias_rowwise_config()
