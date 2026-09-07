# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SPMD layouts for Kimi K3 context parallelism."""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_activation_placement,
    dense_param_placement,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_config,
    set_vision_transformer_block_sharding_config,
    vision_colwise_config,
    vision_invariant_linear_config,
    vision_scaled_bias_rowwise_config,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from .model import KimiK3Model
    from .vision_encoder import KimiK3VisionEncoder

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP

_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


def _set_inner_kda_sharding(inner_kda: Module.Config) -> None:
    token_channels = dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    token_heads = attention_activation_placement()
    token_head_scalars = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.V},
        partition_spec=spmd.PartitionSpec((DP, CP), TP),
    )
    parameter = dense_param_placement(tp=spmd.R)
    parameter_gradient = SpmdType({DP: spmd.R, CP: spmd.P, TP: spmd.R})

    inner_kda.sharding_config = ShardingConfig(
        in_src_shardings={
            "query_TC": token_channels,
            "key_TC": token_channels,
            "value_TC": token_channels,
            "raw_gate_THK": token_heads,
            "raw_beta_TH": token_head_scalars,
            "conv_q_weight_C1W": parameter,
            "conv_k_weight_C1W": parameter,
            "conv_v_weight_C1W": parameter,
            "A_log_H": parameter,
            "dt_bias_HK": parameter,
        },
        in_dst_shardings={
            "query_TC": token_channels,
            "key_TC": token_channels,
            "value_TC": token_channels,
            "raw_gate_THK": token_heads,
            "raw_beta_TH": token_head_scalars,
            "conv_q_weight_C1W": parameter,
            "conv_k_weight_C1W": parameter,
            "conv_v_weight_C1W": parameter,
            "A_log_H": parameter,
            "dt_bias_HK": parameter,
        },
        out_src_shardings=token_heads,
        out_dst_shardings=token_heads,
        local_map=LocalMapConfig(
            in_grad_placements=(
                token_channels,
                token_channels,
                token_channels,
                token_heads,
                token_head_scalars,
                parameter_gradient,
                parameter_gradient,
                parameter_gradient,
                parameter_gradient,
                parameter_gradient,
            ),
        ),
    )


def set_kimi_k3_sharding_config(config: "KimiK3Model.Config") -> None:
    """Install the local kernel boundaries used by Kimi K3 CP."""
    if config.vision_encoder is not None:
        _set_vision_encoder_sharding(config.vision_encoder)
    for layer in config.layers:
        if layer.attention is not None:
            set_gqa_inner_attention_local_map(layer.attention.inner_attention)
        if layer.delta_attention is not None:
            _set_inner_kda_sharding(layer.delta_attention.inner_kda)
        if layer.moe is not None:
            set_moe_sharding_config(
                layer.moe,
                enable_ep=False,
                enable_sp=False,
                expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
            )


def _set_vision_encoder_sharding(config: "KimiK3VisionEncoder.Config") -> None:
    """Replicate the Kimi K3 vision encoder across the CP axis."""
    vision_state = SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I})
    vision_activation = SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.I})
    config.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": vision_state},
        out_src_shardings=vision_activation,
        out_dst_shardings=SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.R}),
    )
    config.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={"inv_freq": vision_state},
        out_src_shardings=SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I}),
    )
    config.patch_embed_proj.sharding_config = vision_invariant_linear_config(
        include_cp_axis=True
    )
    set_vision_transformer_block_sharding_config(
        config.block,
        rope_cache_dp=spmd.V,
        include_cp_axis=True,
    )
    config.final_norm.sharding_config = invariant_norm_config(include_cp_axis=True)
    config.projector.linear_1.sharding_config = vision_colwise_config(
        include_cp_axis=True
    )
    config.projector.linear_2.sharding_config = vision_scaled_bias_rowwise_config(
        include_cp_axis=True
    )
    config.projector.post_norm.sharding_config = invariant_norm_config(
        include_cp_axis=True
    )
