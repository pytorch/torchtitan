# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Kimi K3. Same pattern as ``qwen3_5/sharding.py``.

Declarations only: functions here set ``ShardingConfig`` on sub-configs of an
already-built config tree, and ``model.parallelize()`` applies them through the
Module protocol. Nothing here touches a mesh or a device.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import set_gqa_inner_attention_local_map
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
    from torchtitan.models.kimi_k3.model import KimiK3Model
    from torchtitan.models.kimi_k3.vision_encoder import KimiK3VisionEncoder


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def _set_inner_kda_sharding(inner_kda: Module.Config) -> None:
    """Set the local boundary around Attention Gym's KDA kernels."""
    token_channels = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.I},
        partition_spec=spmd.PartitionSpec((DP, CP), None),
    )
    token_head_vectors = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.V},
        partition_spec=spmd.PartitionSpec((DP, CP), TP, None),
    )
    token_head_scalars = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.V},
        partition_spec=spmd.PartitionSpec((DP, CP), TP),
    )
    parameter = SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.R})
    parameter_gradient = SpmdType({DP: spmd.R, CP: spmd.P, TP: spmd.R})

    inner_kda.sharding_config = ShardingConfig(
        in_src_shardings={
            "query_TC": token_channels,
            "key_TC": token_channels,
            "value_TC": token_channels,
            "raw_gate_THK": token_head_vectors,
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
            "raw_gate_THK": token_head_vectors,
            "raw_beta_TH": token_head_scalars,
            "conv_q_weight_C1W": parameter,
            "conv_k_weight_C1W": parameter,
            "conv_v_weight_C1W": parameter,
            "A_log_H": parameter,
            "dt_bias_HK": parameter,
        },
        out_src_shardings=token_head_vectors,
        out_dst_shardings=token_head_vectors,
        local_map=LocalMapConfig(
            in_grad_placements=(
                token_channels,
                token_channels,
                token_channels,
                token_head_vectors,
                token_head_scalars,
                parameter_gradient,
                parameter_gradient,
                parameter_gradient,
                parameter_gradient,
                parameter_gradient,
            ),
        ),
    )


def set_kimi_k3_sharding_config(
    config: "KimiK3Model.Config", *, enable_ep: bool, enable_sp: bool = False
) -> None:
    """Declare SPMD layouts for KDA, MLA, the vision encoder, and MoE.

    KDA and MLA activations shard tokens across CP ranks. Vision buffers
    replicate across DP and CP ranks. Routed experts shard on the expert
    axis; ``set_moe_sharding_config`` declares that layout, and its input
    boundary lifts the plain incoming activations itself.
    """
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
                enable_ep=enable_ep,
                enable_sp=enable_sp,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )


def _set_vision_encoder_sharding(ve_cfg: "KimiK3VisionEncoder.Config") -> None:
    """Replicate MoonViT3d parameters and activations across the CP axis."""
    vision_state = SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I})
    vision_activation = SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.I})
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": vision_state},
        out_src_shardings=vision_activation,
        out_dst_shardings=SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.R}),
    )
    ve_cfg.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={"inv_freq": vision_state},
        out_src_shardings=SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I}),
    )
    ve_cfg.patch_embed_proj.sharding_config = vision_invariant_linear_config(
        include_cp_axis=True
    )
    set_vision_transformer_block_sharding_config(
        ve_cfg.block,
        rope_cache_dp=spmd.V,
        include_cp_axis=True,
    )
    ve_cfg.final_norm.sharding_config = invariant_norm_config(include_cp_axis=True)

    proj = ve_cfg.projector
    proj.linear_1.sharding_config = vision_colwise_config(include_cp_axis=True)
    proj.linear_2.sharding_config = vision_scaled_bias_rowwise_config(
        include_cp_axis=True
    )
    proj.post_norm.sharding_config = invariant_norm_config(include_cp_axis=True)
