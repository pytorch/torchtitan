# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

_dense_param_rep = dense_param_placement(tp=spmd.R)
_act_shard0_tp_rep = dense_activation_placement(tp=spmd.R)
_attn_sink_placement = dense_param_placement(tp=spmd.S(0))
DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP
_replicated_layout = SpmdLayout({DP: spmd.R, CP: spmd.R, TP: spmd.R})


if TYPE_CHECKING:
    from torchtitan.models.deepseek_v4.model import (
        DeepSeekV4Model,
        DeepSeekV4TransformerBlock,
    )

_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}

_replicate_weight = ShardingConfig(
    state_shardings={"weight": _dense_param_rep},
)


def dense_token_ids_sequence_parallel_placement() -> SpmdLayout:
    return SpmdLayout(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=(DP, (CP, TP)),
    )


def hc_head_input_sequence_parallel_placement() -> SpmdLayout:
    return SpmdLayout(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=(DP, (CP, TP), None, None),
    )


def set_dsa_flex_attention_sharding(inner_attention_cfg) -> None:
    query_states = dense_activation_placement(tp=spmd.S(2))
    replicated_activation = dense_activation_placement(tp=spmd.R)
    partial_activation = dense_activation_placement(tp=spmd.P)

    input_shardings = {
        "q": query_states,
        "swa_k": replicated_activation,
    }
    grad_placements = [
        query_states,
        partial_activation,
    ]

    compress_ratio = getattr(inner_attention_cfg, "compress_ratio", 1)
    if compress_ratio == 4:
        input_shardings.update(
            {
                "cmp_k": replicated_activation,
                "idx_q": replicated_activation,
                "idx_k": replicated_activation,
                "idx_w": replicated_activation,
                "attn_sink": _attn_sink_placement,
            }
        )
        grad_placements.extend(
            [
                partial_activation,
                replicated_activation,
                replicated_activation,
                replicated_activation,
                _attn_sink_placement,
            ]
        )
    elif compress_ratio > 1:
        input_shardings.update(
            {
                "cmp_k": replicated_activation,
                "attn_sink": _attn_sink_placement,
            }
        )
        grad_placements.extend([partial_activation, _attn_sink_placement])
    else:
        input_shardings["attn_sink"] = _attn_sink_placement
        grad_placements.append(_attn_sink_placement)

    inner_attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=query_states,
        out_dst_shardings=query_states,
        local_map=LocalMapConfig(in_grad_placements=tuple(grad_placements)),
    )


def set_deepseek_v4_attention_sharding(attention_cfg, *, enable_sp):
    attention = attention_cfg
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )

    attention.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": attn_x_layout,
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=spmd.R),
        },
    )

    set_dsa_flex_attention_sharding(attention.inner_attention)

    # Sub-module configs are declared as fields on Attention.Config, so we
    # can set sharding_config directly (same pattern as deepseek_v3).
    attention.wq_a.sharding_config = _replicate_weight
    attention.q_norm.sharding_config = _replicate_weight
    attention.wq_b.sharding_config = colwise_config()
    attention.wkv.sharding_config = _replicate_weight
    attention.kv_norm.sharding_config = _replicate_weight
    # wo_a is a Linear holding a grouped LoRA-A weight used via einsum (not a
    # standard matmul). Colwise sharding distributes the weight along dim-0.
    attention.wo_a.sharding_config = colwise_config()
    attention.wo_b.sharding_config = rowwise_config(output_sp=enable_sp)
    # attn_sink is a Linear holding a (n_heads, 1) weight used as a head-wise
    # vector in sparse attention, so shard it on the head dimension under TP.
    attention.attn_sink.sharding_config = ShardingConfig(
        state_shardings={"weight": _attn_sink_placement},
    )
    attention.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )

    if attention.compressor is not None:
        set_compressor_sharding(attention.compressor)
    if attention.compressor_128 is not None:
        set_compressor_sharding(attention.compressor_128)
    if attention.indexer is not None:
        set_indexer_sharding(attention.indexer)


def set_compressor_sharding(compressor_cfg):
    compressor_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    compressor_cfg.wkv.sharding_config = _replicate_weight
    compressor_cfg.wgate.sharding_config = _replicate_weight
    compressor_cfg.norm.sharding_config = _replicate_weight
    compressor_cfg.sharding_config = ShardingConfig(
        state_shardings={"ape": _dense_param_rep},
    )


def set_indexer_sharding(indexer_cfg):
    replicated_activation = dense_activation_placement(tp=spmd.R)
    indexer_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": replicated_activation,
            "qr": replicated_activation,
        },
        in_dst_shardings={
            "x": replicated_activation,
            "qr": replicated_activation,
        },
    )
    indexer_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    indexer_cfg.wq_b.sharding_config = ShardingConfig(
        state_shardings={"weight": _dense_param_rep},
    )
    indexer_cfg.weights_proj.sharding_config = ShardingConfig(
        state_shardings={"weight": _dense_param_rep},
    )
    set_compressor_sharding(indexer_cfg.compressor)


def set_deepseek_v4_layer_sharding(
    layer_cfg: "DeepSeekV4TransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    hc_pre_sharding = ShardingConfig(
        state_shardings={
            "hc_fn": _dense_param_rep,
            "hc_base": _dense_param_rep,
            "hc_scale": _dense_param_rep,
        },
    )
    layer_cfg.hc_attn_pre.sharding_config = hc_pre_sharding
    layer_cfg.hc_ffn_pre.sharding_config = hc_pre_sharding

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )

    set_deepseek_v4_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)

    # Dense FFN (non-MoE layers only)
    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=attn_x_layout,
            enable_sp=enable_sp,
        )

    # MoE FFN (MoE-enabled layers only).
    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )
        input_ids_src_placement = dense_activation_placement(tp=spmd.R)
        input_ids_dst_placement = (
            dense_token_ids_sequence_parallel_placement()
            if enable_ep
            else dense_activation_placement(tp=spmd.R)
        )
        layer_cfg.moe.sharding_config.in_src_shardings[
            "input_ids"
        ] = input_ids_src_placement
        layer_cfg.moe.sharding_config.in_dst_shardings[
            "input_ids"
        ] = input_ids_dst_placement
        router_sharding = layer_cfg.moe.router.sharding_config or ShardingConfig()
        router_sharding.state_shardings["tid2eid"] = _replicated_layout
        layer_cfg.moe.router.sharding_config = router_sharding


def set_deepseek_v4_sharding_config(
    config: "DeepSeekV4Model.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    set_decoder_sharding_config(config, enable_sp=enable_sp)

    hc_head_input = (
        hc_head_input_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    hc_head_output = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    config.hc_head.sharding_config = ShardingConfig(
        state_shardings={
            "hc_fn": _dense_param_rep,
            "hc_base": _dense_param_rep,
            "hc_scale": _dense_param_rep,
        },
        in_src_shardings={"x": hc_head_input},
        out_src_shardings=hc_head_output,
    )

    for layer_cfg in config.layers:
        set_deepseek_v4_layer_sharding(
            layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
        )

    if config.mtp_layers is not None:
        replicated_activation = dense_activation_placement(tp=spmd.R)
        for mtp_cfg in config.mtp_layers:
            set_deepseek_v4_layer_sharding(
                mtp_cfg.block, enable_sp=enable_sp, enable_ep=enable_ep
            )
            mtp_cfg.e_proj.sharding_config = _replicate_weight
            mtp_cfg.h_proj.sharding_config = _replicate_weight
            mtp_cfg.enorm.sharding_config = _replicate_weight
            mtp_cfg.hnorm.sharding_config = _replicate_weight
            mtp_cfg.norm.sharding_config = _replicate_weight
            mtp_cfg.sharding_config = ShardingConfig(
                state_shardings={
                    "hc_head_fn": _dense_param_rep,
                    "hc_head_base": _dense_param_rep,
                    "hc_head_scale": _dense_param_rep,
                },
                in_src_shardings={
                    "x": replicated_activation,
                    "input_ids": dense_activation_placement(tp=spmd.R),
                },
                out_src_shardings=replicated_activation,
            )
