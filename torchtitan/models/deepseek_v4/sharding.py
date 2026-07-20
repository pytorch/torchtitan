# Copyright (c) Meta Platforms, Inc. and affiliates.
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


def set_dsa_flex_attention_sharding(inner_attention_cfg) -> None:
    query_states = dense_activation_placement(tp=spmd.S(2))
    replicated_activation = dense_activation_placement(tp=spmd.R)

    input_shardings = {
        "query_states": query_states,
        "kv_states": replicated_activation,
        "attn_sink": _attn_sink_placement,
        "topk_idxs": replicated_activation,
    }
    grad_placements = [
        query_states,
        dense_activation_placement(tp=spmd.P),
        _attn_sink_placement,
        None,
    ]

    returns_lse = getattr(inner_attention_cfg, "return_lse", False)
    out_src_shardings = (query_states, query_states) if returns_lse else query_states
    out_dst_shardings = None if returns_lse else query_states

    inner_attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=out_src_shardings,
        out_dst_shardings=out_dst_shardings,
        local_map=LocalMapConfig(in_grad_placements=tuple(grad_placements)),
    )


def set_dsa_indexer_aux_loss_sharding(indexer_aux_loss_cfg) -> None:
    query_states = dense_activation_placement(tp=spmd.S(2))
    replicated_activation = dense_activation_placement(tp=spmd.R)
    partial_activation = dense_activation_placement(tp=spmd.P)

    input_shardings = {
        "carrier": query_states,
        "q": query_states,
        "kv_compress": replicated_activation,
        "compress_topk_idxs": replicated_activation,
        "index_score": replicated_activation,
        "attn_lse": query_states,
    }

    indexer_aux_loss_cfg.sharding_config = ShardingConfig(
        state_shardings={"_acc": _dense_param_rep},
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=query_states,
        out_dst_shardings=query_states,
        local_map=LocalMapConfig(
            in_grad_placements=(
                query_states,
                query_states,
                replicated_activation,
                None,
                partial_activation,
                query_states,
            )
        ),
    )


def set_deepseek_v4_attention_sharding(attention_cfg, *, enable_sp):
    at = attention_cfg
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )

    at.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": attn_x_layout,
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=spmd.R),
        },
    )

    set_dsa_flex_attention_sharding(at.inner_attention)

    # Sub-module configs are declared as fields on Attention.Config, so we
    # can set sharding_config directly (same pattern as deepseek_v3).
    at.wq_a.sharding_config = _replicate_weight
    at.q_norm.sharding_config = _replicate_weight
    at.wq_b.sharding_config = colwise_config()
    at.wkv.sharding_config = _replicate_weight
    at.kv_norm.sharding_config = _replicate_weight
    # wo_a is a Linear holding a grouped LoRA-A weight used via einsum (not a
    # standard matmul). Colwise sharding distributes the weight along dim-0.
    at.wo_a.sharding_config = colwise_config()
    at.wo_b.sharding_config = rowwise_config(output_sp=enable_sp)
    # attn_sink is a Linear holding a (n_heads, 1) weight used as a head-wise
    # vector in sparse attention, so shard it on the head dimension under TP.
    at.attn_sink.sharding_config = ShardingConfig(
        state_shardings={"weight": _attn_sink_placement},
    )
    at.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    at.single_rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )

    if at.compressor is not None:
        set_compressor_sharding(at.compressor)
    if at.compressor_128 is not None:
        set_compressor_sharding(at.compressor_128)
    if at.indexer is not None:
        set_indexer_sharding(at.indexer)
    if at.indexer_aux_loss is not None:
        set_dsa_indexer_aux_loss_sharding(at.indexer_aux_loss)


def set_compressor_sharding(compressor_cfg):
    # wkv, wgate, norm, and ape are all sub-module Config fields.
    compressor_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    compressor_cfg.single_rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    compressor_cfg.wkv.sharding_config = _replicate_weight
    compressor_cfg.wgate.sharding_config = _replicate_weight
    compressor_cfg.norm.sharding_config = _replicate_weight
    compressor_cfg.ape.sharding_config = _replicate_weight


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
    indexer_cfg.single_rope.sharding_config = ShardingConfig(
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
    hc_rep = ShardingConfig(
        state_shardings={
            n: _dense_param_rep
            for n in [
                "hc_attn_fn", "hc_ffn_fn",
                "hc_attn_base", "hc_ffn_base",
                "hc_attn_scale", "hc_ffn_scale",
            ]
        },
    )
    layer_cfg.sharding_config = hc_rep

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
        layer_cfg.moe.sharding_config.in_src_shardings["input_ids"] = (
            input_ids_src_placement
        )
        layer_cfg.moe.sharding_config.in_dst_shardings["input_ids"] = (
            input_ids_dst_placement
        )
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

    hc_rep = ShardingConfig(
        state_shardings={
            n: _dense_param_rep
            for n in ["hc_head_fn", "hc_head_base", "hc_head_scale"]
        },
    )
    model_sharding = config.sharding_config or ShardingConfig()
    model_sharding.state_shardings.update(hc_rep.state_shardings)
    config.sharding_config = model_sharding

    for layer_cfg in config.layers:
        set_deepseek_v4_layer_sharding(
            layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
        )
