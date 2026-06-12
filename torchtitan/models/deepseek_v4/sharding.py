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


def set_deepseek_v4_attention_sharding(attention_cfg, *, enable_sp):
    at = attention_cfg
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.R)
    )

    at.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": attn_x_layout,
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=spmd.R),
        },
    )

    if at.li_compute is not None:
        at.li_compute.sharding_config = ShardingConfig(
            in_dst_shardings={
                "q_indexer": dense_activation_placement(tp=spmd.R),
                "k_indexer": dense_activation_placement(tp=spmd.R),
                "weights": dense_activation_placement(tp=spmd.R),
            },
            out_src_shardings=(
                dense_activation_placement(tp=spmd.R),
                dense_activation_placement(tp=spmd.R),
            ),
            local_map=LocalMapConfig(
                in_grad_placements=(
                    dense_activation_placement(tp=spmd.R),
                    dense_activation_placement(tp=spmd.R),
                    dense_activation_placement(tp=spmd.R),
                ),
            ),
        )
    if at.compress_ratio == 1:
        at.sparse_attn.sharding_config = ShardingConfig(
            in_src_shardings={
                "query_states": dense_activation_placement(tp=spmd.S(2)),
                "kv_states": dense_activation_placement(tp=spmd.R),
                "attn_sink": _attn_sink_placement,
            },
            in_dst_shardings={
                "query_states": dense_activation_placement(tp=spmd.S(2)),
                "kv_states": dense_activation_placement(tp=spmd.R),
                "attn_sink": _attn_sink_placement,
            },
            out_src_shardings=dense_activation_placement(tp=spmd.S(2)),
            out_dst_shardings=dense_activation_placement(tp=spmd.S(2)),
            local_map=LocalMapConfig(
                in_grad_placements=(
                    dense_activation_placement(tp=spmd.S(2)),
                    dense_activation_placement(tp=spmd.P),
                    _attn_sink_placement,
                ),
            ),
        )
    elif at.compress_ratio == 4:
        at.sparse_attn.sharding_config = ShardingConfig(
            in_src_shardings={
                "query_states": dense_activation_placement(tp=spmd.S(2)),
                "kv_states": dense_activation_placement(tp=spmd.R),
                "attn_sink": _attn_sink_placement,
                "kv_compress": dense_activation_placement(tp=spmd.R),
                "compress_topk_idxs": dense_activation_placement(tp=spmd.R),
            },
            in_dst_shardings={
                "query_states": dense_activation_placement(tp=spmd.S(2)),
                "kv_states": dense_activation_placement(tp=spmd.R),
                "attn_sink": _attn_sink_placement,
                "kv_compress": dense_activation_placement(tp=spmd.R),
                "compress_topk_idxs": dense_activation_placement(tp=spmd.R),
            },
            out_src_shardings=dense_activation_placement(tp=spmd.S(2)),
            out_dst_shardings=dense_activation_placement(tp=spmd.S(2)),
            local_map=LocalMapConfig(
                in_grad_placements=(
                    dense_activation_placement(tp=spmd.S(2)),
                    dense_activation_placement(tp=spmd.P),
                    _attn_sink_placement,
                    dense_activation_placement(tp=spmd.P),
                    dense_activation_placement(tp=spmd.R),
                ),
            ),
        )
    else:
        at.sparse_attn.sharding_config = ShardingConfig(
            in_src_shardings={
                "query_states": dense_activation_placement(tp=spmd.S(2)),
                "kv_states": dense_activation_placement(tp=spmd.R),
                "attn_sink": _attn_sink_placement,
                "kv_compress": dense_activation_placement(tp=spmd.R),
            },
            in_dst_shardings={
                "query_states": dense_activation_placement(tp=spmd.S(2)),
                "kv_states": dense_activation_placement(tp=spmd.R),
                "attn_sink": _attn_sink_placement,
                "kv_compress": dense_activation_placement(tp=spmd.R),
            },
            out_src_shardings=dense_activation_placement(tp=spmd.S(2)),
            out_dst_shardings=dense_activation_placement(tp=spmd.S(2)),
            local_map=LocalMapConfig(
                in_grad_placements=(
                    dense_activation_placement(tp=spmd.S(2)),
                    dense_activation_placement(tp=spmd.P),
                    _attn_sink_placement,
                    dense_activation_placement(tp=spmd.P),
                ),
            ),
        )


    # Sub-module configs are declared as fields on Attention.Config, so we
    # can set sharding_config directly (same pattern as deepseek_v3).
    at.wq_a.sharding_config = _replicate_weight
    at.q_norm.sharding_config = norm_config(enable_sp=False)
    at.wq_b.sharding_config = colwise_config()
    at.wkv.sharding_config = _replicate_weight
    at.kv_norm.sharding_config = norm_config(enable_sp=False)
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

    if at.compressor is not None:
        set_compressor_sharding(at.compressor)
    if at.compressor_128 is not None:
        set_compressor_sharding(at.compressor_128)
    if at.indexer is not None:
        set_indexer_sharding(at.indexer)


def set_compressor_sharding(compressor_cfg):
    # wkv, wgate, norm, and ape are all sub-module Config fields.
    compressor_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    compressor_cfg.wkv.sharding_config = _replicate_weight
    compressor_cfg.wgate.sharding_config = _replicate_weight
    compressor_cfg.norm.sharding_config = norm_config(enable_sp=False)
    compressor_cfg.ape.sharding_config = _replicate_weight


def set_indexer_sharding(indexer_cfg):
    indexer_cfg.sharding_config = ShardingConfig(
      state_shardings={"hadamard_mat": _dense_param_rep},
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

    layer_cfg.attention_norm.sharding_config = norm_config(enable_sp=False)
    layer_cfg.ffn_norm.sharding_config = norm_config(enable_sp=False)
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.R)
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


def set_deepseek_v4_sharding_config(
    config: "DeepSeekV4Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )

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
        set_deepseek_v4_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)
