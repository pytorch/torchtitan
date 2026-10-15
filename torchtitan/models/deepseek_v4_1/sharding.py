# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The complete V4.1 parameter/activation placement policy.

V4.1-specific placements for attention, compressor, indexer, and mHC.
MoE and decoder-level placements (embedding, norm, lm_head) are already
set by ``make_moe_config`` / ``make_router_config`` in config_utils and
the flavors builder; this module only adds the V4.1-specific overrides.
"""

import spmd_types as spmd
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse
from torchtitan.models.base.sharding import ShardingConfig

DP, CP, TP = MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP

from .indexer import REUSE

_dense_param_rep = dense_param_placement(tp=spmd.R)
_attn_sink_placement = dense_param_placement(tp=spmd.S(0))

_replicate_weight = ShardingConfig(
    state_shardings={"weight": _dense_param_rep},
)


def _engram_hidden_placement(*, enable_sp: bool):
    if not enable_sp:
        return dense_activation_placement(tp=spmd.I, cp=spmd.R)
    return dense_sequence_parallel_placement()


def set_engram_sharding_config(engram_cfg, *, enable_ep: bool, enable_sp: bool) -> None:
    """Declare the dense mHC boundary and the sparse table storage family."""
    dense_param_rep = dense_param_placement(tp=spmd.R)
    table_state = {
        "weight": expert_param_placement_sparse() if enable_ep else dense_param_rep,
        "token_id_map": dense_param_rep,
        "head_vocab_sizes": dense_param_rep,
        "offsets": dense_param_rep,
        "hash_multipliers": dense_param_rep,
        "compressed_pad_id": dense_param_rep,
    }
    engram_cfg.table.sharding_config = ShardingConfig(state_shardings=table_state)

    engram_cfg.gate.sharding_config = ShardingConfig(
        state_shardings={
            "wkv": dense_param_rep,
            "q_weight": dense_param_rep,
            "k_weight": dense_param_rep,
        }
    )
    hidden = _engram_hidden_placement(enable_sp=enable_sp)
    token_src = dense_activation_placement(tp=spmd.R, cp=spmd.R)
    engram_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "hidden_states_BLMD": hidden,
            "input_ids_BL": token_src,
            "positions_BL": token_src,
            "image_mask": token_src,
            "token_history": token_src,
        },
        in_dst_shardings={
            "hidden_states_BLMD": hidden,
            "input_ids_BL": token_src,
            "positions_BL": token_src,
            "image_mask": token_src,
            "token_history": token_src,
        },
        out_src_shardings=hidden,
        out_dst_shardings=hidden,
    )


def dense_token_ids_sequence_parallel_placement():
    return dense_sequence_parallel_placement()


def set_inner_attention_sharding(inner_cfg) -> None:
    """Placement for the CSA2 sparse attention core.

    Under context parallelism the KV streams are the whole row -- gathered
    declaratively at this boundary -- while ``q``, ``topk_indices`` and the
    sink keep the placements they have at every degree.
    """
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.R)
    row = dense_activation_placement(tp=spmd.R, cp=spmd.R)

    placements = {
        "q": dense_activation_placement(tp=spmd.S(2), cp=spmd.R),
        "swa_k": replicated_activation,
        "cmp_k": replicated_activation,
        "topk_indices": replicated_activation,
        "attn_sink": _attn_sink_placement,
    }
    consumed = {**placements, "swa_k": row, "cmp_k": row}
    inner_cfg.sharding_config = ShardingConfig(
        in_src_shardings=placements,
        in_dst_shardings=consumed,
        out_src_shardings=placements["q"],
        out_dst_shardings=placements["q"],
    )


def set_compressor_sharding(compressor_cfg):
    compressor_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )
    compressor_cfg.wkv.sharding_config = _replicate_weight
    if compressor_cfg.wgate is not None:
        compressor_cfg.wgate.sharding_config = _replicate_weight
    compressor_cfg.norm.sharding_config = _replicate_weight


def set_indexer_sharding(indexer_cfg):
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.R)
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
    indexer_cfg.selector.sharding_config = ShardingConfig(
        in_src_shardings={
            "idx_q_BLHiDi": replicated_activation,
            "idx_k_BNDi": replicated_activation,
            "weights_BLHi": replicated_activation,
        },
        in_dst_shardings={
            "idx_q_BLHiDi": replicated_activation,
            "idx_k_BNDi": dense_activation_placement(tp=spmd.R, cp=spmd.R),
            "weights_BLHi": replicated_activation,
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
    if indexer_cfg.wk is not None:
        indexer_cfg.wk.sharding_config = _replicate_weight
    if indexer_cfg.k_norm is not None:
        indexer_cfg.k_norm.sharding_config = _replicate_weight


def set_v41_attention_sharding(attention_cfg, *, enable_sp: bool):
    at = attention_cfg
    attn_x_layout = dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.I, cp=spmd.R)

    at.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": attn_x_layout,
        },
        state_shardings={"attn_sink": _attn_sink_placement},
    )

    set_inner_attention_sharding(at.inner_attention)

    at.wq_a.sharding_config = _replicate_weight
    at.q_norm.sharding_config = _replicate_weight
    at.wq_b.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )
    at.wkv.sharding_config = _replicate_weight
    at.kv_norm.sharding_config = _replicate_weight
    at.wo_a.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )
    at.wo_b.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(1))},
    )
    at.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": _dense_param_rep},
    )

    if at.compressor.is_source:
        set_compressor_sharding(at.compressor)
    if at.indexer.mode is not REUSE:
        set_indexer_sharding(at.indexer)


def set_v41_layer_sharding(layer_cfg, *, enable_sp: bool, enable_ep: bool) -> None:
    hc_pre_rep = ShardingConfig(
        state_shardings={
            "hc_fn": _dense_param_rep,
            "hc_base": _dense_param_rep,
            "hc_scale": _dense_param_rep,
        },
    )
    layer_cfg.hc_attn_pre.sharding_config = hc_pre_rep
    layer_cfg.hc_ffn_pre.sharding_config = hc_pre_rep

    norm = norm_sharding_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    set_v41_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)


def set_deepseek_v4_1_sharding_config(
    config,
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Assign the V4.1-specific placement policy.

    Decoder-level placements (embedding, final norm, lm_head) and MoE
    placements are already set by the flavors builder via
    ``make_moe_config`` / ``make_router_config``. This function only
    adds V4.1-specific overrides: attention, compressor, indexer, mHC,
    and engram (when present).
    """
    config.tok_embeddings.sharding_config = token_embedding_sharding_config(enable_sp=enable_sp)
    config.norm.sharding_config = pre_lm_head_norm_sharding_config(enable_sp=enable_sp)
    config.lm_head.sharding_config = lm_head_sharding_config()

    for layer in config.layers:
        if getattr(layer, "engram", None) is not None:
            set_engram_sharding_config(layer.engram, enable_ep=enable_ep, enable_sp=enable_sp)

    for layer_cfg in config.layers:
        set_v41_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)
