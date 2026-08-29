# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    set_decoder_sharding_config,
)
from torchtitan.models.deepseek_v3.sharding import _set_deepseek_v3_layer_sharding
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.deepseek_v3_2.model import (
        Attention,
        DeepSeekV3Model,
    )


def set_deepseek_v3_2_sharding_config(
    config: DeepSeekV3Model.Config,
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_deepseek_v3_layer_sharding(
            layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
        )
        _apply_v3_2_attention_sharding(
            layer_cfg.attention, enable_sp=enable_sp
        )


def _apply_v3_2_attention_sharding(
    attention: Attention.Config,
    *,
    enable_sp: bool,
) -> None:
    # w_uk / w_uv : TP Shard(0) on n_heads (analogous to colwise wkv_b, which
    # V3 sets unused; override with BatchedLinear-aware sharding)
    attention.w_uk.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )
    attention.w_uv.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )

    # Indexer sub-configs
    indexer_cfg = attention.indexer
    indexer_cfg.wq_b.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )
    indexer_cfg.weights_proj.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )
    indexer_cfg.wk.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
    )
    indexer_cfg.k_norm.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
    )
    indexer_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": dense_param_placement(tp=spmd.R)},
    )


    # Inner attention: MQA layout -- k/v are Replicate on TP (single head)
    inner_cfg = attention.inner_attention
    inner_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "q_nope": dense_activation_placement(tp=spmd.S(2)),
            "k_nope": dense_activation_placement(tp=spmd.R),
            "q_rope": dense_activation_placement(tp=spmd.S(2)),
            "k_rope": dense_activation_placement(tp=spmd.R),
            "idx_q": dense_activation_placement(tp=spmd.S(2)),
            "idx_k": dense_activation_placement(tp=spmd.R),
            "idx_w": dense_activation_placement(tp=spmd.S(2)),
        },
        in_dst_shardings={
            "q_nope": dense_activation_placement(tp=spmd.S(2)),
            "k_nope": dense_activation_placement(tp=spmd.R, cp=spmd.R),
            "q_rope": dense_activation_placement(tp=spmd.S(2)),
            "k_rope": dense_activation_placement(tp=spmd.R, cp=spmd.R),
            "idx_q": dense_activation_placement(tp=spmd.I),
            "idx_k": dense_activation_placement(tp=spmd.I, cp=spmd.R),
            "idx_w": dense_activation_placement(tp=spmd.I),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.S(2)),
        # spmd_types mode derives grad types from the forward dst types.
        local_map=LocalMapConfig(in_grad_placements=None),
    )

    indexer_loss_cfg = attention.inner_attention.indexer_loss
    indexer_loss_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "q": dense_activation_placement(tp=spmd.S(2)),
        },
        in_dst_shardings={
            "q": dense_activation_placement(tp=spmd.I),
        },
    )
