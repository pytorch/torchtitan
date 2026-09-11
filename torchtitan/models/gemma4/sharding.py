# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Gemma-4 Sharding Configuration

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_map,
    ShardingConfig,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config

if TYPE_CHECKING:
    from torchtitan.models.gemma4.model import Gemma4Model, Gemma4TransformerBlock


def set_gemma4_sharding_config(
    config: "Gemma4Model.Config",
    *,
    enable_sp: bool,
    enable_ep: bool = False,
) -> None:
    """Fill ``sharding_config`` on all Gemma-4 sub-configs.

    Specs are populated unconditionally — the mesh actually passed to
    ``Module.parallelize()`` at runtime determines which declarations
    apply. Declarations for mesh axes that aren't enabled (e.g. ``TP``
    placements under FSDP-only) are skipped at parallelize time.

    ``enable_sp`` controls SequenceParallel (decoupled from TP).
    """
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    if getattr(config, "ple", None) is not None:
        replicate_param = dense_param_placement(tp=spmd.R)
        config.ple.tok_embeddings_per_layer.sharding_config = ShardingConfig(
            state_shardings={"weight": replicate_param}
        )
        config.ple.per_layer_model_projection.sharding_config = ShardingConfig(
            state_shardings={"weight": replicate_param}
        )
        config.ple.per_layer_projection_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": replicate_param}
        )

    for layer_cfg in config.layers:
        _set_gemma4_layer_sharding(
            layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
        )


def _set_gemma4_layer_sharding(
    layer_cfg: "Gemma4TransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool = False,
) -> None:
    """Set sharding on one Gemma-4 transformer layer.

    ``enable_sp=True``  -> SP norms and Shard(0) activations around attention/FFN;
    ``attention.wo`` and ``feed_forward.w2`` reduce-scatter to Shard(0).
    ``enable_sp=False`` -> norms stay Replicate (no parallelism), activations
    stay Replicate; ``attention.wo`` and ``feed_forward.w2`` all-reduce to
    Replicate.
    """
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.post_attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    layer_cfg.post_ffn_norm.sharding_config = norm
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    layer_cfg.attention.sharding_config = ShardingConfig(
        in_src_shardings={
            "x_TD": attn_x_layout,
        },
        in_dst_shardings={
            "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
        },
    )
    if layer_cfg.attention.rope is not None:
        layer_cfg.attention.rope.sharding_config = ShardingConfig(
            state_shardings={"cache": dense_param_placement(tp=spmd.R)},
        )

    qkv = layer_cfg.attention.qkv_linear
    if hasattr(qkv, "wq") and qkv.wq is not None:
        qkv.wq.sharding_config = colwise_config()
    if hasattr(qkv, "wk") and qkv.wk is not None:
        qkv.wk.sharding_config = colwise_config()
    if hasattr(qkv, "wv") and qkv.wv is not None:
        qkv.wv.sharding_config = colwise_config()
    layer_cfg.attention.wo.sharding_config = rowwise_config(output_sp=enable_sp)
    set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

    # Shard qk_norm if present
    qk_norm = getattr(layer_cfg.attention, "qk_norm", None)
    if qk_norm is not None:
        head_layout = attention_activation_placement()
        qk_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)},
            in_src_shardings={"input": head_layout},
            in_dst_shardings={"input": head_layout},
            out_src_shardings=head_layout,
            out_dst_shardings=head_layout,
        )

    # Shard v_norm if present
    if (
        hasattr(layer_cfg.attention, "v_norm")
        and layer_cfg.attention.v_norm is not None
    ):
        head_layout = attention_activation_placement()
        layer_cfg.attention.v_norm.sharding_config = ShardingConfig(
            in_src_shardings={"input": head_layout},
            in_dst_shardings={"input": head_layout},
            out_src_shardings=head_layout,
            out_dst_shardings=head_layout,
        )

    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=attn_x_layout,
            enable_sp=enable_sp,
        )
    elif getattr(layer_cfg, "moe", None) is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            attn_x_layout=attn_x_layout,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        )

    if getattr(layer_cfg, "ple", None) is not None:
        replicate_param = dense_param_placement(tp=spmd.R)
        layer_cfg.ple.per_layer_input_gate.sharding_config = ShardingConfig(
            state_shardings={"weight": replicate_param}
        )
        layer_cfg.ple.per_layer_projection.sharding_config = ShardingConfig(
            state_shardings={"weight": replicate_param}
        )
        layer_cfg.ple.post_per_layer_input_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": replicate_param}
        )
