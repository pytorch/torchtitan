# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 sharding scaffold.

This file is intentionally strategy-free. Autoresearch is expected to replace
``set_qwen3_sharding_config`` with train-command-specific tensor placement
contracts for the target model flavor and machine or cluster.
"""

from typing import TYPE_CHECKING

from torch.distributed.tensor import Replicate, Shard

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.qwen3.model import Qwen3Model


def set_qwen3_sharding_config(
    config: "Qwen3Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Generated machine-specific Qwen3 sharding configuration hook.

    A valid implementation should attach ``sharding_config`` objects to the
    relevant Qwen3 configs before ``parallelize_qwen3`` calls
    ``model.parallelize(...)``. It may make narrow assumptions about the exact
    train command, model flavor, mesh shape, and hardware being optimized.
    """
    set_decoder_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_sp=enable_sp,
    )

    attention_output_tp = Shard(1) if enable_sp else Replicate()
    qk_norm_placement = dense_activation_placement(tp=Shard(2))
    qk_norm_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Replicate())},
        in_src_shardings={"input": qk_norm_placement},
        in_dst_shardings={"input": qk_norm_placement},
        out_dst_shardings=qk_norm_placement,
    )

    for layer_cfg in config.layers:
        layer_cfg.attention_norm.sharding_config = norm_config(enable_sp=enable_sp)
        layer_cfg.ffn_norm.sharding_config = norm_config(enable_sp=enable_sp)

        attention_cfg = layer_cfg.attention
        if not isinstance(attention_cfg, GQAttention.Config):
            raise TypeError(
                "Qwen3 dense TP sharding expects GQAttention layers, got "
                f"{type(attention_cfg).__name__}."
            )

        set_gqa_attention_sharding(attention_cfg, enable_sp=enable_sp)
        if attention_cfg.qk_norm is not None:
            attention_cfg.qk_norm.sharding_config = qk_norm_config
        set_gqa_inner_attention_local_map(attention_cfg.inner_attention)

        if layer_cfg.moe is not None:
            raise NotImplementedError(
                "Qwen3 autoresearch TP sharding only supports dense FFN layers."
            )
        assert layer_cfg.feed_forward is not None
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attention_output_tp,
            enable_sp=enable_sp,
        )
