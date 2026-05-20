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

from torchtitan.models.common.decoder_sharding import (
    norm_config,
    set_decoder_sharding_config as set_common_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)

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
    set_common_decoder_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_sp=enable_sp,
    )
    attn_x_placement = Shard(1) if enable_sp else Replicate()
    for layer_cfg in config.layers:
        layer_cfg.attention_norm.sharding_config = norm_config(enable_sp=enable_sp)
        set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
        if layer_cfg.attention.qk_norm is not None:
            layer_cfg.attention.qk_norm.sharding_config = norm_config(enable_sp=False)
        set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)
        layer_cfg.ffn_norm.sharding_config = norm_config(enable_sp=enable_sp)
        if layer_cfg.feed_forward is None:
            raise NotImplementedError(
                "Qwen3 TP sharding candidate only supports dense feed-forward layers."
            )
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )
