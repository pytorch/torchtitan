# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config

if TYPE_CHECKING:
    from torchtitan.models.llama4.model import Llama4Model, Llama4TransformerBlock


def set_llama4_sharding_config(
    config: "Llama4Model.Config",
    *,
    loss_parallel: bool,
    enable_tp: bool,
    enable_cp: bool,
    enable_sp: bool,
    enable_ep: bool,
    chunked_loss: bool,
) -> None:
    """Fill ``sharding_config`` on all Llama4 sub-configs."""

    set_decoder_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_sp=enable_sp,
        chunked_loss=chunked_loss,
    )
    for layer_cfg in config.layers:
        _set_llama4_layer_sharding(
            layer_cfg,
            enable_tp=enable_tp,
            enable_cp=enable_cp,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        )


def _set_llama4_layer_sharding(
    layer_cfg: "Llama4TransformerBlock.Config",
    *,
    enable_tp: bool,
    enable_cp: bool,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Set sharding on one Llama4 transformer layer.

    Attention and norms are sharded on all blocks (MoE and non-MoE).
    Dense FFN is sharded on non-MoE blocks. MoE blocks get full
    sharding configs for the wrapper, router, shared experts, and
    routed experts.
    """
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_placement = spmd.S(1) if enable_sp else spmd.I

    set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )

    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_tp=enable_tp,
            enable_cp=enable_cp,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
        )
