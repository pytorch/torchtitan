# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    norm_spec,
    set_decoder_sharding_spec,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
)

if TYPE_CHECKING:
    from torchtitan.models.llama4.model import Llama4Model, Llama4TransformerBlock


def set_llama4_sharding_spec(
    config: "Llama4Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_spec`` on all Llama4 sub-configs.

    No-op when TP is not enabled.
    """

    set_decoder_sharding_spec(config, loss_parallel=loss_parallel, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_llama4_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_llama4_layer_sharding(
    layer_cfg: "Llama4TransformerBlock.Config", *, enable_sp: bool
) -> None:
    """Set sharding on one Llama4 transformer layer.

    Attention and norms are sharded on all blocks (MoE and non-MoE).
    Dense FFN is only sharded on non-MoE blocks — MoE FFN stays
    under apply_moe_ep_tp.
    """
    norm = norm_spec(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_spec = norm
    layer_cfg.ffn_norm.sharding_spec = norm
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)

    # Dense FFN (non-MoE layers only)
    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )
