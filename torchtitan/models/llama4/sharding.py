# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

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


# Routed-expert layout for the shared ``GroupedExperts`` (w1/w2/w3).
_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, Placement] = {
    "w1": Shard(1),
    "w2": Shard(2),
    "w3": Shard(1),
}


def set_llama4_sharding_config(
    config: "Llama4Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
    tp_enabled: bool,
    ep_enabled: bool,
) -> None:
    """Fill ``sharding_config`` on all Llama4 sub-configs.

    Dense sub-configs are populated unconditionally (``Module.parallelize``
    filters disabled axes at runtime). MoE sub-configs are populated when
    ``tp_enabled or ep_enabled`` -- ``set_moe_sharding_config`` itself
    branches dense vs sparse expert placements based on ``ep_enabled``.
    """

    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    for layer_cfg in config.layers:
        _set_llama4_layer_sharding(
            layer_cfg,
            enable_sp=enable_sp,
            tp_enabled=tp_enabled,
            ep_enabled=ep_enabled,
        )


def _set_llama4_layer_sharding(
    layer_cfg: "Llama4TransformerBlock.Config",
    *,
    enable_sp: bool,
    tp_enabled: bool,
    ep_enabled: bool,
) -> None:
    """Set sharding on one Llama4 transformer layer.

    Attention and norms are sharded on all blocks (MoE and non-MoE).
    Dense FFN is only sharded on non-MoE blocks; MoE FFN is routed
    through ``set_moe_sharding_config``.
    """
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

    # Dense FFN (non-MoE layers only)
    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )

    # MoE FFN (MoE-enabled layers only).
    if layer_cfg.moe is not None and (tp_enabled or ep_enabled):
        set_moe_sharding_config(
            layer_cfg.moe,
            tp_enabled=tp_enabled,
            ep_enabled=ep_enabled,
            enable_sp=enable_sp,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )
