# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

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
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock


# Routed-expert weight layout for the shared ``GroupedExperts`` (w1/w2/w3).
# Maps each parameter name to its in/out-dim placement: ``Shard(1)`` for
# colwise (w1, w3), ``Shard(2)`` for rowwise (w2). Reused across qwen3,
# llama4, deepseek_v3 -- all three share ``GroupedExperts`` from
# ``models/common/moe.py``.
_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, Placement] = {
    "w1": Shard(1),
    "w2": Shard(2),
    "w3": Shard(1),
}


def set_qwen3_sharding_config(
    config: "Qwen3Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
    tp_enabled: bool,
    ep_enabled: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3 sub-configs.

    Dense sub-configs (attention, dense FFN, norms, embeddings, lm_head)
    are populated unconditionally; ``Module.parallelize`` filters disabled
    axes at runtime, so a TP=1 / EP=1 run is a no-op for those.

    MoE sub-configs are populated when ``tp_enabled or ep_enabled`` --
    ``set_moe_sharding_config`` itself branches dense vs sparse expert
    placements based on ``ep_enabled``.
    """

    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    for layer_cfg in config.layers:
        _set_qwen3_layer_sharding(
            layer_cfg,
            enable_sp=enable_sp,
            tp_enabled=tp_enabled,
            ep_enabled=ep_enabled,
        )


def _set_qwen3_layer_sharding(
    layer_cfg: "Qwen3TransformerBlock.Config",
    *,
    enable_sp: bool,
    tp_enabled: bool,
    ep_enabled: bool,
) -> None:
    """Set sharding on one Qwen3 transformer layer."""
    attention = layer_cfg.attention
    assert isinstance(attention, GQAttention.Config)

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    set_gqa_attention_sharding(attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # QK norms: shard on head dim (dim=2) — independent of SP.
    if attention.qk_norm is not None:
        attention.qk_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=Replicate())},
            in_src_shardings={"input": dense_activation_placement(tp=Shard(2))},
            in_dst_shardings={"input": dense_activation_placement(tp=Shard(2))},
            out_dst_shardings=dense_activation_placement(tp=Shard(2)),
        )

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
