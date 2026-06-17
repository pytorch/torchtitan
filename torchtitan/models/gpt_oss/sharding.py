# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_gqa_inner_attention_local_map,
    set_qkv_linear_sharding,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.gpt_oss.model import Attention
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.gpt_oss.model import GptOssModel, GptOssTransformerBlock


# Routed-expert layout for ``GptOssGroupedExperts`` (mlp1/mlp2 fused
# weights + biases): mlp1 colwise, mlp2 rowwise, mlp2_bias replicated.
_GPT_OSS_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "mlp1_weight_EGD": spmd.S(1),
    "mlp1_bias_EG": spmd.S(1),
    "mlp2_weight_EDF": spmd.S(2),
    "mlp2_bias_ED": spmd.R,
}


def set_gpt_oss_sharding_config(
    config: "GptOssModel.Config",
    *,
    tp_gather_logits: bool,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all GPT-OSS sub-configs.

    Dense sub-configs (attention, norms) are populated unconditionally —
    ``Module.parallelize`` filters disabled axes at runtime.

    MoE sub-configs (router, routed experts) are populated when TP or
    EP is enabled.
    """

    set_decoder_sharding_config(
        config, tp_gather_logits=tp_gather_logits, enable_sp=enable_sp
    )
    for layer_cfg in config.layers:
        _set_gpt_oss_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_gpt_oss_layer_sharding(
    layer_cfg: "GptOssTransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Set sharding on one GPT-OSS transformer layer.

    Attention and norms are sharded on all blocks. MoE FFN is routed
    through ``set_moe_sharding_config``.
    """
    attention = layer_cfg.attention
    assert isinstance(attention, Attention.Config)

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.R)
    )

    # Attention: input x gathered to Replicate.
    # sinks parameter is sharded across heads via state_shardings.
    attention.sharding_config = ShardingConfig(
        state_shardings={"sinks": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={
            "x": attn_x_layout,
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=spmd.R),
        },
    )
    attention.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": dense_param_placement(tp=spmd.R)},
    )
    set_qkv_linear_sharding(attention.qkv_linear)
    attention.wo.sharding_config = rowwise_config(output_sp=enable_sp)

    # GPT-OSS flash attention always returns (output, lse).
    set_gqa_inner_attention_local_map(attention.inner_attention, return_lse=True)

    # MoE FFN (all GPT-OSS blocks are MoE).
    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout=_GPT_OSS_EXPERTS_PARAM_LAYOUT,
        )
