# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_gqa_inner_attention_local_spmd,
)
from torchtitan.models.common.moe_sharding import (
    expert_param_placement_sparse,
    set_moe_block_padding_mask_sharding,
    set_moe_sharding_config,
)
from torchtitan.models.gpt_oss.model import Attention
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.gpt_oss.model import GptOssModel, GptOssTransformerBlock


def set_gpt_oss_sharding_config(
    config: "GptOssModel.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all GPT-OSS sub-configs.

    Dense sub-configs (attention, norms) are populated unconditionally —
    ``Module.parallelize`` filters disabled axes at runtime.

    MoE sub-configs (router, routed experts) are populated when TP or
    EP is enabled.
    """

    set_decoder_sharding_config(config, enable_sp=enable_sp)
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
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )

    # Attention: input x gathered to Replicate.
    # sinks parameter is sharded across heads via state_shardings.
    attention.sharding_config = ShardingConfig(
        state_shardings={"sinks": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"x": attn_x_layout},
        out_src_shardings=attn_x_layout,
    )
    attention.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": dense_param_placement(tp=spmd.R)},
    )
    attention.qkv_linear.wqkv.sharding_config = colwise_config(
        input_layout=attn_x_layout
    )
    attention.wo.sharding_config = rowwise_config(output_layout=attn_x_layout)

    set_gqa_inner_attention_local_spmd(attention.inner_attention)

    # MoE FFN (all GPT-OSS blocks are MoE).
    if layer_cfg.moe is not None:
        set_moe_block_padding_mask_sharding(layer_cfg, enable_sp=enable_sp)
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
        )
        if enable_ep:
            layer_cfg.moe.routed_experts.inner_experts.sharding_config = ShardingConfig(
                state_shardings={
                    name: expert_param_placement_sparse()
                    for name in (
                        "mlp1_weight_EGD",
                        "mlp1_bias_EG",
                        "mlp2_weight_EDF",
                        "mlp2_bias_ED",
                    )
                }
            )
