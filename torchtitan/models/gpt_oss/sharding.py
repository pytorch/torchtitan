# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_gqa_inner_attention_local_map,
    set_qkv_linear_sharding,
)
from torchtitan.models.gpt_oss.model import Attention
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.gpt_oss.model import GptOssModel, GptOssTransformerBlock


def set_gpt_oss_sharding_config(
    config: "GptOssModel.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on all GPT-OSS sub-configs.

    No-op when TP is not enabled.
    """

    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    for layer_cfg in config.layers:
        _set_gpt_oss_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_gpt_oss_layer_sharding(
    layer_cfg: "GptOssTransformerBlock.Config", *, enable_sp: bool
) -> None:
    """Set sharding on one GPT-OSS transformer layer.

    All GPT-OSS blocks are MoE — only attention/norms are sharded here.
    MoE FFN stays under apply_moe_ep_tp.
    """
    attention = layer_cfg.attention
    assert isinstance(attention, Attention.Config)

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    # Attention: input x gathered to Replicate, freqs_cis always Replicate.
    # sinks parameter is sharded across heads via state_shardings.
    attention.sharding_config = ShardingConfig(
        state_shardings={"sinks": dense_param_placement(tp=Shard(0))},
        in_src_shardings={
            "x": dense_activation_placement(tp=attn_x_placement),
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate()),
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
    )
    set_qkv_linear_sharding(attention.qkv_linear)
    attention.wo.sharding_config = rowwise_config(output_sp=enable_sp)

    # GPT-OSS flash attention always returns (output, lse).
    set_gqa_inner_attention_local_map(attention.inner_attention, return_lse=True)
