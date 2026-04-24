# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.deepseek_v3.model import Attention
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.deepseek_v3.model import (
        DeepSeekV3Model,
        DeepSeekV3TransformerBlock,
    )


def set_deepseek_v3_sharding_config(
    config: "DeepSeekV3Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on all DeepSeek V3 sub-configs.

    No-op when TP is not enabled.
    """

    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    for layer_cfg in config.layers:
        _set_deepseek_v3_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_deepseek_v3_layer_sharding(
    layer_cfg: "DeepSeekV3TransformerBlock.Config", *, enable_sp: bool
) -> None:
    """Set sharding on one DeepSeek V3 transformer layer.

    MLA attention: low-rank projections (wkv_a, wq_a, kv_norm, q_norm)
    stay replicated. Up-projections (wkv_b, wq_b, wq) are colwise.
    """
    attention = layer_cfg.attention
    assert isinstance(attention, Attention.Config)

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    # MLA attention input: x is gathered to Replicate; freqs_cis always Replicate.
    attention.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": dense_activation_placement(tp=attn_x_placement),
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate()),
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
    )
    # Low-rank projections and norms keep Replicate weights on TP. We still
    # distribute them (Replicate DTensor) so DTensor activations flow through
    # without mixing plain Tensor + DTensor in the matmul.
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Replicate())},
    )
    attention.wkv_a.sharding_config = replicate_weight
    attention.kv_norm.sharding_config = replicate_weight

    attention.wkv_b.sharding_config = colwise_config()
    attention.wo.sharding_config = rowwise_config(output_sp=enable_sp)

    set_gqa_inner_attention_local_map(attention.inner_attention)

    # Query projection: depends on q_lora_rank
    if attention.q_lora_rank == 0:
        assert attention.wq is not None
        attention.wq.sharding_config = colwise_config()
    else:
        # Low-rank: wq_a + q_norm stay Replicate DTensors; wq_b is Colwise.
        assert attention.wq_a is not None
        assert attention.wq_b is not None
        attention.wq_a.sharding_config = replicate_weight
        attention.q_norm.sharding_config = replicate_weight
        attention.wq_b.sharding_config = colwise_config()

    # Dense FFN (non-MoE layers only)
    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )
