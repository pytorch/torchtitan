# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.sharding import (
    replicate_norm_spec,
    rowwise_spec,
    sequence_parallel_spec,
    set_decoder_sharding_spec,
    set_gqa_inner_attention_local_map,
    set_qkv_linear_sharding,
)
from torchtitan.models.gpt_oss.model import Attention
from torchtitan.protocols.sharding import MeshDimName, ShardingSpec

TP = MeshDimName.TP

if TYPE_CHECKING:
    from torchtitan.models.gpt_oss.model import GptOssModel, GptOssTransformerBlock


def set_gpt_oss_sharding_spec(
    config: "GptOssModel.Config",
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_spec`` on all GPT-OSS sub-configs.

    No-op when TP is not enabled.
    """
    if not parallel_dims.tp_enabled:
        return

    set_decoder_sharding_spec(config, loss_parallel=loss_parallel, enable_sp=enable_sp)
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

    norm_spec = sequence_parallel_spec() if enable_sp else replicate_norm_spec()
    layer_cfg.attention_norm.sharding_spec = norm_spec
    layer_cfg.ffn_norm.sharding_spec = norm_spec
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    # Attention: input x gathered to Replicate, freqs_cis always Replicate.
    # sinks parameter is sharded across heads via state_shardings.
    attention.sharding_spec = ShardingSpec(
        state_shardings={"sinks": {TP: Shard(0)}},
        input_layouts={
            "x": {TP: attn_x_placement},
            "freqs_cis": {TP: Replicate()},
        },
        in_shardings={
            "x": {TP: Replicate()},
            "freqs_cis": {TP: Replicate()},
        },
    )
    set_qkv_linear_sharding(attention.qkv_linear)
    attention.wo.sharding_spec = rowwise_spec(output_sp=enable_sp)

    # GPT-OSS flash attention always returns (output, lse), hence num_outputs=2.
    set_gqa_inner_attention_local_map(attention.inner_attention, num_outputs=2)
