# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.tensor import Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.sharding import (
    colwise_spec,
    rowwise_spec,
    sequence_parallel_spec,
    set_decoder_sharding_spec,
)
from torchtitan.protocols.sharding import LocalMapSpec, MeshDimName, ShardingSpec

TP = MeshDimName.TP


def set_qwen3_sharding_spec(
    config,
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
) -> None:
    """Fill ``sharding_spec`` on all Qwen3 sub-configs.

    No-op when TP is not enabled.
    """
    if not parallel_dims.tp_enabled:
        return

    set_decoder_sharding_spec(config, loss_parallel)
    for layer_cfg in config.layers:
        _set_qwen3_layer_sharding(layer_cfg)


def _set_qwen3_layer_sharding(layer_cfg) -> None:
    """Set sharding on one Qwen3 transformer layer."""
    # Norms: SequenceParallel
    layer_cfg.attention_norm.sharding_spec = sequence_parallel_spec()
    layer_cfg.ffn_norm.sharding_spec = sequence_parallel_spec()

    # Attention: input x is Shard(1) from sequence-parallel norm,
    # needs all-gather to Replicate. rope_cache is a plain tensor
    # that must be wrapped as DTensor Replicate.
    layer_cfg.attention.sharding_spec = ShardingSpec(
        input_layouts={
            "x": {TP: Shard(1)},
            "rope_cache": {TP: Replicate()},
        },
        in_shardings={
            "x": {TP: Replicate()},
            "rope_cache": {TP: Replicate()},
        },
    )
    # wq/wkv: ColwiseParallel
    for w in (layer_cfg.attention.wq, layer_cfg.attention.wkv):
        w.sharding_spec = colwise_spec()
    # wo: RowwiseParallel with reduce-scatter to Shard(1)
    layer_cfg.attention.wo.sharding_spec = rowwise_spec(out_shardings={TP: Shard(1)})

    # Inner attention: local_map to convert TP DTensors to local tensors.
    # q/k/v are (bs, seq, heads, head_dim) from GQAttention, heads at dim 2.
    qkv_placements = (Shard(2),)
    layer_cfg.attention.inner_attention.sharding_spec = ShardingSpec(
        local_map=LocalMapSpec(
            in_placements=(qkv_placements, qkv_placements, qkv_placements),
            out_placements=(qkv_placements,),
            in_grad_placements=(qkv_placements, qkv_placements, qkv_placements),
        ),
    )

    # QK norms: SequenceParallel on head dim (dim=2)
    if layer_cfg.attention.qk_norm is not None:
        qk_norm_spec = ShardingSpec(
            state_shardings={"weight": {TP: Replicate()}},
            input_layouts={"input": {TP: Shard(2)}},
            in_shardings={"input": {TP: Shard(2)}},
            out_shardings={TP: Shard(2)},
        )
        layer_cfg.attention.qk_norm.sharding_spec = qk_norm_spec

    # Dense FFN (non-MoE layers only)
    if layer_cfg.feed_forward is not None:
        layer_cfg.feed_forward.sharding_spec = ShardingSpec(
            input_layouts={"x": {TP: Shard(1)}},
            in_shardings={"x": {TP: Replicate()}},
        )
        layer_cfg.feed_forward.w1.sharding_spec = colwise_spec()
        layer_cfg.feed_forward.w3.sharding_spec = colwise_spec()
        layer_cfg.feed_forward.w2.sharding_spec = rowwise_spec(
            out_shardings={TP: Shard(1)}
        )
