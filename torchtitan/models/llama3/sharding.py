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
from torchtitan.protocols.sharding import MeshDimName, ShardingSpec


def set_llama3_sharding_spec(
    config,
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
) -> None:
    """Fill ``sharding_spec`` on all Llama3 sub-configs.

    No-op when TP is not enabled.
    """
    if not parallel_dims.tp_enabled:
        return

    set_decoder_sharding_spec(config, loss_parallel)
    for layer_cfg in config.layers:
        _set_llama3_layer_sharding(layer_cfg)


def _set_llama3_layer_sharding(layer_cfg) -> None:
    """Set sharding on one Llama3 transformer layer.

    - DTensors flow everywhere (use_local_output=False in the original style)
    - ColwiseParallel outputs DTensor Shard(-1)
    - RowwiseParallel outputs DTensor Shard(1) (sequence parallel)
    - SequenceParallel norms keep DTensor
    - rope_cache is wrapped as DTensor Replicate via in_shardings
    """
    TP = MeshDimName.TP

    # Norms: SequenceParallel
    layer_cfg.attention_norm.sharding_spec = sequence_parallel_spec()
    layer_cfg.ffn_norm.sharding_spec = sequence_parallel_spec()

    # Attention: input x is Shard(1) from sequence-parallel norm,
    # needs all-gather to Replicate. rope_cache (freqs_cis) is a plain
    # tensor that must be wrapped as DTensor Replicate.
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
    for w in (
        layer_cfg.attention.wq,
        layer_cfg.attention.wkv,
    ):
        w.sharding_spec = colwise_spec()
    layer_cfg.attention.wo.sharding_spec = rowwise_spec(out_shardings={TP: Shard(1)})

    # FFN: input x is Shard(1) from sequence-parallel norm,
    # needs all-gather to Replicate.
    assert layer_cfg.feed_forward is not None
    layer_cfg.feed_forward.sharding_spec = ShardingSpec(
        input_layouts={"x": {TP: Shard(1)}},
        in_shardings={"x": {TP: Replicate()}},
    )
    layer_cfg.feed_forward.w1.sharding_spec = colwise_spec()
    layer_cfg.feed_forward.w3.sharding_spec = colwise_spec()
    layer_cfg.feed_forward.w2.sharding_spec = rowwise_spec(out_shardings={TP: Shard(1)})
