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
CP = MeshDimName.CP


def set_llama3_sharding_spec(
    config,
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
    full_dtensor: bool = False,
) -> None:
    """Fill ``sharding_spec`` on all Llama3 sub-configs.

    No-op when TP is not enabled and not full_dtensor.
    """
    if not parallel_dims.tp_enabled and not full_dtensor:
        return

    set_decoder_sharding_spec(config, loss_parallel)
    for layer_cfg in config.layers:
        _set_llama3_layer_sharding(layer_cfg, parallel_dims, full_dtensor)


def _build_inner_attn_local_map_spec(
    parallel_dims: ParallelDims,
    full_dtensor: bool,
) -> LocalMapSpec:
    """Build LocalMapSpec for inner attention based on active mesh dims.

    q/k/v are (bs, seq, heads, head_dim). TP shards on heads (dim 2).
    In full DTensor CP mode, Q keeps Shard(1) on CP (sequence),
    K/V get Replicate on CP — DTensor all-gathers K/V across CP ranks.

    Placements are ordered to match the SPMD mesh dim order:
    dp_replicate, dp_shard, cp, tp (only active dims present).
    """
    # Build placements in mesh dim order for Q and K/V
    q_placements: list = []
    kv_placements: list = []

    if full_dtensor:
        # DP dims: q/k/v are batch-sharded (Shard(0)) on dp_replicate/dp_shard
        if parallel_dims.dp_replicate_enabled:
            q_placements.append(Shard(0))
            kv_placements.append(Shard(0))
        if parallel_dims.dp_shard_enabled:
            q_placements.append(Shard(0))
            kv_placements.append(Shard(0))
        if parallel_dims.cp_enabled:
            q_placements.append(Shard(1))  # Q: keep sequence-sharded
            kv_placements.append(Replicate())  # K/V: all-gather
    if parallel_dims.tp_enabled:
        q_placements.append(Shard(2))  # heads
        kv_placements.append(Shard(2))  # heads

    q_pl = tuple(q_placements)
    kv_pl = tuple(kv_placements)

    return LocalMapSpec(
        in_placements=(q_pl, kv_pl, kv_pl),
        out_placements=(q_pl,),
        in_grad_placements=(q_pl, kv_pl, kv_pl),
    )


def _set_llama3_layer_sharding(
    layer_cfg,
    parallel_dims: ParallelDims,
    full_dtensor: bool,
) -> None:
    """Set sharding on one Llama3 transformer layer."""
    # Norms: SequenceParallel
    layer_cfg.attention_norm.sharding_spec = sequence_parallel_spec()
    layer_cfg.ffn_norm.sharding_spec = sequence_parallel_spec()

    # Attention: input x is Shard(1) from sequence-parallel norm,
    # needs all-gather to Replicate. rope_cache is Replicate.
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

    # Inner attention: local_map to convert DTensors to local tensors.
    layer_cfg.attention.inner_attention.sharding_spec = ShardingSpec(
        local_map=_build_inner_attn_local_map_spec(parallel_dims, full_dtensor),
    )

    # FFN: input x is Shard(1) from sequence-parallel norm.
    assert layer_cfg.feed_forward is not None
    layer_cfg.feed_forward.sharding_spec = ShardingSpec(
        input_layouts={"x": {TP: Shard(1)}},
        in_shardings={"x": {TP: Replicate()}},
    )
    layer_cfg.feed_forward.w1.sharding_spec = colwise_spec()
    layer_cfg.feed_forward.w3.sharding_spec = colwise_spec()
    layer_cfg.feed_forward.w2.sharding_spec = rowwise_spec(out_shardings={TP: Shard(1)})
