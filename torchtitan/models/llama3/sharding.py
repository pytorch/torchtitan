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
    sequence_parallel_spec,
    set_decoder_sharding_spec,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
)
from torchtitan.protocols.sharding import LocalMapSpec, ShardingSpec

if TYPE_CHECKING:
    from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


def set_llama3_sharding_spec(
    config: "Llama3Model.Config",
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
    enable_sp: bool,
    full_dtensor: bool = False,
) -> None:
    """Fill ``sharding_spec`` on all Llama3 sub-configs.

    No-op when TP is not enabled and not full_dtensor.

    ``enable_sp`` controls SequenceParallel (decoupled from TP).
    ``full_dtensor`` extends the inner-attention ``LocalMapSpec`` to also
    carry DP/CP placements so all params/inputs flow as DTensors on the
    multi-D SPMD mesh.
    """
    if not parallel_dims.tp_enabled and not full_dtensor:
        return

    set_decoder_sharding_spec(config, loss_parallel=loss_parallel, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_llama3_layer_sharding(
            layer_cfg,
            parallel_dims,
            enable_sp=enable_sp,
            full_dtensor=full_dtensor,
        )


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
    q_placements: list[Placement] = []
    kv_placements: list[Placement] = []

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
    layer_cfg: "Llama3TransformerBlock.Config",
    parallel_dims: ParallelDims,
    *,
    enable_sp: bool,
    full_dtensor: bool,
) -> None:
    """Set sharding on one Llama3 transformer layer.

    ``enable_sp=True``  -> SP norms and Shard(1) activations around attention/FFN;
    ``attention.wo`` and ``feed_forward.w2`` reduce-scatter to Shard(1).
    ``enable_sp=False`` -> norms stay Replicate (no parallelism), activations
    stay Replicate; ``attention.wo`` and ``feed_forward.w2`` all-reduce to
    Replicate.
    """
    norm_spec = sequence_parallel_spec() if enable_sp else replicate_norm_spec()
    layer_cfg.attention_norm.sharding_spec = norm_spec
    layer_cfg.ffn_norm.sharding_spec = norm_spec
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)

    # Inner attention: local_map to convert DTensors to local tensors.
    # Under full DTensor, placements include DP/CP dims (K/V all-gathered on CP).
    layer_cfg.attention.inner_attention.sharding_spec = ShardingSpec(
        local_map=_build_inner_attn_local_map_spec(parallel_dims, full_dtensor),
    )

    assert layer_cfg.feed_forward is not None
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_placement=attn_x_placement,
        enable_sp=enable_sp,
    )
