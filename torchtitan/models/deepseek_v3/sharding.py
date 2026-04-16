# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.sharding import (
    colwise_spec,
    replicate_norm_spec,
    rowwise_spec,
    sequence_parallel_spec,
    set_decoder_sharding_spec,
)
from torchtitan.protocols.sharding import LocalMapSpec, MeshDimName, ShardingSpec

TP = MeshDimName.TP


def set_deepseekv3_sharding_spec(
    config,
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_spec`` on all DeepSeek V3 sub-configs.

    No-op when TP is not enabled.
    """
    if not parallel_dims.tp_enabled:
        return

    set_decoder_sharding_spec(config, loss_parallel=loss_parallel, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_deepseekv3_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_deepseekv3_layer_sharding(layer_cfg, *, enable_sp: bool) -> None:
    """Set sharding on one DeepSeek V3 transformer layer.

    MLA attention: low-rank projections (wkv_a, wq_a, kv_norm, q_norm)
    stay replicated. Up-projections (wkv_b, wq_b, wq) are colwise.
    """
    norm_spec = sequence_parallel_spec() if enable_sp else replicate_norm_spec()
    layer_cfg.attention_norm.sharding_spec = norm_spec
    layer_cfg.ffn_norm.sharding_spec = norm_spec
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    # MLA attention input: x is gathered to Replicate; freqs_cis always Replicate.
    layer_cfg.attention.sharding_spec = ShardingSpec(
        input_layouts={
            "x": {TP: attn_x_placement},
            "freqs_cis": {TP: Replicate()},
        },
        in_shardings={
            "x": {TP: Replicate()},
            "freqs_cis": {TP: Replicate()},
        },
    )
    # wkv_a, kv_norm: no spec (low-rank, stays replicated)
    # wkv_b: ColwiseParallel (expands to full heads)
    layer_cfg.attention.wkv_b.sharding_spec = colwise_spec()
    layer_cfg.attention.wo.sharding_spec = rowwise_spec(output_sp=enable_sp)

    # Inner attention: local_map to convert TP DTensors to local tensors.
    # MLA: q/k/v are (bs, seq, heads, head_dim) — no transpose, heads at dim 2.
    qkv_placements = (Shard(2),)
    layer_cfg.attention.inner_attention.sharding_spec = ShardingSpec(
        local_map=LocalMapSpec(
            in_placements=(qkv_placements, qkv_placements, qkv_placements),
            out_placements=(qkv_placements,),
            in_grad_placements=(qkv_placements, qkv_placements, qkv_placements),
        ),
    )

    # Query projection: depends on q_lora_rank
    if layer_cfg.attention.q_lora_rank == 0:
        layer_cfg.attention.wq.sharding_spec = colwise_spec()
    else:
        # Low-rank: wq_a/q_norm stay replicated, wq_b is colwise
        layer_cfg.attention.wq_b.sharding_spec = colwise_spec()

    # Dense FFN (non-MoE layers only)
    if layer_cfg.feed_forward is not None:
        layer_cfg.feed_forward.sharding_spec = ShardingSpec(
            input_layouts={"x": {TP: attn_x_placement}},
            in_shardings={"x": {TP: Replicate()}},
        )
        layer_cfg.feed_forward.w1.sharding_spec = colwise_spec()
        layer_cfg.feed_forward.w3.sharding_spec = colwise_spec()
        layer_cfg.feed_forward.w2.sharding_spec = rowwise_spec(output_sp=enable_sp)
