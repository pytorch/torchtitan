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


def set_qwen3_sharding_spec(
    config,
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_spec`` on all Qwen3 sub-configs.

    No-op when TP is not enabled.
    """
    if not parallel_dims.tp_enabled:
        return

    set_decoder_sharding_spec(config, loss_parallel=loss_parallel, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_qwen3_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_qwen3_layer_sharding(layer_cfg, *, enable_sp: bool) -> None:
    """Set sharding on one Qwen3 transformer layer."""
    norm_spec = sequence_parallel_spec() if enable_sp else replicate_norm_spec()
    layer_cfg.attention_norm.sharding_spec = norm_spec
    layer_cfg.ffn_norm.sharding_spec = norm_spec
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    # Attention: input x arrives per ``attn_x_placement``; all-gather to Replicate.
    # rope_cache (freqs_cis) is always plain; annotate as Replicate.
    layer_cfg.attention.sharding_spec = ShardingSpec(
        input_layouts={
            "x": {TP: attn_x_placement},
            "rope_cache": {TP: Replicate()},
        },
        in_shardings={
            "x": {TP: Replicate()},
            "rope_cache": {TP: Replicate()},
        },
    )
    for w in (layer_cfg.attention.wq, layer_cfg.attention.wkv):
        w.sharding_spec = colwise_spec()
    layer_cfg.attention.wo.sharding_spec = rowwise_spec(output_sp=enable_sp)

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

    # QK norms: shard on head dim (dim=2) — independent of SP.
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
            input_layouts={"x": {TP: attn_x_placement}},
            in_shardings={"x": {TP: Replicate()}},
        )
        layer_cfg.feed_forward.w1.sharding_spec = colwise_spec()
        layer_cfg.feed_forward.w3.sharding_spec = colwise_spec()
        layer_cfg.feed_forward.w2.sharding_spec = rowwise_spec(output_sp=enable_sp)
