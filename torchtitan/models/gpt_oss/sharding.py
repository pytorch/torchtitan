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


def set_gptoss_sharding_spec(
    config,
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
) -> None:
    """Fill ``sharding_spec`` on all GPT-OSS sub-configs.

    No-op when TP is not enabled.
    """
    if not parallel_dims.tp_enabled:
        return

    set_decoder_sharding_spec(config, loss_parallel)
    for layer_cfg in config.layers:
        _set_gptoss_layer_sharding(layer_cfg)


def _set_gptoss_layer_sharding(layer_cfg) -> None:
    """Set sharding on one GPT-OSS transformer layer.

    All GPT-OSS blocks are MoE — only attention/norms are sharded here.
    MoE FFN stays under apply_moe_ep_tp.
    """
    # Norms: SequenceParallel
    layer_cfg.attention_norm.sharding_spec = sequence_parallel_spec()
    layer_cfg.ffn_norm.sharding_spec = sequence_parallel_spec()

    # Attention: input x needs all-gather, freqs_cis is Replicate.
    # sinks parameter is sharded across heads via state_shardings.
    layer_cfg.attention.sharding_spec = ShardingSpec(
        state_shardings={"sinks": {TP: Shard(0)}},
        input_layouts={
            "x": {TP: Shard(1)},
            "freqs_cis": {TP: Replicate()},
        },
        in_shardings={
            "x": {TP: Replicate()},
            "freqs_cis": {TP: Replicate()},
        },
    )
    # wq/wkv: ColwiseParallel
    for w in (layer_cfg.attention.wq, layer_cfg.attention.wkv):
        w.sharding_spec = colwise_spec()
    # wo: RowwiseParallel with reduce-scatter to Shard(1)
    layer_cfg.attention.wo.sharding_spec = rowwise_spec(out_shardings={TP: Shard(1)})

    # Inner attention: local_map to convert TP DTensors to local tensors.
    # GPT-OSS: q/k/v are (bs, seq, heads, head_dim) — no transpose, heads at dim 2.
    # return_lse=True always, so out_placements is a 2-tuple (output, lse).
    qkv_placements = (Shard(2),)
    layer_cfg.attention.inner_attention.sharding_spec = ShardingSpec(
        local_map=LocalMapSpec(
            in_placements=(qkv_placements, qkv_placements, qkv_placements),
            out_placements=(qkv_placements, qkv_placements),
            in_grad_placements=(qkv_placements, qkv_placements, qkv_placements),
        ),
    )
