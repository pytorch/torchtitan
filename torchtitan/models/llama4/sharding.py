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
from torchtitan.protocols.sharding import MeshDimName, ShardingSpec

TP = MeshDimName.TP


def set_llama4_sharding_spec(
    config,
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_spec`` on all Llama4 sub-configs.

    No-op when TP is not enabled.
    """
    if not parallel_dims.tp_enabled:
        return

    set_decoder_sharding_spec(config, loss_parallel=loss_parallel, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_llama4_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_llama4_layer_sharding(layer_cfg, *, enable_sp: bool) -> None:
    """Set sharding on one Llama4 transformer layer.

    Attention and norms are sharded on all blocks (MoE and non-MoE).
    Dense FFN is only sharded on non-MoE blocks — MoE FFN stays
    under apply_moe_ep_tp.
    """
    norm_spec = sequence_parallel_spec() if enable_sp else replicate_norm_spec()
    layer_cfg.attention_norm.sharding_spec = norm_spec
    layer_cfg.ffn_norm.sharding_spec = norm_spec
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

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

    # Dense FFN (non-MoE layers only)
    if layer_cfg.feed_forward is not None:
        layer_cfg.feed_forward.sharding_spec = ShardingSpec(
            input_layouts={"x": {TP: attn_x_placement}},
            in_shardings={"x": {TP: Replicate()}},
        )
        layer_cfg.feed_forward.w1.sharding_spec = colwise_spec()
        layer_cfg.feed_forward.w3.sharding_spec = colwise_spec()
        layer_cfg.feed_forward.w2.sharding_spec = rowwise_spec(output_sp=enable_sp)
