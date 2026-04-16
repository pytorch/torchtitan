# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.sharding import (
    colwise_spec,
    replicate_norm_spec,
    rowwise_spec,
    sequence_parallel_spec,
    set_decoder_sharding_spec,
)
from torchtitan.models.common.attention import GQAttention
from torchtitan.protocols.sharding import MeshDimName, ShardingSpec

if TYPE_CHECKING:
    from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


def set_llama3_sharding_spec(
    config: "Llama3Model.Config",
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_spec`` on all Llama3 sub-configs.

    No-op when TP is not enabled. ``enable_sp`` controls SequenceParallel,
    which is decoupled from TP (see ``ParallelismConfig.enable_sequence_parallel``).
    """
    if not parallel_dims.tp_enabled:
        return

    set_decoder_sharding_spec(config, loss_parallel=loss_parallel, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_llama3_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_llama3_layer_sharding(
    layer_cfg: "Llama3TransformerBlock.Config",
    *,
    enable_sp: bool,
) -> None:
    """Set sharding on one Llama3 transformer layer.

    ``enable_sp=True``  → SP norms and Shard(1) activations around attention/FFN;
    ``attention.wo`` and ``feed_forward.w2`` reduce-scatter to Shard(1).
    ``enable_sp=False`` → norms stay Replicate (no parallelism), activations
    stay Replicate; ``attention.wo`` and ``feed_forward.w2`` all-reduce to
    Replicate.
    """
    TP = MeshDimName.TP

    # Narrow attention type — Llama3 always uses GQAttention.
    attention = layer_cfg.attention
    assert isinstance(
        attention, GQAttention.Config
    ), f"Llama3 layer attention must be GQAttention, got {type(attention).__name__}"

    norm_spec = sequence_parallel_spec() if enable_sp else replicate_norm_spec()
    layer_cfg.attention_norm.sharding_spec = norm_spec
    layer_cfg.ffn_norm.sharding_spec = norm_spec
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    # Attention needs Replicate activations internally for the linear ops.
    # Under SP, x arrives Shard(1) and is all-gathered to Replicate.
    # Without SP, x is already Replicate — in_shardings is an identity.
    # rope_cache (freqs_cis) is always plain; annotate as Replicate.
    attention.sharding_spec = ShardingSpec(
        input_layouts={
            "x": {TP: attn_x_placement},
            "rope_cache": {TP: Replicate()},
        },
        in_shardings={
            "x": {TP: Replicate()},
            "rope_cache": {TP: Replicate()},
        },
    )
    attention.wq.sharding_spec = colwise_spec()
    attention.wkv.sharding_spec = colwise_spec()
    attention.wo.sharding_spec = rowwise_spec(output_sp=enable_sp)

    assert layer_cfg.feed_forward is not None
    layer_cfg.feed_forward.sharding_spec = ShardingSpec(
        input_layouts={"x": {TP: attn_x_placement}},
        in_shardings={"x": {TP: Replicate()}},
    )
    layer_cfg.feed_forward.w1.sharding_spec = colwise_spec()
    layer_cfg.feed_forward.w3.sharding_spec = colwise_spec()
    layer_cfg.feed_forward.w2.sharding_spec = rowwise_spec(output_sp=enable_sp)
