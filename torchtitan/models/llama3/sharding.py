# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    replicate_norm_spec,
    sequence_parallel_spec,
    set_decoder_sharding_spec,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
)
from torchtitan.protocols.sharding import (
    LocalMapSpec,
    MeshDimName,
    NamedPlacement,
    ShardingSpec,
)

if TYPE_CHECKING:
    from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


def set_llama3_sharding_spec(
    config: "Llama3Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
    full_dtensor: bool = False,
) -> None:
    """Fill ``sharding_spec`` on all Llama3 sub-configs.

    Specs are populated unconditionally — the mesh actually passed to
    ``Module.parallelize()`` at runtime determines which declarations
    apply. Declarations for mesh dims that aren't enabled (e.g. ``TP``
    placements under FSDP-only) are skipped at parallelize time.

    ``enable_sp`` controls SequenceParallel (decoupled from TP).
    ``loss_parallel`` controls whether the output projection is vocab-parallel.
    ``full_dtensor`` extends the inner-attention ``LocalMapSpec`` to also
    carry DP/CP placements so q/k/v flow as DTensors on the multi-D SPMD mesh.
    """
    set_decoder_sharding_spec(config, loss_parallel=loss_parallel, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_llama3_layer_sharding(
            layer_cfg, enable_sp=enable_sp, full_dtensor=full_dtensor
        )


def _build_inner_attn_local_map_spec(full_dtensor: bool) -> LocalMapSpec:
    """Build LocalMapSpec for inner attention.

    q/k/v are (bs, seq, heads, head_dim). TP shards on heads (dim 2).
    Under full DTensor: q/k/v are batch-sharded on DP dims; Q keeps
    sequence-sharding on CP while K/V are Replicate so DTensor all-gathers
    them across CP ranks.

    Non-full_dtensor path: parallelize is called with a 1-D TP mesh, so
    declaring only ``{TP: Shard(2)}`` matches (strict resolve iterates
    mesh dims only).
    """
    q: NamedPlacement
    kv: NamedPlacement
    if full_dtensor:
        q = dense_activation_placement(tp=Shard(2))
        kv = dense_activation_placement(tp=Shard(2), cp=Replicate())
    else:
        q = {MeshDimName.TP: Shard(2)}
        kv = {MeshDimName.TP: Shard(2)}

    return LocalMapSpec(
        in_placements=(q, kv, kv),
        out_placements=(q,),
        in_grad_placements=(q, kv, kv),
    )


def _set_llama3_layer_sharding(
    layer_cfg: "Llama3TransformerBlock.Config",
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
        local_map=_build_inner_attn_local_map_spec(full_dtensor),
    )

    assert layer_cfg.feed_forward is not None
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_placement=attn_x_placement,
        enable_sp=enable_sp,
    )
