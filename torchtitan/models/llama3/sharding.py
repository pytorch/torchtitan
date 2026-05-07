# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
    set_gqa_inner_attention_local_spmd,
)
from torchtitan.protocols.types import MeshAxisName

if TYPE_CHECKING:
    from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock

DP_REPLICATE = MeshAxisName.DP_REPLICATE
DP_SHARD = MeshAxisName.DP_SHARD
FSDP = MeshAxisName.FSDP
TP = MeshAxisName.TP


def set_llama3_sharding_config(
    config: "Llama3Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
    full_spmd_types: bool = False,
    dp_replicate_enabled: bool = False,
    dp_shard_enabled: bool = True,
    enable_tp: bool = False,
) -> None:
    """Fill ``sharding_config`` on all Llama3 sub-configs.

    Specs are populated unconditionally — the mesh actually passed to
    ``Module.parallelize()`` at runtime determines which declarations
    apply. Declarations for mesh axes that aren't enabled (e.g. ``TP``
    placements under FSDP-only) are skipped at parallelize time.

    ``enable_sp`` controls SequenceParallel (decoupled from TP).
    ``loss_parallel`` controls whether the output projection is vocab-parallel.
    ``full_spmd_types`` uses LocalSpmdConfig instead of LocalMapConfig.
    """
    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp,
        full_spmd_types=full_spmd_types,
    )
    for layer_cfg in config.layers:
        _set_llama3_layer_sharding(
            layer_cfg,
            enable_sp=enable_sp,
            full_spmd_types=full_spmd_types,
            dp_replicate_enabled=dp_replicate_enabled,
            dp_shard_enabled=dp_shard_enabled,
            enable_tp=enable_tp,
        )


def _set_llama3_layer_sharding(
    layer_cfg: "Llama3TransformerBlock.Config",
    *,
    enable_sp: bool,
    full_spmd_types: bool = False,
    dp_replicate_enabled: bool = False,
    dp_shard_enabled: bool = True,
    enable_tp: bool = False,
) -> None:
    """Set sharding on one Llama3 transformer layer.

    ``enable_sp=True``  -> SP norms and Shard(1) activations around attention/FFN;
    ``attention.wo`` and ``feed_forward.w2`` reduce-scatter to Shard(1).
    ``enable_sp=False`` -> norms stay Replicate (no parallelism), activations
    stay Replicate; ``attention.wo`` and ``feed_forward.w2`` all-reduce to
    Replicate.
    """
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
    if full_spmd_types:
        _set_inner_attention_local_spmd(
            layer_cfg.attention.inner_attention,
            dp_replicate_enabled=dp_replicate_enabled,
            dp_shard_enabled=dp_shard_enabled,
            enable_tp=enable_tp,
        )
    else:
        set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

    assert layer_cfg.feed_forward is not None
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_placement=attn_x_placement,
        enable_sp=enable_sp,
    )


def _set_inner_attention_local_spmd(
    inner_attention_cfg,
    *,
    dp_replicate_enabled: bool = False,
    dp_shard_enabled: bool = True,
    enable_tp: bool = False,
) -> None:
    """Install a LocalSpmdConfig for inner attention with DP and/or TP axes."""
    from spmd_types import S, V

    from torchtitan.protocols.sharding import SpmdAnnotation

    # q/k/v shape: (batch, seq, heads, head_dim)
    batch_axes: list = []
    heads_axis = None

    if dp_replicate_enabled:
        batch_axes.append(DP_REPLICATE)
    if dp_shard_enabled:
        batch_axes.append(DP_SHARD)

    if enable_tp:
        heads_axis = TP

    # When multiple axes shard different tensor dims, we need explicit V types
    # + PartitionSpec template. With a single axis, S(dim) suffices (local_map
    # auto-decays S(dim) to V + PartitionSpec).
    sharded_axes = batch_axes + ([heads_axis] if heads_axis else [])
    needs_spec = len(sharded_axes) > 1

    if needs_spec:
        qkv_type: dict = {axis: V for axis in sharded_axes}
        batch_entry = tuple(batch_axes) if len(batch_axes) > 1 else batch_axes[0]
        spec_template = (batch_entry, None, heads_axis, None)
        annotation = SpmdAnnotation(types=qkv_type, partition_spec=spec_template)
    else:
        qkv_type = {}
        if batch_axes:
            qkv_type[batch_axes[0]] = S(0)
        if heads_axis:
            qkv_type[heads_axis] = S(2)
        annotation = SpmdAnnotation(types=qkv_type)

    set_gqa_inner_attention_local_spmd(
        inner_attention_cfg,
        inputs=(annotation, annotation, annotation),
        out=annotation,
    )
