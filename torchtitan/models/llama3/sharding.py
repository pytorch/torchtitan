# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    replicate_norm_spec,
    sequence_parallel_spec,
    set_decoder_sharding_spec,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)

if TYPE_CHECKING:
    from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


def set_llama3_sharding_spec(
    config: "Llama3Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_spec`` on all Llama3 sub-configs.

    Specs are populated unconditionally — the mesh actually passed to
    ``Module.parallelize()`` at runtime determines which declarations
    apply. Declarations for mesh dims that aren't enabled (e.g. ``TP``
    placements under FSDP-only) are skipped at parallelize time.

    ``enable_sp`` controls SequenceParallel (decoupled from TP).
    ``loss_parallel`` controls whether the output projection is vocab-parallel.
    """
    set_decoder_sharding_spec(config, loss_parallel=loss_parallel, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_llama3_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_llama3_layer_sharding(
    layer_cfg: "Llama3TransformerBlock.Config",
    *,
    enable_sp: bool,
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
    set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

    assert layer_cfg.feed_forward is not None
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_placement=attn_x_placement,
        enable_sp=enable_sp,
    )
