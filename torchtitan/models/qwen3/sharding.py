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
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.attention import GQAttention
from torchtitan.protocols.sharding import MeshDimName, ShardingSpec

TP = MeshDimName.TP

if TYPE_CHECKING:
    from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock


def set_qwen3_sharding_spec(
    config: "Qwen3Model.Config",
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


def _set_qwen3_layer_sharding(
    layer_cfg: "Qwen3TransformerBlock.Config",
    *,
    enable_sp: bool,
) -> None:
    """Set sharding on one Qwen3 transformer layer."""
    attention = layer_cfg.attention
    assert isinstance(attention, GQAttention.Config)

    norm_spec = sequence_parallel_spec() if enable_sp else replicate_norm_spec()
    layer_cfg.attention_norm.sharding_spec = norm_spec
    layer_cfg.ffn_norm.sharding_spec = norm_spec
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    set_gqa_attention_sharding(attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # QK norms: shard on head dim (dim=2) — independent of SP.
    if attention.qk_norm is not None:
        attention.qk_norm.sharding_spec = ShardingSpec(
            state_shardings={"weight": {TP: Replicate()}},
            input_layouts={"input": {TP: Shard(2)}},
            in_shardings={"input": {TP: Shard(2)}},
            out_shardings={TP: Shard(2)},
        )

    # Dense FFN (non-MoE layers only)
    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )
