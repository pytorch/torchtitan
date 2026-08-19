# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_sequence_parallel_placement,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)

if TYPE_CHECKING:
    from torchtitan.models.olmo3.model import Olmo3Model, Olmo3TransformerBlock


def set_olmo3_sharding_config(
    config: "Olmo3Model.Config",
    *,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on all Olmo3 sub-configs.

    Specs are populated unconditionally -- the mesh actually passed to
    ``Module.parallelize()`` at runtime determines which declarations
    apply (e.g. TP placements are skipped under FSDP-only).
    """
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_olmo3_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_olmo3_layer_sharding(
    layer_cfg: "Olmo3TransformerBlock.Config",
    *,
    enable_sp: bool,
) -> None:
    """Set sharding on one Olmo3 transformer layer (dense GQA + qk_norm + FFN)."""
    attention = layer_cfg.attention
    assert isinstance(attention, GQAttention.Config)

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    set_gqa_attention_sharding(attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # q_norm/k_norm intentionally have no sharding_config: Olmo3Attention
    # normalizes over the *full* (non-head-sharded) projection width, which
    # has no valid TP-axis placement. Olmo3Model.Config.update_from_config
    # rejects tensor_parallel_degree > 1, so this module is never TP-sharded;
    # FSDP still shards its parameter via apply_fsdp_to_decoder regardless of
    # sharding_config (see Module.parallelize: a None sharding_config just
    # skips this module's DTensor state distribution).

    assert layer_cfg.feed_forward is not None
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_layout=attn_x_layout,
        enable_sp=enable_sp,
    )
