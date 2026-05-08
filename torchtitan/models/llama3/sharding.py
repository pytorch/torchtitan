# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.distributed.spmd_state import is_spmd_active
from torchtitan.models.common.decoder_sharding import (
    EMBED_OUT,
    QKV_LOCAL_SPMD,
    EMBED_OUT_SP,
    LM_HEAD_OUTPUT_REDIST,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)

if TYPE_CHECKING:
    from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


def set_llama3_sharding_config(
    config: "Llama3Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
    full_spmd_types: bool = False,
) -> None:
    """Fill ``sharding_config`` on all Llama3 sub-configs.

    Config-time choices:
    - ``enable_sp``: controls S(1) vs R on TP for activations.
    - ``loss_parallel``: controls whether lm_head output is all-gathered.
    - ``full_spmd_types``: enables SPMD annotations (constant, resolved
      at parallelize time from ``mesh()``).
    """
    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp,
    )
    if full_spmd_types:
        config.tok_embeddings.local_spmd = EMBED_OUT_SP if enable_sp else EMBED_OUT
        # SPMD path always all-gathers lm_head output (loss_parallel not yet implemented)
        config.lm_head.global_spmd = LM_HEAD_OUTPUT_REDIST
    for layer_cfg in config.layers:
        _set_llama3_layer_sharding(
            layer_cfg, enable_sp=enable_sp, full_spmd_types=full_spmd_types,
        )


def _set_llama3_layer_sharding(
    layer_cfg: "Llama3TransformerBlock.Config",
    *,
    enable_sp: bool,
    full_spmd_types: bool = False,
) -> None:
    """Set sharding on one Llama3 transformer layer."""
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
    if full_spmd_types:
        layer_cfg.attention.inner_attention.local_spmd = QKV_LOCAL_SPMD
    else:
        set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

    assert layer_cfg.feed_forward is not None
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_placement=attn_x_placement,
        enable_sp=enable_sp,
    )

