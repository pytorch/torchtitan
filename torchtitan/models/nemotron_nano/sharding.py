# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Nemotron-3 Nano Sharding Configuration

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_sequence_parallel_placement,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
    colwise_config,
    rowwise_config,
    dense_param_placement,
)
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.models.common.moe_sharding import set_moe_sharding_config

_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}

if TYPE_CHECKING:
    from torchtitan.models.nemotron_nano.model import (
        Nemotron3NanoModel,
        NemotronTransformerBlock,
    )


def set_nemotron_sharding_config(
    config: "Nemotron3NanoModel.Config",
    *,
    enable_sp: bool,
    enable_ep: bool = False,
) -> None:
    """Fill ``sharding_config`` on all Nemotron-3 Nano sub-configs.

    Specs are populated unconditionally — the mesh actually passed to
    ``Module.parallelize()`` at runtime determines which declarations
    apply. Declarations for mesh axes that aren't enabled (e.g. ``TP``
    placements under FSDP-only) are skipped at parallelize time.

    ``enable_sp`` controls SequenceParallel (decoupled from TP).
    ``enable_ep`` controls ExpertParallel (for MoE layers).
    """
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_nemotron_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_nemotron_layer_sharding(
    layer_cfg: "NemotronTransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool = False,
) -> None:
    """Set sharding on one Nemotron-3 Nano transformer layer.

    For Transformer blocks (GQA):
    ``enable_sp=True``  -> SP norms and Shard(0) activations around attention/FFN;
    ``attention.wo`` and ``feed_forward.w2`` reduce-scatter to Shard(0).
    ``enable_sp=False`` -> norms stay Replicate (no parallelism), activations
    stay Replicate; ``attention.wo`` and ``feed_forward.w2`` all-reduce to Replicate.

    For Mamba blocks: No attention sharding; FFN sharding follows same pattern.
    """
    norm = norm_config(enable_sp=enable_sp)
    
    if layer_cfg.is_mamba_block:
        # Mamba block sharding: norm + mamba projections + state_matrix
        layer_cfg.attention_norm.sharding_config = norm
        if hasattr(layer_cfg, 'mamba_input_projection') and layer_cfg.mamba_input_projection is not None:
            mamba_in_cfg = colwise_config()
            mamba_x_layout = (
                dense_sequence_parallel_placement()
                if enable_sp
                else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
            )
            mamba_in_cfg.in_src_shardings = {"input": mamba_x_layout}
            mamba_in_cfg.in_dst_shardings = {"input": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))}
            layer_cfg.mamba_input_projection.sharding_config = mamba_in_cfg
        if hasattr(layer_cfg, 'mamba_output_projection') and layer_cfg.mamba_output_projection is not None:
            layer_cfg.mamba_output_projection.sharding_config = rowwise_config(output_sp=enable_sp)
    else:
        # Transformer block sharding: attention + FFN/MoE
        layer_cfg.attention_norm.sharding_config = norm
        layer_cfg.ffn_norm.sharding_config = norm
        
        attn_x_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        )

        # Set attention sharding
        if hasattr(layer_cfg, 'attention') and layer_cfg.attention is not None:
            set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
            set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

        # Set FFN sharding (MoE handled by expert parallel)
        if hasattr(layer_cfg, 'feed_forward') and layer_cfg.feed_forward is not None:
            set_dense_ffn_sharding(
                layer_cfg.feed_forward,
                attn_x_layout=attn_x_layout,
                enable_sp=enable_sp,
            )
            
        if hasattr(layer_cfg, 'moe') and layer_cfg.moe is not None:
            set_moe_sharding_config(
                layer_cfg.moe,
                enable_ep=enable_ep,
                enable_sp=enable_sp,
                expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
            )
