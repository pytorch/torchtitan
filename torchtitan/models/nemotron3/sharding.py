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
    colwise_config,
    dense_activation_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

# Nemotron-H's grouped experts are UNGATED: only an up projection (``w1_EFD``,
# ``[E, hidden, dim]``) and a down projection (``w2_EDF``, ``[E, dim, hidden]``).
# There is no ``w3_EFD`` gate, so the hidden dim is sharded on axis 1 of w1 and
# axis 2 of w2 -- the same column/row pair a SwiGLU expert uses for w3/w2.
_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
}

if TYPE_CHECKING:
    from torchtitan.models.nemotron3.model import (
        Nemotron3Model,
        NemotronTransformerBlock,
    )


def _set_nemotron_mlp_sharding(
    feed_forward_cfg,
    *,
    attn_x_layout: spmd.SpmdType,
    enable_sp: bool,
) -> None:
    """TP sharding for the ungated Nemotron-H MLP (``up_proj``/``down_proj``).

    The dense FFN is ``down_proj(relu(up_proj(x)) ** 2)`` -- there is no gate
    projection, so ``set_dense_ffn_sharding``'s ``w1``/``w2``/``w3`` layout does
    not apply. The two matrices map onto the SwiGLU case as ``up_proj`` -> ``w3``
    (column-parallel, hidden dim sharded on the output axis) and ``down_proj``
    -> ``w2`` (row-parallel, hidden dim sharded on the input axis), which keeps
    the hidden dimension sharded end to end and leaves a single reduction on the
    down projection.
    """
    feed_forward_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": attn_x_layout},
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
    )
    feed_forward_cfg.up_proj.sharding_config = colwise_config()
    feed_forward_cfg.down_proj.sharding_config = rowwise_config(output_sp=enable_sp)


def set_nemotron_sharding_config(
    config: "Nemotron3Model.Config",
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
        _set_nemotron_layer_sharding(
            layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
        )


def _set_nemotron_layer_sharding(
    layer_cfg: "NemotronTransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool = False,
) -> None:
    """Set sharding on one Nemotron-3 Nano transformer layer.

    For Transformer blocks (GQA):
    ``enable_sp=True``  -> SP norms and Shard(0) activations around attention/FFN;
    ``attention.wo`` and ``feed_forward.down_proj`` reduce-scatter to Shard(0).
    ``enable_sp=False`` -> norms stay Replicate (no parallelism), activations
    stay Replicate; ``attention.wo`` and ``feed_forward.down_proj`` all-reduce to
    Replicate.

    Layers are typed by ``block_type`` ("mamba" | "attention" | "mlp" | "moe")
    and hold exactly one mixer each, so each mixer below is applied only when
    that layer actually owns it.
    """
    norm = norm_config(enable_sp=enable_sp)

    # ``block_type`` is authoritative (see NemotronTransformerBlock.__init__);
    # ``is_mamba_block`` is only a fallback for configs predating it. Exactly one
    # mixer is non-None per layer, so every mixer below stays guarded.
    block_type = getattr(layer_cfg, "block_type", None)
    is_mamba = (
        block_type == "mamba"
        if block_type is not None
        else bool(getattr(layer_cfg, "is_mamba_block", False))
    )

    if is_mamba:
        # Mamba block sharding: norm + mamba projections + state_matrix.
        # NOTE: the block builds raw nn.Linear/nn.Conv1d/nn.Parameter members
        # (in_proj, conv1d, out_proj, A_log, D, dt_bias) rather than these
        # configs; they are Replicated over the dense mesh by
        # NemotronTransformerBlock.parallelize, so nothing is left unannotated,
        # but the mamba mixer is NOT tensor-parallel today.
        if layer_cfg.attention_norm is not None:
            layer_cfg.attention_norm.sharding_config = norm
        # Deliberately no sharding_config is written onto
        # ``mamba_input_projection`` / ``mamba_output_projection``: model.py
        # reads only ``.in_features`` off those configs to derive the model
        # dim and never builds a Linear from them, so any sharding set here
        # would be silently discarded while implying mamba is tensor-parallel.
        # Implementing real mamba TP means head-sharding in_proj over the
        # fused (z | x,B,C | dt) layout, grouping conv1d by channel, sharding
        # the per-head A_log/D/dt_bias to match, and aligning mamba_norm's
        # groups -- a deliberate scheme, not a default.
    else:
        # Transformer block sharding: attention + FFN/MoE. Attention layers carry
        # only ``attention_norm``; mlp/moe layers carry only ``ffn_norm``, so both
        # are guarded rather than assumed present.
        if layer_cfg.attention_norm is not None:
            layer_cfg.attention_norm.sharding_config = norm
        if layer_cfg.ffn_norm is not None:
            layer_cfg.ffn_norm.sharding_config = norm

        attn_x_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        )

        # Set attention sharding
        if hasattr(layer_cfg, "attention") and layer_cfg.attention is not None:
            set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
            set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

        # Set FFN sharding (MoE handled by expert parallel). Nemotron-H's dense
        # FFN is the ungated NemotronMLP (up_proj/down_proj); fall back to the
        # shared w1/w2/w3 helper only for a genuinely gated SwiGLU config.
        if hasattr(layer_cfg, "feed_forward") and layer_cfg.feed_forward is not None:
            ffn_cfg = layer_cfg.feed_forward
            if hasattr(ffn_cfg, "up_proj") and hasattr(ffn_cfg, "down_proj"):
                _set_nemotron_mlp_sharding(
                    ffn_cfg,
                    attn_x_layout=attn_x_layout,
                    enable_sp=enable_sp,
                )
            else:
                set_dense_ffn_sharding(
                    ffn_cfg,
                    attn_x_layout=attn_x_layout,
                    enable_sp=enable_sp,
                )

        if hasattr(layer_cfg, "moe") and layer_cfg.moe is not None:
            set_moe_sharding_config(
                layer_cfg.moe,
                enable_ep=enable_ep,
                enable_sp=enable_sp,
                expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
            )
