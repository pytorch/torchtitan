# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Qwen3.5 hybrid attention model.

Sets ``ShardingConfig`` on all sub-configs so that ``model.parallelize()``
applies TP via the Module protocol. Same pattern as ``qwen3/sharding.py``.

Full-attention layers: TP on wq/wk/wv/wo with local_map for inner attention;
each layer's MRoPE ``cache`` buffer is sharded Replicate.
GatedDeltaNet layers: head-sharded TP on projections (ColwiseParallel) and
out_proj (RowwiseParallel); the FLA kernel and depthwise Conv1d run on local
tensors via local_map.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
import torch

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_config,
    set_vision_transformer_block_sharding_config,
    vision_colwise_config,
    vision_invariant_linear_config,
    vision_scaled_bias_rowwise_config,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

DP = MeshAxisName.DP
TP = MeshAxisName.TP

if TYPE_CHECKING:
    from torchtitan.models.common import SigmoidGatedFeedForward
    from torchtitan.models.qwen3_5.model import (
        GatedDeltaNet,
        Qwen35Attention,
        Qwen35AttentionMaskDict,
        Qwen35Model,
        Qwen35TransformerBlock,
    )
    from torchtitan.models.qwen3_5.vision_encoder import Qwen35VisionEncoder


def annotate_qwen35_input_spmd_types(
    *,
    attention_masks: "Qwen35AttentionMaskDict | None",
    mrope_positions: torch.Tensor | None,
    pixel_values: torch.Tensor | None,
    pixel_values_videos: torch.Tensor | None,
    grid_thw: torch.Tensor | None,
    grid_thw_videos: torch.Tensor | None,
) -> None:
    """Annotate Qwen3.5 structured inputs with their local SPMD types."""
    token_type = {
        MeshAxisName.DP: spmd.S(0),
        MeshAxisName.TP: spmd.R,
    }
    multimodal_type = {
        MeshAxisName.DP: spmd.V,
        MeshAxisName.TP: spmd.I,
    }

    if mrope_positions is not None:
        spmd.assert_type(mrope_positions, token_type)
    if attention_masks is not None:
        deltanet_metadata = attention_masks["deltanet"]
        assert isinstance(deltanet_metadata, VarlenMetadata)
        spmd.assert_type(
            deltanet_metadata.cu_seq_q,
            {MeshAxisName.DP: spmd.V, MeshAxisName.TP: spmd.R},
        )
    for tensor in (
        pixel_values,
        pixel_values_videos,
        grid_thw,
        grid_thw_videos,
    ):
        if tensor is not None:
            spmd.assert_type(tensor, multimodal_type)


def _qk_norm_sharding() -> ShardingConfig:
    """Per-head QK-norm sharding: weight Replicate, activations Shard(2)."""
    head_plc = dense_activation_placement(tp=spmd.S(2))
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": head_plc},
        in_dst_shardings={"input": head_plc},
        out_src_shardings=head_plc,
        out_dst_shardings=head_plc,
    )


def _decoder_norm_sharding(activation_layout: SpmdLayout) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": activation_layout},
        out_src_shardings=activation_layout,
    )


def _conv_weight_sharding() -> ShardingConfig:
    """Depthwise Conv1d weight sharded Shard(0) on out-channels (head-sharded)."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )


_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


def set_qwen35_sharding_config(
    config: "Qwen35Model.Config",
    *,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3.5 sub-configs.

    Uses SP for decoder layers, norm, and lm_head. tok_embeddings output
    stays Replicate so vision scatter and MRoPE can access the full sequence.
    The model forward redistributes to Shard(1) before entering the layers.
    """
    # SP on norm, lm_head, and layers. Each full-attention layer owns its rope;
    # its cache buffer is sharded Replicate in _set_full_attention_sharding.
    set_decoder_sharding_config(config, enable_sp=True)
    # Override tok_embeddings: output Replicate (not Shard(1)) for vision scatter
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=spmd.R),
        local_map=LocalMapConfig(in_grad_placements=None),
    )
    _set_vision_encoder_sharding(config.vision_encoder)
    # The embedding path stays replicated through multimodal vision scatter.
    # Layer 0 restores SP at the block boundary; later decoder blocks are SP.
    decoder_input_layout = dense_activation_placement(tp=spmd.R)
    layer_input_layout = dense_sequence_parallel_placement()
    for layer_idx, layer_cfg in enumerate(config.layers):
        layer_cfg.sharding_config = ShardingConfig(
            in_src_shardings={
                "x_BLD": (
                    decoder_input_layout if layer_idx == 0 else layer_input_layout
                )
            },
            in_dst_shardings={"x_BLD": layer_input_layout},
            out_src_shardings=layer_input_layout,
        )
        _set_qwen35_layer_sharding(
            layer_cfg,
            attention_input_layout=layer_input_layout,
            enable_ep=enable_ep,
        )


def _set_qwen35_layer_sharding(
    layer_cfg: "Qwen35TransformerBlock.Config",
    *,
    attention_input_layout: SpmdLayout,
    enable_ep: bool,
) -> None:
    layer_cfg.attention_norm.sharding_config = _decoder_norm_sharding(
        attention_input_layout
    )
    layer_cfg.ffn_norm.sharding_config = norm_config(enable_sp=True)

    if layer_cfg.attention is not None:
        _set_full_attention_sharding(
            layer_cfg.attention,
            attention_input_layout=attention_input_layout,
        )
    else:
        assert layer_cfg.delta_net is not None
        _set_deltanet_sharding(
            layer_cfg.delta_net,
            attention_input_layout=attention_input_layout,
        )

    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=dense_sequence_parallel_placement(),
            enable_sp=True,
        )

    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=True,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )
        # pyrefly: ignore [missing-attribute]
        _set_shared_expert_gate_sharding(layer_cfg.moe.shared_experts)


def _set_shared_expert_gate_sharding(
    shared_experts: "SigmoidGatedFeedForward.Config | None",
) -> None:
    """Shard Qwen3.5's shared-expert sigmoid gate.

    The common MoE sharding handles the shared FFN (w1/w2/w3) and the
    module-boundary gather that feeds the gate a Replicate ``x``. Here we only
    add the gate: its weight and local output are Replicate, then the output is
    sliced into the sequence-sharded layout produced by the shared FFN before
    the pointwise multiply. ``getattr`` keeps this a no-op when the MoE has no
    shared expert (``None``); Qwen3.5's shared expert always carries the gate.
    """
    gate = getattr(shared_experts, "gate", None)
    if gate is None:
        return
    gate.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.R),
        out_dst_shardings=dense_sequence_parallel_placement(),
    )


def _set_vision_encoder_sharding(ve_cfg: "Qwen35VisionEncoder.Config") -> None:
    """Sharding for the vision encoder.

    All activations flow as Replicate — no SP in the vision encoder.
    Linear layers are ColwiseParallel/RowwiseParallel for memory savings.
    Norms are Replicate. pos_embed is Replicate via state_shardings.
    """
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "pos_embed": SpmdLayout({DP: spmd.R, TP: spmd.I}),
        },
        # I->R convert to scatter into text embeddings.
        out_src_shardings=SpmdLayout({DP: spmd.V, TP: spmd.I}),
        out_dst_shardings=SpmdLayout({DP: spmd.V, TP: spmd.R}),
    )
    ve_cfg.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={
            "inv_freq": SpmdLayout({DP: spmd.R, TP: spmd.I}),
        },
        out_src_shardings=SpmdLayout({DP: spmd.R, TP: spmd.I}),
    )

    ve_cfg.patch_embed_proj.sharding_config = vision_invariant_linear_config()

    set_vision_transformer_block_sharding_config(
        ve_cfg.block,
        rope_cache_dp=spmd.R,
    )

    # Merger sub-modules
    merger = ve_cfg.merger
    merger.norm.sharding_config = invariant_norm_config()
    merger.fc1.sharding_config = vision_colwise_config()
    merger.fc2.sharding_config = vision_scaled_bias_rowwise_config()


def _set_full_attention_sharding(
    attention_cfg: "Qwen35Attention.Config",
    *,
    attention_input_layout: SpmdLayout,
) -> None:
    """TP sharding for Qwen35Attention (output gating + partial RoPE)."""
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x_BLD": attention_input_layout},
        in_dst_shardings={"x_BLD": dense_activation_placement(tp=spmd.R)},
    )
    # The per-layer rope ``cache`` buffer is a Replicate DTensor; MRoPE builds the
    # position-resolved cache from it (``positions`` stays a plain input).
    attention_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": dense_param_placement(tp=spmd.R)},
    )
    attention_cfg.wq.sharding_config = colwise_config()
    attention_cfg.wk.sharding_config = colwise_config()
    attention_cfg.wv.sharding_config = colwise_config()
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=True)

    attention_cfg.q_norm.sharding_config = _qk_norm_sharding()
    attention_cfg.k_norm.sharding_config = _qk_norm_sharding()

    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)


def _set_deltanet_sharding(
    deltanet_cfg: "GatedDeltaNet.Config",
    *,
    attention_input_layout: SpmdLayout,
) -> None:
    """Sharding for GatedDeltaNet: head-sharded TP on projections.

    Input is allgathered (Shard(1)→Replicate) so that the recurrence
    sees the full sequence. Projections are ColwiseParallel (head-sharded
    output). The FLA kernel runs on local tensors via local_map.
    out_proj is RowwiseParallel (reduce-scatter back to Shard(1)).

    A_log and dt_bias are per-head parameters, Shard(0) on TP.
    Conv1d weights are Shard(0) (out-channels); the DTensor->local conversion
    for the depthwise conv is handled in the model's ``_causal_conv``.
    """
    # ColwiseParallel on all input projections
    for name in (
        "in_proj_q",
        "in_proj_k",
        "in_proj_v",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
    ):
        getattr(deltanet_cfg, name).sharding_config = colwise_config()

    # Depthwise Conv1d weights: Shard(0) on out-channels (head-sharded).
    deltanet_cfg.conv_q.sharding_config = _conv_weight_sharding()
    deltanet_cfg.conv_k.sharding_config = _conv_weight_sharding()
    deltanet_cfg.conv_v.sharding_config = _conv_weight_sharding()

    # RowwiseParallel on output projection (reduce-scatter to SP)
    deltanet_cfg.out_proj.sharding_config = rowwise_config(output_sp=True)

    # Training folds (B, L) to (1, B * L), so tensor DP shards dim 1. Inference
    # already supplies folded tokens and uses separate vLLM DP workers. CP is
    # omitted because Qwen3.5 rejects CP, whose sequence sharding would collide
    # with tensor DP on dim 1.
    deltanet_activation_layout = SpmdLayout({DP: spmd.S(1), TP: spmd.S(2)})

    # RMSNormGated: per-head norm, weight Replicate, activations Shard(2)
    deltanet_cfg.norm.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={
            "x": deltanet_activation_layout,
            "gate": deltanet_activation_layout,
        },
        out_src_shardings=deltanet_activation_layout,
    )

    # GatedDeltaKernel: local_map converts DTensor q/k/v/g/beta to local.
    deltanet_cfg.kernel.sharding_config = ShardingConfig(
        in_src_shardings={
            "xq_BLNK": deltanet_activation_layout,
            "xk_BLNK": deltanet_activation_layout,
            "xv_BLNV": deltanet_activation_layout,
            "g_BLN": deltanet_activation_layout,
            "beta_BLN": deltanet_activation_layout,
        },
        out_src_shardings=deltanet_activation_layout,
        local_map=LocalMapConfig(
            in_grad_placements=(deltanet_activation_layout,) * 5,
        ),
    )

    deltanet_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "A_log": dense_param_placement(tp=spmd.S(0)),
            "dt_bias": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"x_BLD": attention_input_layout},
        in_dst_shardings={"x_BLD": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_sequence_parallel_placement(),
    )
