# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_config,
    set_vision_transformer_block_sharding_config,
    vision_invariant_linear_config,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from .model import MuseGlimmerModel, MuseGlimmerTransformerBlock
    from .vision_encoder import MuseGlimmerVisionAdapter, MuseGlimmerVisionEncoder


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def vision_bank_indices_placement(*, enable_sp: bool) -> SpmdType:
    """Placement for token-aligned indices into the packed vision bank."""
    token_axes = (DP, CP, TP) if enable_sp else (DP, CP)
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V if enable_sp else spmd.I,
        },
        partition_spec=spmd.PartitionSpec(token_axes),
    )


def set_muse_glimmer_sharding_config(
    config: "MuseGlimmerModel.Config",
    *,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on all Muse Glimmer sub-configs.

    Text-only and multimodal models use the standard decoder activation layout.
    With SP, each TP rank gathers vision rows for its local token shard, so the
    vision bank becomes TP-replicated before fusion.

    All sub-configs are populated unconditionally -- ``Module.parallelize``
    filters disabled axes at runtime.
    """
    # Base sharding shared by both models: decoder roots, token embeddings, layers.
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    _set_tok_embeddings_sharding(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_muse_glimmer_layer_sharding(layer_cfg, enable_sp=enable_sp)

    if config.vision_encoder is not None:
        _set_multimodal_sharding(config, enable_sp=enable_sp)


def _set_tok_embeddings_sharding(
    config: "MuseGlimmerModel.Config",
    *,
    enable_sp: bool,
) -> None:
    """Move the decoder embedding sharding onto the ``EmbeddingWithNorm`` children.

    ``tok_embeddings`` is an ``EmbeddingWithNorm`` container:
    ``set_decoder_sharding_config`` placed the embedding sharding on the container,
    but it belongs on the inner embedding child so the vocab-parallel all-reduce
    happens at the embedding boundary -- before the scaleless norm runs
    (normalizing a Partial sum would be wrong). The norm child then runs on the
    already-reduced activation.
    """
    # TODO: verify spmd_types/shard prop raises a hard error if an incorrect
    # sharding config (by running without the following sharding reassignment on the
    # children) lets the norm run on a Partial output. Confirm once we start
    # validating spmd_types with MUSE_GLIMMER.
    # Currently, without the following reassignment, emb_cfg.embedding and
    # emb_cfg.norm do not have sharding at all, so we get a mixed operation error
    # between Tensor and DTensor.
    emb_cfg = config.tok_embeddings
    emb_cfg.embedding.sharding_config = emb_cfg.sharding_config
    emb_cfg.sharding_config = None
    # The inner norm is scaleless (elementwise_affine=False); norm_config's
    # weight declaration is harmlessly ignored since there is no weight param.
    emb_cfg.norm.sharding_config = norm_config(enable_sp=enable_sp)


def _set_multimodal_sharding(
    config: "MuseGlimmerModel.Config",
    *,
    enable_sp: bool,
) -> None:
    """Configure token-local multimodal fusion."""
    if config.vision_projection is not None:
        config.vision_projection.sharding_config = vision_invariant_linear_config(
            include_cp_axis=True
        )
    if config.perception_emb_norm is not None:
        vision_norm = invariant_norm_config(include_cp_axis=True)
        if enable_sp:
            vision_norm.out_dst_shardings = SpmdType(
                {DP: spmd.V, CP: spmd.R, TP: spmd.R}
            )
        config.perception_emb_norm.sharding_config = vision_norm


def _set_muse_glimmer_layer_sharding(
    layer_cfg: "MuseGlimmerTransformerBlock.Config",
    *,
    enable_sp: bool,
) -> None:
    """Set sharding on one decoder block."""
    from .model import Attention

    attention = layer_cfg.attention
    assert isinstance(attention, Attention.Config)

    sp_activation = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    # All four norms operate on the sequence-parallel activation.
    layer_cfg.attention_norm.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": sp_activation},
        out_src_shardings=sp_activation,
    )
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.ffn_norm.sharding_config = norm
    layer_cfg.post_attention_norm.sharding_config = norm
    layer_cfg.post_ffn_norm.sharding_config = norm

    set_gqa_attention_sharding(attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # QK norms: shard on head dim (dim=1), independent of SP. Scaleless, so no
    # weight state to distribute.
    if attention.qk_norm is not None:
        head_shard = attention_activation_placement()
        attention.qk_norm.sharding_config = ShardingConfig(
            in_src_shardings={"input": head_shard},
            in_dst_shardings={"input": head_shard},
            out_src_shardings=head_shard,
            out_dst_shardings=head_shard,
        )

    # Output gate: colwise so its Shard(-1) output aligns with the head-sharded
    # attention output before ``wo``.
    if attention.o_gate is not None:
        attention.o_gate.sharding_config = colwise_config()

    assert layer_cfg.feed_forward is not None
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_layout=sp_activation,
        enable_sp=enable_sp,
    )


def set_muse_glimmer_vision_sharding_config(
    encoder_cfg: "MuseGlimmerVisionEncoder.Config",
    adapter_cfg: "MuseGlimmerVisionAdapter.Config | None" = None,
) -> None:
    """Fill ``sharding_config`` on the Muse Glimmer vision encoder (+ optional adapter).

    Vision activations are invariant across TP and replicated across CP. The
    shared block/linear/norm helpers carry TP sharding and explicit CP layouts;
    only the Muse-specific learned positional grid, RoPE frequencies, patch
    ``conv1``, and local permutation boundaries are declared here.

    Must be called BEFORE the configs are built (``config.build()``): the built
    modules copy these configs into ``Module.parallelize``.
    """
    # The learned positional grid is a raw nn.Parameter on the encoder; keep it
    # Replicate (it is bilinearly resampled per image, see _get_pos_emb).
    encoder_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "positional_embedding_vlm": SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I}),
        },
    )
    encoder_cfg.rope_freq.sharding_config = ShardingConfig(
        state_shardings={
            "inv_freq": SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I}),
        },
    )

    # conv1 builds ``self.conv1_linear``; sharding goes on the *config* field
    # ``conv1``. Plain pixel patches enter invariant; the (bias-free) weight stays
    # Replicate. Mirrors qwen3_5's patch_embed_proj (vision_invariant_linear_config).
    encoder_cfg.conv1.sharding_config = vision_invariant_linear_config(
        include_cp_axis=True
    )
    encoder_cfg.ln_pre.sharding_config = invariant_norm_config(include_cp_axis=True)
    encoder_cfg.ln_post.sharding_config = invariant_norm_config(include_cp_axis=True)

    # Per-block TP via the shared helper (norms, q/k/v/proj, fc1/fc2, and the
    # inner-attention local_map), same as qwen3_5/kimi_k2_7. ``rope_cache`` is a
    # per-image vision activation, so it flows {DP: V, CP: R, TP: I}.
    set_vision_transformer_block_sharding_config(
        encoder_cfg.block,
        rope_cache_dp=spmd.V,
        include_cp_axis=True,
    )

    vision_invariant = SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.I})
    vision_invariant_grad = SpmdType({DP: spmd.V, CP: spmd.P, TP: spmd.I})
    pos_param_invariant = SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I})
    pos_param_grad = SpmdType({DP: spmd.P, CP: spmd.P, TP: spmd.I})
    encoder_cfg.pos_embed.sharding_config = ShardingConfig(
        in_src_shardings={"pos_param": pos_param_invariant},
        in_dst_shardings={"pos_param": pos_param_invariant},
        out_src_shardings=vision_invariant,
        local_map=LocalMapConfig(in_grad_placements=(pos_param_grad,)),
    )
    encoder_cfg.token_permute.sharding_config = ShardingConfig(
        in_src_shardings={"x": vision_invariant, "index": vision_invariant},
        in_dst_shardings={"x": vision_invariant, "index": vision_invariant},
        out_src_shardings=vision_invariant,
        local_map=LocalMapConfig(
            in_grad_placements=(vision_invariant_grad, vision_invariant)
        ),
    )

    if adapter_cfg is not None:
        # The adapter runs on the flattened 2D [tokens, dim] vision features (the
        # encoder cats per-image tokens), so the block helpers' fixed Shard(2)
        # layout is out of bounds. Keep both linears TP-invariant (dimension-
        # agnostic); the adapter output stays {DP: V, CP: R, TP: I}, matching
        # the LLM-side vision_projection input.
        adapter_cfg.c_fc.sharding_config = vision_invariant_linear_config(
            include_cp_axis=True
        )
        adapter_cfg.c_proj.sharding_config = vision_invariant_linear_config(
            include_cp_axis=True
        )
