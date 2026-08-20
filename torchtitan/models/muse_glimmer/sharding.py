# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd
import torch

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
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
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

if TYPE_CHECKING:
    from .model import MuseGlimmerModel, MuseGlimmerTransformerBlock
    from .vision_encoder import MuseGlimmerVisionAdapter, MuseGlimmerVisionEncoder


DP = MeshAxisName.DP
TP = MeshAxisName.TP


def annotate_muse_glimmer_input_spmd_types(
    *,
    pixel_values: torch.Tensor | None,
    grid_thw: torch.Tensor | None,
) -> None:
    """Annotate Muse Glimmer multimodal inputs with their local SPMD types.

    Called inside ``MuseGlimmerModel.multimodal_context`` (a DP-local mesh), so
    the pixel/grid tensors carry per-rank (``V@DP``) types the vision encoder can
    consume. Mirrors kimi_k2_7's ``annotate_multimodal_input_spmd_types``.
    """
    multimodal_type = {
        MeshAxisName.DP: spmd.V,
        MeshAxisName.TP: spmd.I,
    }
    for tensor in (pixel_values, grid_thw):
        if tensor is not None:
            spmd.assert_type(tensor, multimodal_type)


def set_muse_glimmer_sharding_config(
    config: "MuseGlimmerModel.Config",
    *,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on all Muse Glimmer sub-configs.

    The text-only and multimodal models share the same decoder sharding; they
    differ only in how the token embeddings flow into the first decoder layer:

    * **Text-only**: the base decoder sharding is the whole story. The token
      embeddings emit sequence-parallel (``Shard(1)``) activations that flow
      straight into the decoder layers.
    * **Multimodal** (``config.vision_encoder is not None``): ``MuseGlimmerModel.forward``
      scatters vision features into the token embeddings over the full
      ``[batch, seq]`` *between* ``tok_embeddings`` and the decoder layers, so the
      embedding output must be ``Replicate`` -- not ``Shard(1)``/SP.
      :func:`_set_multimodal_sharding` overrides the embedding + norm (and the
      vision-injection modules) to ``Replicate``, and the layer loop gives the
      first decoder layer a ``Replicate`` input that its attention reduce-scatters
      back to SP. Everything else is identical to the text path.

    All sub-configs are populated unconditionally -- ``Module.parallelize``
    filters disabled axes at runtime.
    """
    # Base sharding shared by both models: decoder roots, token embeddings, layers.
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    _set_tok_embeddings_sharding(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_muse_glimmer_layer_sharding(layer_cfg, enable_sp=enable_sp)

    # Multimodal-only override: re-point the embedding (and the first decoder
    # layer) at Replicate so the vision scatter can index the full sequence.
    # No-op for the text model.
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
    """Override the text-path sharding for the multimodal (vision) model.

    ``MuseGlimmerModel.forward`` scatters vision features into the token embeddings
    (masked index over the full ``[batch, seq]``) between ``tok_embeddings`` and
    the decoder layers, so that activation must be ``Replicate`` -- not
    ``Shard(1)``/SP. This re-points the embedding children at ``Replicate``
    outputs, marks the vision-injection modules ``Replicate`` so the whole vision
    path stays DTensor-consistent, and (under SP) re-shards the first decoder
    layer to take that ``Replicate`` input -- its rowwise ``wo`` reduce-scatters
    back to ``Shard(1)``, restoring SP activations for every later layer. Mirrors
    qwen3_5's multimodal sharding overrides.
    """
    replicate = dense_activation_placement(tp=spmd.R)
    # The vision encoder + adapter emit TP-invariant activations (the common
    # vision_encoder_sharding helpers flow {DP: V, TP: I}). The LLM-side injection
    # modules only promote the TP axis I->R so the features become TP-Replicate for
    # the scatter; the DP axis stays V (per-image, local under multimodal_context).
    # DP is deliberately NOT redistributed to S(0): config-based redistribution
    # cannot move an axis to/from spmd.V, so (like qwen3_5's scatter helper) the
    # raw scatter below writes the V vision rows into the S(0) text positions
    # per-rank instead.
    vision_invariant = SpmdLayout({DP: spmd.V, TP: spmd.I})
    vision_tp_replicate = SpmdLayout({DP: spmd.V, TP: spmd.R})
    emb_cfg = config.tok_embeddings

    # Embedding output Replicate (vs Shard(1)/SP): the vision scatter needs the
    # full sequence. Vocab-parallel Embedding.forward runs a manual local masked
    # lookup on a Shard(0) weight and emits a Partial sum; local_map localizes the
    # Replicate DTensor input so the manual path sees a plain tensor (otherwise it
    # mixes a DTensor input with the local weight), and the Partial output is
    # all-reduced to Replicate. Mirrors the text-path tok_embeddings config in
    # set_decoder_sharding_config.
    emb_cfg.embedding.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": replicate},
        in_dst_shardings={"input": replicate},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=replicate,
        local_map=LocalMapConfig(in_grad_placements=None),
    )
    emb_cfg.norm.sharding_config = ShardingConfig(
        in_src_shardings={"input": replicate},
        in_dst_shardings={"input": replicate},
        out_src_shardings=replicate,
        out_dst_shardings=replicate,
    )

    # LLM-side vision injection (vision_projection + perception_emb_norm) consumes
    # the adapter output, which flows TP-invariant ({DP: V, TP: I}). vision_projection
    # promotes the TP axis I->R (single-axis) so the features are TP-Replicate; the
    # norm keeps them there. The DP axis stays V through both, matching the vision
    # features that the scatter writes into the Replicate text stream.
    if config.vision_projection is not None:
        config.vision_projection.sharding_config = ShardingConfig(
            state_shardings={
                "weight": dense_param_placement(tp=spmd.R),
                # Harmless no-op if the projection has no bias.
                "bias": dense_param_placement(tp=spmd.R),
            },
            in_src_shardings={"input": vision_invariant},
            in_dst_shardings={"input": vision_tp_replicate},
            out_src_shardings=vision_tp_replicate,
        )
    if config.perception_emb_norm is not None:
        config.perception_emb_norm.sharding_config = ShardingConfig(
            in_src_shardings={"input": vision_tp_replicate},
            in_dst_shardings={"input": vision_tp_replicate},
            out_src_shardings=vision_tp_replicate,
        )

    # First-layer SP bridge: tok_embeddings now emits Replicate activations (the
    # vision scatter needs the full sequence), but the decoder blocks run
    # sequence-parallel (Shard(1)). Give the first block a block-level sharding
    # config that redistributes its Replicate input to SP at the block boundary,
    # so the block internals (and the residual around attention) are uniformly SP.
    # Every later block already receives SP from the previous block's SP output.
    # Only needed under SP -- without SP the whole decoder uses Replicate
    # activations. Mirrors qwen3_5's per-layer x_BLD redistribution.
    if enable_sp and config.layers:
        config.layers[0].sharding_config = ShardingConfig(
            in_src_shardings={"x": replicate},
            in_dst_shardings={"x": dense_sequence_parallel_placement()},
            out_src_shardings=dense_sequence_parallel_placement(),
        )


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
        else dense_activation_placement(tp=spmd.I)
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

    # QK norms: shard on head dim (dim=2), independent of SP. Scaleless, so no
    # weight state to distribute.
    if attention.qk_norm is not None:
        head_shard = dense_activation_placement(tp=spmd.S(2))
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

    All vision activations flow as TP-invariant (no sequence parallelism), exactly
    like qwen3_5's vision encoder. The shared block/linear/norm helpers in
    :mod:`torchtitan.models.common.vision_encoder_sharding` carry the actual TP
    sharding (colwise q/k/v + rowwise proj, colwise fc1 + rowwise fc2, invariant
    norms, inner-attention local_map); only the Muse-Glimmer-specific learned
    positional grid, RoPE frequencies, and patch ``conv1`` are declared here.

    Must be called BEFORE the configs are built (``config.build()``): the built
    modules copy these configs into ``Module.parallelize``.
    """
    # The learned positional grid is a raw nn.Parameter on the encoder; keep it
    # Replicate (it is bilinearly resampled per image, see _get_pos_emb).
    encoder_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "positional_embedding_vlm": SpmdLayout({DP: spmd.R, TP: spmd.I}),
        },
    )
    encoder_cfg.rope_freq.sharding_config = ShardingConfig(
        state_shardings={
            "inv_freq": SpmdLayout({DP: spmd.R, TP: spmd.I}),
        },
    )

    # conv1 builds ``self.conv1_linear``; sharding goes on the *config* field
    # ``conv1``. Plain pixel patches enter invariant; the (bias-free) weight stays
    # Replicate. Mirrors qwen3_5's patch_embed_proj (vision_invariant_linear_config).
    encoder_cfg.conv1.sharding_config = vision_invariant_linear_config()
    encoder_cfg.ln_pre.sharding_config = invariant_norm_config()
    encoder_cfg.ln_post.sharding_config = invariant_norm_config()

    # Per-block TP via the shared helper (norms, q/k/v/proj, fc1/fc2, and the
    # inner-attention local_map), same as qwen3_5/kimi_k2_7. ``rope_cache`` is a
    # per-image vision activation, so it flows {DP: V, TP: I} like kimi_k2_7.
    set_vision_transformer_block_sharding_config(
        encoder_cfg.block,
        rope_cache_dp=spmd.V,
    )

    if adapter_cfg is not None:
        # The adapter runs on the flattened 2D [tokens, dim] vision features (the
        # encoder cats per-image tokens), so the block helpers' fixed Shard(2)
        # layout is out of bounds. Keep both linears TP-invariant (dimension-
        # agnostic); the adapter output then stays {DP: V, TP: I}, matching the
        # LLM-side vision_projection input.
        adapter_cfg.c_fc.sharding_config = vision_invariant_linear_config()
        adapter_cfg.c_proj.sharding_config = vision_invariant_linear_config()
