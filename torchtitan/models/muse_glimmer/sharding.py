# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

if TYPE_CHECKING:
    from .model import MuseGlimmerModel, MuseGlimmerTransformerBlock
    from .vision_encoder import MuseGlimmerVisionAdapter, MuseGlimmerVisionEncoder


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
        out_dst_shardings=replicate,
    )

    # LLM-side vision injection (vision_projection + perception_emb_norm) consumes
    # the encoder/adapter output, which flows Replicate. Without a sharding config
    # these modules' params stay plain tensors, so under TP they would mix with the
    # Replicate-DTensor encoder output in the projection matmul. Mark both
    # Replicate so params become Replicate DTensors and the whole vision path stays
    # DTensor-consistent.
    if config.vision_projection is not None:
        config.vision_projection.sharding_config = ShardingConfig(
            state_shardings={
                "weight": dense_param_placement(tp=spmd.R),
                # Harmless no-op if the projection has no bias.
                "bias": dense_param_placement(tp=spmd.R),
            },
            in_src_shardings={"input": replicate},
            in_dst_shardings={"input": replicate},
            out_dst_shardings=replicate,
        )
    if config.perception_emb_norm is not None:
        config.perception_emb_norm.sharding_config = ShardingConfig(
            in_src_shardings={"input": replicate},
            in_dst_shardings={"input": replicate},
            out_dst_shardings=replicate,
        )

    # First-layer SP bridge: tok_embeddings now emits Replicate activations, but the
    # rest of the decoder expects sequence-parallel activations (Shard(1)). Re-shard
    # the first layer to take a Replicate input; its rowwise ``wo`` (output_sp)
    # reduce-scatters back to Shard(1), restoring SP activations for every later layer.
    # Only needed under SP -- without SP the whole decoder already uses Replicate
    # activations. Mirrors qwen3_5's first-layer Replicate input layout.
    if enable_sp and config.layers:
        _set_muse_glimmer_layer_sharding(
            config.layers[0],
            enable_sp=enable_sp,
            attention_input_layout=replicate,
        )


def _set_muse_glimmer_layer_sharding(
    layer_cfg: "MuseGlimmerTransformerBlock.Config",
    *,
    enable_sp: bool,
    attention_input_layout: SpmdLayout | None = None,
) -> None:
    """Set sharding on one decoder block.

    ``attention_input_layout`` is the layout the block's input ``x`` arrives in,
    which the ``attention_norm`` and the attention input must match. It defaults to
    the sequence-parallel activation (Replicate when ``enable_sp=False``) used by
    every layer; the multimodal first layer passes ``Replicate`` instead, since it
    receives the Replicate embedding output and reduce-scatters back to SP via the
    attention's rowwise ``wo`` (see :func:`_set_multimodal_sharding`). Everything
    downstream of the attention (post-attention norm, FFN, residuals) is plain SP
    regardless.
    """
    from .model import Attention

    attention = layer_cfg.attention
    assert isinstance(attention, Attention.Config)

    sp_activation = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    if attention_input_layout is None:
        attention_input_layout = sp_activation

    # attention_norm matches the block input layout; the other three norms always
    # see the sequence-parallel activation that flows after the attention.
    layer_cfg.attention_norm.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": attention_input_layout},
        out_src_shardings=attention_input_layout,
    )
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.ffn_norm.sharding_config = norm
    layer_cfg.post_attention_norm.sharding_config = norm
    layer_cfg.post_ffn_norm.sharding_config = norm

    set_gqa_attention_sharding(attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(attention.inner_attention)
    # Re-point the attention's activation input to this layer's input layout.
    # set_gqa_attention_sharding declares exactly the activation arg in in_src, so
    # overwriting every entry pins the input without hardcoding the arg name (a
    # no-op for non-first layers, where the layout already matches).
    # pyrefly: ignore [missing-attribute]
    attn_in_src = attention.sharding_config.in_src_shardings
    assert attn_in_src is not None
    for arg_name in attn_in_src:
        attn_in_src[arg_name] = attention_input_layout

    # QK norms: shard on head dim (dim=2), independent of SP. Scaleless, so no
    # weight state to distribute.
    if attention.qk_norm is not None:
        head_shard = dense_activation_placement(tp=spmd.S(2))
        attention.qk_norm.sharding_config = ShardingConfig(
            in_src_shardings={"input": head_shard},
            in_dst_shardings={"input": head_shard},
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


def _replicate_norm() -> ShardingConfig:
    """Replicate norm (weight/bias and activations).

    The vision encoder runs without sequence parallelism, so its LayerNorms keep
    weight/bias and activations Replicate (mirrors qwen3_5's vision encoder).
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_dst_shardings=dense_activation_placement(tp=spmd.R),
    )


def set_muse_glimmer_vision_sharding_config(
    encoder_cfg: "MuseGlimmerVisionEncoder.Config",
    adapter_cfg: "MuseGlimmerVisionAdapter.Config | None" = None,
) -> None:
    """Fill ``sharding_config`` on the Muse Glimmer vision encoder (+ optional adapter).

    All vision activations flow as Replicate (no sequence parallelism), exactly
    like qwen3_5's vision encoder. TP shards only the per-block linears
    (colwise q/k/v + rowwise proj, colwise fc1 + rowwise fc2) and the adapter
    linears; norms, ``conv1``, ``positional_embedding_vlm``, and the
    ``rope_cache`` activation stay Replicate.

    Must be called BEFORE the configs are built (``config.build()``): the built
    modules copy these configs into ``Module.parallelize``.
    """
    # The learned positional grid is a raw nn.Parameter on the encoder; keep it
    # Replicate (it is bilinearly resampled per image, see _get_pos_emb).
    encoder_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "positional_embedding_vlm": dense_param_placement(tp=spmd.R),
        },
    )

    # conv1 builds ``self.conv1_linear``; sharding goes on the *config* field
    # ``conv1``. Plain pixel patches enter as DTensor(Replicate); the weight
    # (bias-free) stays Replicate. Mirrors qwen3_5's patch_embed_proj.
    encoder_cfg.conv1.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_dst_shardings=dense_activation_placement(tp=spmd.R),
    )
    encoder_cfg.ln_pre.sharding_config = _replicate_norm()
    encoder_cfg.ln_post.sharding_config = _replicate_norm()

    block = encoder_cfg.block
    block.norm1.sharding_config = _replicate_norm()
    block.norm2.sharding_config = _replicate_norm()

    # rope_cache is a complex Replicate activation: declare it so the Module
    # wrapper wraps the plain tensor as DTensor(Replicate) and never shards it.
    # (rope_apply is a callable and attention_mask is a BlockMask, not tensors,
    # so they pass through.)
    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={"rope_cache": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"rope_cache": dense_activation_placement(tp=spmd.R)},
    )
    block.attn.wq.sharding_config = colwise_config()
    block.attn.wk.sharding_config = colwise_config()
    block.attn.wv.sharding_config = colwise_config()
    block.attn.proj.sharding_config = rowwise_config(output_sp=False)
    set_gqa_inner_attention_local_map(block.attn.inner_attention)

    block.mlp.fc1.sharding_config = colwise_config()
    block.mlp.fc2.sharding_config = rowwise_config(output_sp=False)

    # Stateless leaf modules holding the local-tensor compute that has no DTensor
    # support (grid_sample, advanced indexing). local_map converts their DTensor
    # inputs to local tensors before forward and wraps outputs back to Replicate.
    # All vision activations are Replicate, so every input/output layout is R.
    # (Pixel-shuffle downsampling needs no leaf/local_map: it is pure
    # view/permute/reshape, which runs natively on a Replicate DTensor.)
    _r = dense_activation_placement(tp=spmd.R)
    encoder_cfg.pos_embed.sharding_config = ShardingConfig(
        in_src_shardings={"pos_param": _r},
        in_dst_shardings={"pos_param": _r},
        out_src_shardings=_r,
        local_map=LocalMapConfig(in_grad_placements=(_r,)),
    )
    encoder_cfg.token_permute.sharding_config = ShardingConfig(
        in_src_shardings={"x": _r, "index": _r},
        in_dst_shardings={"x": _r, "index": _r},
        out_src_shardings=_r,
        local_map=LocalMapConfig(in_grad_placements=(_r, _r)),
    )

    if adapter_cfg is not None:
        # Both adapter linears are bias-free; rowwise's bias placement is a
        # harmless no-op when absent.
        adapter_cfg.c_fc.sharding_config = colwise_config()
        adapter_cfg.c_proj.sharding_config = rowwise_config(output_sp=False)
