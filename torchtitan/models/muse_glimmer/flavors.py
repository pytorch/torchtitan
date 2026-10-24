# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer model flavors."""

import dataclasses
import math
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)

from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RowParallelLinear
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    QKVLinear,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import (
    fused_qkv_param_init,
    get_attention_config,
    make_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_input_colwise_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.muse_spark.flavors import (
    muse_spark_navit_adapter_config,
    muse_spark_navit_configs,
    muse_spark_navit_encoder_config,
)
from torchtitan.models.muse_spark.vision_encoder import (
    MuseSparkNaViTAdapter,
    NaViTEncoder,
)

from .model import (
    Attention,
    EmbeddingWithNorm,
    MuseGlimmerModel,
    MuseGlimmerTransformerBlock,
    RMSGainCenterNorm,
    SoftCappedLinear,
)
from .sharding import (
    attention_sharding_config,
    cp_replicated_vision_sharding_config,
    perception_emb_norm_sharding_config,
    scaleless_head_norm_sharding_config,
)

__all__ = [
    "MuseGlimmerModel",
    "MODEL_FLAVORS",
    "build_model_config",
    "muse_glimmer_vision_configs",
]


# The query is scaled by this tuned constant divided by sqrt(head_dim) after
# q-norm.
_SCALE_QUERY_NUMERATOR = 43.7840518911

_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
# Gain-centered norms init their learnable weight to 0 (effective scale is
# weight + gain_center).
_GAIN_NORM_INIT: dict[str, Callable] = {"weight": nn.init.zeros_}
_EMBEDDING_INIT: dict[str, Callable] = {"weight": partial(nn.init.normal_, std=1.0)}

_NORM_EPS = 1e-5
_POST_NORM_EPS = 1e-8
_ROPE_THETA = 500_000.0
_EVERY_N_LAYERS_NOPE = 4
_FFN_DIM_MULTIPLIER = 3.0


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _gain_norm(
    dim: int,
    eps: float,
    gain_center: float,
    *,
    sharding_config: ShardingConfig,
) -> RMSGainCenterNorm.Config:
    return RMSGainCenterNorm.Config(
        normalized_shape=dim,
        eps=eps,
        gain_center=gain_center,
        param_init=_GAIN_NORM_INIT,
        sharding_config=sharding_config,
    )


def _scaleless_norm(
    dim: int, eps: float, *, sharding_config: ShardingConfig
) -> RMSNorm.Config:
    # No learnable scale (QK norm and token-embedding norm).
    return RMSNorm.Config(
        normalized_shape=dim,
        eps=eps,
        elementwise_affine=False,
        sharding_config=sharding_config,
    )


def _layer_use_rope(layer_id: int, n_layers: int) -> bool:
    # iRoPE: NoPE layers counted backward from the last layer.
    return (n_layers - layer_id - 1) % _EVERY_N_LAYERS_NOPE != 0


def _layer_window_size(layer_id: int, n_layers: int, pattern: list[int]) -> int | None:
    # Cyclic SWA/global pattern, aligned to the iRoPE phase (count backward).
    count_backward = layer_id + _EVERY_N_LAYERS_NOPE - n_layers % _EVERY_N_LAYERS_NOPE
    cfg_val = pattern[count_backward % len(pattern)]
    return cfg_val if cfg_val > 0 else None


def _build_muse_glimmer_attention(
    *,
    layer_id: int,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    max_context_length: int,
    window_pattern: list[int],
    attn_backend: str,
    enable_sp: bool,
) -> Attention.Config:
    # Attention.Config mirrors GQAttention.Config plus Muse Glimmer-specific fields
    # (scale_query_by, o_gate, per-layer window_size).
    window = _layer_window_size(layer_id, n_layers, window_pattern)
    inner_attention = get_attention_config(attn_backend)
    if window is not None and isinstance(inner_attention, VarlenInnerAttention.Config):
        inner_attention = SlidingWindowVarlenInnerAttention.Config(
            **{
                f.name: getattr(inner_attention, f.name)
                for f in dataclasses.fields(inner_attention)
            },
            sliding_window_size=window,
        )
    elif window is not None and isinstance(inner_attention, FlexInnerAttention.Config):
        inner_attention = SlidingWindowFlexInnerAttention.Config(
            block_size=inner_attention.block_size,
            kernel_options=inner_attention.kernel_options,
            sliding_window_size=window,
        )
    inner_attention = dataclasses.replace(
        inner_attention, sharding_config=inner_attention_local_spmd_sharding_config()
    )
    return Attention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=QKVLinear.Config(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            wqkv=Linear.Config(
                in_features=dim,
                out_features=(n_heads + 2 * n_kv_heads) * head_dim,
                param_init=fused_qkv_param_init(
                    _LINEAR_INIT,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                ),
                sharding_config=replicated_input_colwise_sharding_config(),
            ),
        ),
        wo=RowParallelLinear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_sharding_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        qk_norm=_scaleless_norm(
            head_dim, _NORM_EPS, sharding_config=scaleless_head_norm_sharding_config()
        ),
        inner_attention=inner_attention,
        rope=ComplexRoPE.Config(
            dim=head_dim,
            max_context_length=max_context_length,
            theta=_ROPE_THETA,
            sharding_config=rope_cache_sharding_config(),
        )
        if _layer_use_rope(layer_id, n_layers)
        else None,
        scale_query_by=_SCALE_QUERY_NUMERATOR / math.sqrt(head_dim),
        o_gate=Linear.Config(
            in_features=dim,
            out_features=n_heads * head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        window_size=window,
        sharding_config=attention_sharding_config(enable_sp=enable_sp),
    )


def _build_muse_glimmer_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    max_context_length: int,
    window_pattern: list[int],
    attn_backend: str,
    enable_sp: bool,
) -> list[MuseGlimmerTransformerBlock.Config]:
    hidden_dim = int(_FFN_DIM_MULTIPLIER * dim)
    # All four norms operate on the residual activation.
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            MuseGlimmerTransformerBlock.Config(
                attention_norm=_gain_norm(
                    dim, _NORM_EPS, gain_center=1.0, sharding_config=norm_sharding
                ),
                ffn_norm=_gain_norm(
                    dim, _NORM_EPS, gain_center=1.0, sharding_config=norm_sharding
                ),
                post_attention_norm=_gain_norm(
                    dim,
                    _POST_NORM_EPS,
                    gain_center=1.0,
                    sharding_config=norm_sharding,
                ),
                post_ffn_norm=_gain_norm(
                    dim,
                    _POST_NORM_EPS,
                    gain_center=1.0,
                    sharding_config=norm_sharding,
                ),
                attention=_build_muse_glimmer_attention(
                    layer_id=layer_id,
                    n_layers=n_layers,
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    max_context_length=max_context_length,
                    window_pattern=window_pattern,
                    attn_backend=attn_backend,
                    enable_sp=enable_sp,
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w13_param_init=_LINEAR_INIT,
                    w2_param_init=_depth_init(layer_id),
                    enable_sp=enable_sp,
                ),
            )
        )
    return layers


def _configure_glimmer_vision(
    encoder: NaViTEncoder.Config, adapter: MuseSparkNaViTAdapter.Config
) -> None:
    encoder.pos_emb_grid_order = "raster"
    for config in (encoder, adapter):
        for _, linear_config, _, _ in config.traverse(Linear.Config):
            linear_config.param_init = _LINEAR_INIT


def muse_glimmer_vision_configs(
    vision_adapter_dim: int = 4096,
) -> tuple[NaViTEncoder.Config, MuseSparkNaViTAdapter.Config]:
    """Build the production vision tower with Glimmer's raster positional grid."""
    encoder, adapter = muse_spark_navit_configs(vision_adapter_dim)
    _configure_glimmer_vision(encoder, adapter)
    return encoder, adapter


def _muse_glimmer_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    max_context_length: int,
    window_pattern: list[int],
    output_multiplier: float,
    attn_backend: str,
    enable_sp: bool,
    vision_adapter_dim: int | None = None,
    vision_encoder: NaViTEncoder.Config | None = None,
    vision_adapter: MuseSparkNaViTAdapter.Config | None = None,
) -> MuseGlimmerModel.Config:
    # LLM-side multimodal injection: project adapter output -> model dim, then
    # apply a scaleless norm. Left as None for the text-only model.
    vision_projection = None
    perception_emb_norm = None
    if vision_adapter_dim is not None:
        vision_projection = Linear.Config(
            in_features=vision_adapter_dim,
            out_features=dim,
            param_init=_LINEAR_INIT,
            sharding_config=cp_replicated_vision_sharding_config(),
        )
        perception_emb_norm = _scaleless_norm(
            dim,
            _NORM_EPS,
            sharding_config=perception_emb_norm_sharding_config(enable_sp=enable_sp),
        )

    return MuseGlimmerModel.Config(
        max_context_length=max_context_length,
        dim=dim,
        vocab_size=vocab_size,
        # Token embedding bundled with its scaleless norm so the norm travels
        # with the embedding under pipeline-parallel module splitting. The
        # embedding sharding sits on the inner embedding so the vocab-parallel
        # all-reduce happens before the scaleless norm (normalizing a Partial
        # sum would be wrong); the norm runs on the reduced activation.
        tok_embeddings=EmbeddingWithNorm.Config(
            embedding=Embedding.Config(
                num_embeddings=vocab_size,
                embedding_dim=dim,
                param_init=_EMBEDDING_INIT,
                sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
            ),
            # Scaleless: norm_sharding_config's weight declaration is ignored since
            # there is no weight param.
            norm=_scaleless_norm(
                dim,
                _NORM_EPS,
                sharding_config=norm_sharding_config(enable_sp=enable_sp),
            ),
        ),
        # lm_head applies the output multiplier + tanh soft-cap so the transform
        # runs wherever lm_head runs (full forward or per-chunk ChunkedLossWrapper).
        lm_head=SoftCappedLinear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            output_multiplier=output_multiplier,
            output_soft_cap_temp=20.0,
            sharding_config=lm_head_sharding_config(),
        ),
        # Keep the checkpoint's zero gain center, but initialize to unit scale.
        norm=dataclasses.replace(
            _gain_norm(
                dim,
                _NORM_EPS,
                gain_center=0.0,
                sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
            ),
            param_init={"weight": nn.init.ones_},
        ),
        layers=_build_muse_glimmer_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_context_length=max_context_length,
            window_pattern=window_pattern,
            attn_backend=attn_backend,
            enable_sp=enable_sp,
        ),
        vision_projection=vision_projection,
        perception_emb_norm=perception_emb_norm,
        vision_encoder=vision_encoder,
        vision_adapter=vision_adapter,
    )


def _debugmodel(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> MuseGlimmerModel.Config:
    return _muse_glimmer_config(
        dim=256,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        vocab_size=2048,
        max_context_length=seq_len,
        window_pattern=[128, 128, 128, 0],
        output_multiplier=1.0,
        attn_backend=attn_backend,
        enable_sp=parallelism.enable_sequence_parallel,
    )


def _muse_glimmer_30b(
    attn_backend: str,
    *,
    with_vision: bool = False,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> MuseGlimmerModel.Config:
    vision_adapter_dim = None
    vision_encoder = None
    vision_adapter = None
    if with_vision:
        vision_adapter_dim = 4096
        vision_encoder, vision_adapter = muse_glimmer_vision_configs(
            vision_adapter_dim=vision_adapter_dim
        )
    return _muse_glimmer_config(
        dim=6656,
        n_layers=52,
        n_heads=32,
        n_kv_heads=2,
        head_dim=128,
        vocab_size=202048,
        max_context_length=seq_len,
        window_pattern=[2048, 2048, 2048, 0],
        output_multiplier=0.19611613513,
        attn_backend=attn_backend,
        enable_sp=parallelism.enable_sequence_parallel,
        vision_adapter_dim=vision_adapter_dim,
        vision_encoder=vision_encoder,
        vision_adapter=vision_adapter,
    )


def _muse_glimmer_debugmodel_mm(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> MuseGlimmerModel.Config:
    """Multimodal debug flavor: the debug text decoder that *owns* a scaled-down
    vision encoder + adapter and runs them inside ``forward``.

    The encoder is small (few layers, ``num_heads`` divisible by intended TP) and
    its ``output_dim`` (``latent_dim * downsample_factor**2 = 256``) feeds the
    adapter, whose ``output_dim`` equals ``vision_adapter_dim`` so the LLM-side
    ``vision_projection`` (``vision_adapter_dim -> dim``) lines up.
    """
    encoder = muse_spark_navit_encoder_config(
        latent_dim=64,
        num_layers=3,
        num_heads=4,
        mlp_hidden_dim=128,
        downsample_factor=2,
        sparse_attention_factor=2,
        pos_emb_grid_h=2,
        pos_emb_grid_w=2,
    )
    adapter_dim = 128
    adapter = muse_spark_navit_adapter_config(
        input_dim=encoder.output_dim, output_dim=adapter_dim
    )
    _configure_glimmer_vision(encoder, adapter)
    return _muse_glimmer_config(
        dim=256,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        vocab_size=2048,
        max_context_length=seq_len,
        window_pattern=[128, 128, 128, 0],
        output_multiplier=1.0,
        attn_backend=attn_backend,
        enable_sp=parallelism.enable_sequence_parallel,
        vision_adapter_dim=adapter_dim,
        vision_encoder=encoder,
        vision_adapter=adapter,
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 4096),
    "30B": (_muse_glimmer_30b, 16384),
    "debugmodel_mm": (_muse_glimmer_debugmodel_mm, 4096),
    "30B_mm": (partial(_muse_glimmer_30b, with_vision=True), 16384),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> MuseGlimmerModel.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        seq_len=context_len,
        parallelism=parallelism,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)
    return config
