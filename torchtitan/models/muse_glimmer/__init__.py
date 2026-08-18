# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    Linear,
    ScaledBiasRowwiseLinear,
)
from torchtitan.models.common.attention import QKVLinear, VarlenAttention
from torchtitan.models.common.config_utils import get_attention_config, make_ffn_config
from torchtitan.models.common.nn_modules import GELU, LayerNorm, RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    Attention,
    EmbeddingWithNorm,
    MuseGlimmerModel,
    MuseGlimmerTransformerBlock,
    RMSGainCenterNorm,
    SoftCappedLinear,
)
from .parallelize import parallelize_muse_glimmer, pipeline_muse_glimmer
from .sharding import set_muse_glimmer_vision_sharding_config
from .state_dict_adapter import MuseGlimmerStateDictAdapter
from .vision_encoder import (
    MuseGlimmerVisionAdapter,
    MuseGlimmerVisionEncoder,
    VisionRopeFreq,
)

__all__ = [
    "parallelize_muse_glimmer",
    "pipeline_muse_glimmer",
    "set_muse_glimmer_vision_sharding_config",
    "MuseGlimmerModel",
    "muse_glimmer_configs",
    "model_registry",
    "MuseGlimmerVisionEncoder",
    "MuseGlimmerVisionAdapter",
    "VisionRopeFreq",
    "muse_glimmer_vision_encoder_config",
    "muse_glimmer_vision_adapter_config",
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
_VISION_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_POS_EMB_INIT: dict[str, Callable] = {
    "positional_embedding_vlm": partial(nn.init.trunc_normal_, std=0.02)
}

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


def _gain_norm(dim: int, eps: float, gain_center: float) -> RMSGainCenterNorm.Config:
    return RMSGainCenterNorm.Config(
        normalized_shape=dim,
        eps=eps,
        gain_center=gain_center,
        param_init=_GAIN_NORM_INIT,
    )


def _scaleless_norm(dim: int, eps: float) -> RMSNorm.Config:
    # No learnable scale (QK norm and token-embedding norm).
    return RMSNorm.Config(normalized_shape=dim, eps=eps, elementwise_affine=False)


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
    max_seq_len: int,
    window_pattern: list[int],
    attn_backend: str,
) -> Attention.Config:
    # Attention.Config mirrors GQAttention.Config plus Muse Glimmer-specific fields
    # (scale_query_by, o_gate, per-layer window_size).
    window = _layer_window_size(layer_id, n_layers, window_pattern)
    inner_attention = get_attention_config(attn_backend)
    # Varlen carries the per-layer sliding window as an FA3 kernel arg (mirrors
    # gpt_oss). The flex path instead selects a window-keyed BlockMask in
    # Attention.forward, so it leaves inner_attention's window at the default.
    if window is not None and isinstance(inner_attention, VarlenAttention.Config):
        inner_attention = dataclasses.replace(
            inner_attention, window_size=(window - 1, 0)
        )
    return Attention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=QKVLinear.Config(
            head_dim=head_dim,
            wq=Linear.Config(
                in_features=dim,
                out_features=n_heads * head_dim,
                param_init=_LINEAR_INIT,
            ),
            wkv=Linear.Config(
                in_features=dim,
                out_features=n_kv_heads * head_dim,
                param_init=_LINEAR_INIT,
            ),
        ),
        wo=Linear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        qk_norm=_scaleless_norm(head_dim, _NORM_EPS),
        use_rope=_layer_use_rope(layer_id, n_layers),
        inner_attention=inner_attention,
        # Every layer (incl. NoPE) carries a rope config so the base Decoder's
        # max_seq_len discovery/resize works uniformly; NoPE layers simply never
        # apply it (guarded by use_rope in Attention.forward).
        rope=ComplexRoPE.Config(
            dim=head_dim,
            max_seq_len=max_seq_len,
            theta=_ROPE_THETA,
        ),
        scale_query_by=_SCALE_QUERY_NUMERATOR / math.sqrt(head_dim),
        o_gate=Linear.Config(
            in_features=dim,
            out_features=n_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        window_size=window,
    )


def _build_muse_glimmer_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    window_pattern: list[int],
    attn_backend: str,
) -> list[MuseGlimmerTransformerBlock.Config]:
    hidden_dim = int(_FFN_DIM_MULTIPLIER * dim)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            MuseGlimmerTransformerBlock.Config(
                attention_norm=_gain_norm(dim, _NORM_EPS, gain_center=1.0),
                ffn_norm=_gain_norm(dim, _NORM_EPS, gain_center=1.0),
                post_attention_norm=_gain_norm(dim, _POST_NORM_EPS, gain_center=1.0),
                post_ffn_norm=_gain_norm(dim, _POST_NORM_EPS, gain_center=1.0),
                attention=_build_muse_glimmer_attention(
                    layer_id=layer_id,
                    n_layers=n_layers,
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    max_seq_len=max_seq_len,
                    window_pattern=window_pattern,
                    attn_backend=attn_backend,
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _vision_layer_norm(dim: int) -> LayerNorm.Config:
    return LayerNorm.Config(normalized_shape=dim, eps=_NORM_EPS)


def _vision_linear(in_features: int, out_features: int, *, bias: bool) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        param_init=_VISION_LINEAR_INIT,
    )


def _vision_scaled_bias_rowwise_linear(
    in_features: int, out_features: int
) -> ScaledBiasRowwiseLinear.Config:
    return ScaledBiasRowwiseLinear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_VISION_LINEAR_INIT,
    )


def muse_glimmer_vision_encoder_config(
    *,
    latent_dim: int = 1536,
    num_layers: int = 50,
    num_heads: int = 16,
    mlp_ratio: float = 8960 / 1536,
    patch_size: int = 14,
    patch_temporal: int = 2,
    downsample_factor: int = 2,
    sparse_attention_factor: int = 4,
    pos_emb_grid_h: int = 32,
    pos_emb_grid_w: int = 32,
    rope_theta: float = 10000.0,
) -> MuseGlimmerVisionEncoder.Config:
    head_dim = latent_dim // num_heads
    mlp_hidden = int(mlp_ratio * latent_dim)
    output_dim = latent_dim * downsample_factor**2
    patch_dim = patch_temporal * 3 * patch_size * patch_size
    return MuseGlimmerVisionEncoder.Config(
        latent_dim=latent_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        patch_size=patch_size,
        patch_temporal=patch_temporal,
        downsample_factor=downsample_factor,
        sparse_attention_factor=sparse_attention_factor,
        pos_emb_grid_h=pos_emb_grid_h,
        pos_emb_grid_w=pos_emb_grid_w,
        rope_freq=VisionRopeFreq.Config(
            head_dim=head_dim,
            rope_theta=rope_theta,
        ),
        param_init=_POS_EMB_INIT,
        conv1=_vision_linear(patch_dim, latent_dim, bias=False),
        ln_pre=_vision_layer_norm(latent_dim),
        block=VisionTransformerBlock.Config(
            norm1=_vision_layer_norm(latent_dim),
            attn=VisionAttention.Config(
                dim=latent_dim,
                num_heads=num_heads,
                wq=_vision_linear(latent_dim, num_heads * head_dim, bias=True),
                wk=_vision_linear(latent_dim, num_heads * head_dim, bias=True),
                wv=_vision_linear(latent_dim, num_heads * head_dim, bias=True),
                proj=_vision_scaled_bias_rowwise_linear(
                    num_heads * head_dim, latent_dim
                ),
            ),
            norm2=_vision_layer_norm(latent_dim),
            mlp=VisionMLP.Config(
                fc1=_vision_linear(latent_dim, mlp_hidden, bias=True),
                fc2=_vision_scaled_bias_rowwise_linear(mlp_hidden, latent_dim),
                act_fn=GELU.Config(approximate="none"),
            ),
        ),
        ln_post=_vision_layer_norm(latent_dim),
    )


def muse_glimmer_vision_adapter_config(
    *,
    input_dim: int = 6144,
    output_dim: int = 4096,
) -> MuseGlimmerVisionAdapter.Config:
    return MuseGlimmerVisionAdapter.Config(
        c_fc=_vision_linear(input_dim, output_dim, bias=False),
        c_proj=_vision_linear(output_dim, output_dim, bias=False),
    )


def muse_glimmer_vision_configs(
    vision_adapter_dim: int = 4096,
) -> tuple[MuseGlimmerVisionEncoder.Config, MuseGlimmerVisionAdapter.Config]:
    """Production (30B) vision encoder + adapter configs.

    The adapter maps the encoder ``output_dim`` (latent_dim * downsample^2) to the
    LLM ``vision_adapter_dim`` consumed by ``MuseGlimmerModel.vision_projection``. Pass
    the same ``vision_adapter_dim`` used to build the LLM config so the adapter
    output and ``vision_projection`` in_features cannot drift apart.
    """
    encoder = muse_glimmer_vision_encoder_config()
    adapter = muse_glimmer_vision_adapter_config(
        input_dim=encoder.output_dim, output_dim=vision_adapter_dim
    )
    return encoder, adapter


def _muse_glimmer_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    max_seq_len: int,
    window_pattern: list[int],
    output_multiplier: float,
    attn_backend: str,
    vision_adapter_dim: int | None = None,
    vision_encoder: MuseGlimmerVisionEncoder.Config | None = None,
    vision_adapter: MuseGlimmerVisionAdapter.Config | None = None,
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
        )
        perception_emb_norm = _scaleless_norm(dim, _NORM_EPS)

    # When the model owns the vision stack, fill the encoder/adapter sharding
    # configs so ``model.parallelize`` applies their TP.
    if vision_encoder is not None:
        set_muse_glimmer_vision_sharding_config(vision_encoder, vision_adapter)

    return MuseGlimmerModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        # Token embedding bundled with its scaleless norm so the norm travels
        # with the embedding under pipeline-parallel module splitting.
        tok_embeddings=EmbeddingWithNorm.Config(
            embedding=Embedding.Config(
                num_embeddings=vocab_size,
                embedding_dim=dim,
                param_init=_EMBEDDING_INIT,
            ),
            norm=_scaleless_norm(dim, _NORM_EPS),
        ),
        # lm_head applies the output multiplier + tanh soft-cap so the transform
        # runs wherever lm_head runs (full forward or per-chunk ChunkedLossWrapper).
        lm_head=SoftCappedLinear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            output_multiplier=output_multiplier,
            output_soft_cap_temp=20.0,
        ),
        # Final output norm is gain-centered on 0.0.
        norm=_gain_norm(dim, _NORM_EPS, gain_center=0.0),
        layers=_build_muse_glimmer_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            window_pattern=window_pattern,
            attn_backend=attn_backend,
        ),
        vision_projection=vision_projection,
        perception_emb_norm=perception_emb_norm,
        vision_encoder=vision_encoder,
        vision_adapter=vision_adapter,
    )


def _debugmodel(attn_backend: str) -> MuseGlimmerModel.Config:
    return _muse_glimmer_config(
        dim=256,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        vocab_size=2048,
        max_seq_len=4096,
        window_pattern=[128, 128, 128, 0],
        output_multiplier=1.0,
        attn_backend=attn_backend,
    )


def _muse_glimmer_30b(
    attn_backend: str, *, with_vision: bool = False
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
        max_seq_len=16384,
        window_pattern=[2048, 2048, 2048, 0],
        output_multiplier=0.19611613513,
        attn_backend=attn_backend,
        vision_adapter_dim=vision_adapter_dim,
        vision_encoder=vision_encoder,
        vision_adapter=vision_adapter,
    )


def _muse_glimmer_debugmodel_mm(attn_backend: str) -> MuseGlimmerModel.Config:
    """Multimodal debug flavor: the debug text decoder that *owns* a scaled-down
    vision encoder + adapter and runs them inside ``forward``.

    The encoder is small (few layers, ``num_heads`` divisible by intended TP) and
    its ``output_dim`` (``latent_dim * downsample_factor**2 = 256``) feeds the
    adapter, whose ``output_dim`` equals ``vision_adapter_dim`` so the LLM-side
    ``vision_projection`` (``vision_adapter_dim -> dim``) lines up.
    """
    encoder = muse_glimmer_vision_encoder_config(
        latent_dim=64,
        num_layers=3,
        num_heads=4,
        mlp_ratio=2.0,
        downsample_factor=2,
        sparse_attention_factor=2,
        pos_emb_grid_h=2,
        pos_emb_grid_w=2,
    )
    adapter_dim = 128
    adapter = muse_glimmer_vision_adapter_config(
        input_dim=encoder.output_dim, output_dim=adapter_dim
    )
    return _muse_glimmer_config(
        dim=256,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        vocab_size=2048,
        max_seq_len=4096,
        window_pattern=[128, 128, 128, 0],
        output_multiplier=1.0,
        attn_backend=attn_backend,
        vision_adapter_dim=adapter_dim,
        vision_encoder=encoder,
        vision_adapter=adapter,
    )


muse_glimmer_configs = {
    "debugmodel": _debugmodel,
    "30B": _muse_glimmer_30b,
    "debugmodel_mm": _muse_glimmer_debugmodel_mm,
    "30B_mm": partial(_muse_glimmer_30b, with_vision=True),
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = muse_glimmer_configs[flavor](attn_backend=attn_backend)
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            c.build().convert(config)
    return ModelSpec(
        name="muse_glimmer",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_muse_glimmer,
        pipelining_fn=pipeline_muse_glimmer,
        post_optimizer_build_fn=None,
        state_dict_adapter=MuseGlimmerStateDictAdapter,
    )
