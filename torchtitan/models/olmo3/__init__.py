# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    CosSinRoPE,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.attention import VarlenAttention
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.utils import validate_converter_order

from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import Olmo3Attention, Olmo3Model, Olmo3TransformerBlock
from .parallelize import parallelize_olmo3
from .state_dict_adapter import Olmo3StateDictAdapter

__all__ = [
    "parallelize_olmo3",
    "Olmo3Model",
    "olmo3_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}

_EPS = 1e-6


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


def _olmo3_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _make_olmo3_attention_config(**kwargs) -> Olmo3Attention.Config:
    """Build via ``make_gqa_config`` (reusing its fused-qkv param init logic),
    then re-tag the result as ``Olmo3Attention.Config`` so ``.build()``
    constructs ``Olmo3Attention`` (full-width QK-norm) instead of the base
    ``GQAttention``.
    """
    gqa_config = make_gqa_config(**kwargs)
    return Olmo3Attention.Config(
        **{f.name: getattr(gqa_config, f.name) for f in dataclasses.fields(gqa_config)}
    )


def _build_olmo3_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    sliding_window: int | None,
    full_attention_interval: int,
    fuse_qkv: bool = True,
    attn_backend: str,
    rope: RoPE.Config,
) -> list[TransformerBlock.Config]:
    """Build per-layer TransformerBlock configs with the Olmo3 hybrid
    sliding-window / full-attention pattern: a full-attention layer every
    ``full_attention_interval`` layers (1-indexed), sliding-window elsewhere.
    """
    inner_attention = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        is_full_attention = (layer_id + 1) % full_attention_interval == 0
        layer_sliding_window = None if is_full_attention else sliding_window

        layer_inner_attention = inner_attention
        if layer_sliding_window is not None and isinstance(
            inner_attention, VarlenAttention.Config
        ):
            layer_inner_attention = dataclasses.replace(
                inner_attention, window_size=(layer_sliding_window - 1, 0)
            )

        layers.append(
            Olmo3TransformerBlock.Config(
                attention_norm=_olmo3_norm(dim),
                ffn_norm=_olmo3_norm(dim),
                attention=_make_olmo3_attention_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=layer_inner_attention,
                    fuse_qkv=fuse_qkv,
                    rope=rope,
                    # Olmo2/Olmo3 QK-norm normalizes the *full* concatenated
                    # projection (n_heads*head_dim), not per-head -- see
                    # Olmo3Attention. Only valid when n_heads == n_kv_heads.
                    qk_norm=_olmo3_norm(n_heads * head_dim),
                    sliding_window_size=layer_sliding_window,
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


def _debugmodel(attn_backend: str) -> Olmo3Model.Config:
    dim = 256
    n_heads = 16
    # Olmo3Attention shares one qk_norm config for q_norm/k_norm (see
    # Olmo3Attention), which only supports MHA (n_heads == n_kv_heads).
    n_kv_heads = 16
    head_dim = dim // n_heads
    n_layers = 8
    vocab_size = 2048
    return Olmo3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=_olmo3_norm(dim),
        lm_head=Linear.Config(
            in_features=dim, out_features=vocab_size, param_init=_output_linear_init(dim)
        ),
        layers=_build_olmo3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=768,
            sliding_window=64,
            full_attention_interval=4,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_seq_len=4096,
                theta=500000.0,
            ),
        ),
    )


def _7b(attn_backend: str) -> Olmo3Model.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 32
    head_dim = dim // n_heads
    n_layers = 32
    vocab_size = 100278
    return Olmo3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=_olmo3_norm(dim),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_olmo3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=11008,
            sliding_window=4096,
            full_attention_interval=4,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_seq_len=8192,
                theta=500000.0,
            ),
        ),
    )


olmo3_configs = {
    "debugmodel": _debugmodel,
    "7B": _7b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = olmo3_configs[flavor](attn_backend=attn_backend)
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)
    return ModelSpec(
        name="olmo3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_olmo3,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Olmo3StateDictAdapter,
    )
