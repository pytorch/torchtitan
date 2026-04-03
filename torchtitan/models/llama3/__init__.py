# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from copy import deepcopy
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.attention import (
    FlexAttention,
    ScaledDotProductAttention,
    VarlenAttention,
)
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.protocols.model_spec import ModelSpec

from .model import Llama3Model, Llama3TransformerBlock
from .parallelize import parallelize_llama
from .state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
    "Llama3Model",
    "llama3_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


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


def _build_llama3_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    n_kv_heads: int | None = None,
    inner_attention=None,
    mask_type: str = "causal",
) -> list[TransformerBlock.Config]:
    """Build a list of per-layer TransformerBlock configs with depth-scaled inits."""
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Llama3TransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=(
                        inner_attention
                        if inner_attention is not None
                        else ScaledDotProductAttention.Config()
                    ),
                    mask_type=mask_type,
                    rope_backend="complex",
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


def _debugmodel() -> Llama3Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Llama3Model.Config(
        dim=dim,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim, out_features=2048, param_init=_output_linear_init(dim)
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
        ),
    )


def _debugmodel_flex_attn() -> Llama3Model.Config:
    config = _debugmodel()
    flex_cfg = FlexAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = flex_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _debugmodel_varlen_attn() -> Llama3Model.Config:
    config = _debugmodel()
    varlen_cfg = VarlenAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = varlen_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _1b() -> Llama3Model.Config:
    dim = 2048
    n_heads = 32
    n_kv_heads = 8
    n_layers = 16
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=1024, ffn_dim_multiplier=1.5
            ),
        ),
    )


def _3b() -> Llama3Model.Config:
    dim = 3072
    n_heads = 24
    n_kv_heads = 8
    n_layers = 28
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=1024, ffn_dim_multiplier=1.0
            ),
        ),
    )


def _8b() -> Llama3Model.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=1024, ffn_dim_multiplier=1.3
            ),
        ),
    )


def _8b_flex() -> Llama3Model.Config:
    config = _8b()
    flex_cfg = FlexAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = flex_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _8b_varlen() -> Llama3Model.Config:
    config = _8b()
    varlen_cfg = VarlenAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = varlen_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _70b() -> Llama3Model.Config:
    dim = 8192
    n_heads = 64
    n_kv_heads = 8
    n_layers = 80
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=4096, ffn_dim_multiplier=1.3
            ),
        ),
    )


def _405b() -> Llama3Model.Config:
    dim = 16384
    n_heads = 128
    n_kv_heads = 8
    n_layers = 126
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=4096, ffn_dim_multiplier=1.2
            ),
        ),
    )


llama3_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_flex_attn": _debugmodel_flex_attn,
    "debugmodel_varlen_attn": _debugmodel_varlen_attn,
    "1B": _1b,
    "3B": _3b,
    "8B": _8b,
    "8B_flex": _8b_flex,
    "8B_varlen": _8b_varlen,
    "70B": _70b,
    "405B": _405b,
}


def model_registry(flavor: str) -> ModelSpec:
    # TODO(fegin): revisit
    # https://github.com/pytorch/torchtitan/pull/2785#issuecomment-4184528111
    # and resolve how we expand flex/varlen/sdpa for the config.
    config = llama3_configs[flavor]()
    return ModelSpec(
        name="llama3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
