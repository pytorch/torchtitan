# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.config import Function
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    FeedForward,
    GQAttention,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.param_init import (
    depth_scaled_std,
    resolve_deferred,
    skip_param_init,
)
from torchtitan.protocols.model_spec import ModelSpec

from .model import Llama3Model, Llama3TransformerBlock
from .parallelize import parallelize_llama
from .state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
    "Llama3Model",
    "llama3_configs",
]


def expand_layer_configs(config) -> None:
    """Expand the layer template into per-layer configs for a single model config.

    Deep-copies the ``layer`` template N times, then resolves ``DepthScaled``
    markers. Mutates config in place.
    """
    layers = []
    for layer_id in range(config.n_layers):
        cfg = deepcopy(config.layer)
        resolve_deferred(cfg, layer_id)
        layers.append(cfg)
    config.layers = layers


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_LINEAR_DEPTH_INIT = Function.Config(
    fn=lambda layer_id: {  # pyrefly: ignore [bad-argument-type]
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }
)
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


def _output_linear_init(dim: int):
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _debugmodel() -> Llama3Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            # TODO: find better ways to enforce dim = decoder dim // n_heads, for all models
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


def _debugmodel_flex_attn() -> Llama3Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


def _debugmodel_varlen_attn() -> Llama3Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                inner_attention=VarlenAttention.Config(),
                mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


def _1b() -> Llama3Model.Config:
    dim = 2048
    n_heads = 32
    n_kv_heads = 8
    n_layers = 16
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    dim, multiple_of=1024, ffn_dim_multiplier=1.5
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


def _3b() -> Llama3Model.Config:
    dim = 3072
    n_heads = 24
    n_kv_heads = 8
    n_layers = 28
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    dim, multiple_of=1024, ffn_dim_multiplier=1.0
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


def _8b() -> Llama3Model.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    dim, multiple_of=1024, ffn_dim_multiplier=1.3
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


def _8b_flex() -> Llama3Model.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    dim, multiple_of=1024, ffn_dim_multiplier=1.3
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


def _8b_varlen() -> Llama3Model.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    dim, multiple_of=1024, ffn_dim_multiplier=1.3
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                inner_attention=VarlenAttention.Config(),
                mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


def _70b() -> Llama3Model.Config:
    dim = 8192
    n_heads = 64
    n_kv_heads = 8
    n_layers = 80
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    dim, multiple_of=4096, ffn_dim_multiplier=1.3
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


def _405b() -> Llama3Model.Config:
    dim = 16384
    n_heads = 128
    n_kv_heads = 8
    n_layers = 126
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    dim, multiple_of=4096, ffn_dim_multiplier=1.2
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
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
    config = llama3_configs[flavor]()
    expand_layer_configs(config)
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
