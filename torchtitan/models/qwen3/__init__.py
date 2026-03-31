# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import copy
from copy import deepcopy
from dataclasses import replace
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.config import DeferredCallable
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, FeedForward, GQAttention, Linear, RoPE
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.param_init import (
    depth_scaled_std,
    resolve_deferred,
    skip_param_init,
)
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.model_spec import ModelSpec

from .model import Qwen3Model, Qwen3TransformerBlock
from .parallelize import parallelize_qwen3
from .state_dict_adapter import Qwen3StateDictAdapter

__all__ = [
    "parallelize_qwen3",
    "Qwen3Model",
    "qwen3_configs",
]


def expand_layer_configs(config) -> None:
    """Expand the layer template into per-layer configs for a single model config.

    Handles MoE vs. dense feed-forward selection based on moe_enabled flag.
    Mutates config in place.
    """
    assert isinstance(config.layer, Qwen3TransformerBlock.Config)
    layers = []
    for layer_id in range(config.n_layers):
        cfg = deepcopy(config.layer)
        if cfg.moe_enabled:
            cfg = replace(cfg, feed_forward=None)
        else:
            cfg = replace(cfg, moe=None)
        resolve_deferred(cfg, layer_id)
        layers.append(cfg)
    config.layers = layers


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_LINEAR_DEPTH_INIT = DeferredCallable.Config(
    fn=lambda layer_id: {  # pyrefly: ignore [bad-argument-type]
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }
)
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
_EXPERTS_DEPTH_INIT = DeferredCallable.Config(
    fn=lambda layer_id: {  # pyrefly: ignore [bad-argument-type]
        "w1": partial(nn.init.trunc_normal_, std=0.02),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }
)


def _output_linear_init(dim: int):
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


# Adding different variants of the model


def _debugmodel():
    dim = 256
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=2048,
        dim=dim,
        n_layers=8,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=3072,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


def _debugmodel_flex():
    dim = 256
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=2048,
        dim=dim,
        n_layers=8,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=3072,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="flex",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


def _0_6b():
    dim = 1024
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=28,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=3072,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


def _1_7b():
    dim = 2048
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=28,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=6144,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


def _4b():
    dim = 2560
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=36,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=9728,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


def _8b():
    dim = 4096
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=36,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=12288,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


def _14b():
    dim = 5120
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=40,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=17408,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=40,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


def _32b():
    dim = 5120
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=64,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=25600,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=64,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


# Qwen3-MoE models


def _debugmodel_moe():
    dim = 256
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=2048,
        dim=dim,
        n_layers=8,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=64,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=8,
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=3072,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


def _30b_a3b():
    dim = 2048
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=48,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=128,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=8,
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=6144,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=4,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=262144,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )


def _235b_a22b():
    dim = 4096
    head_dim = 128
    return Qwen3Model.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=94,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=1536,
                num_experts=128,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=8,
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=12288,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=64,
                n_kv_heads=4,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=5000000.0,
            backend="cos_sin",
        ),
    )


qwen3_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_flex": _debugmodel_flex,
    "0.6B": _0_6b,
    "1.7B": _1_7b,
    "4B": _4b,
    "8B": _8b,
    "14B": _14b,
    "32B": _32b,
    "debugmodel_moe": _debugmodel_moe,
    "30B-A3B": _30b_a3b,
    "235B-A22B": _235b_a22b,
}


def model_registry(flavor: str, attn_backend_override: str | None = None) -> ModelSpec:
    config = qwen3_configs[flavor]()
    if attn_backend_override is not None:
        assert attn_backend_override in [
            "sdpa",
            "flex",
            "varlen",
        ], f"Invalid attn_backend_override: {attn_backend_override}"
        config.layer.attention.attn_backend = attn_backend_override
        if attn_backend_override == "varlen":
            config.layer.attention.attn_mask_type = "block_causal"
    expand_layer_configs(config)
    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
