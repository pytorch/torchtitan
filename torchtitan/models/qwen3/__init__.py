# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from copy import deepcopy
from dataclasses import replace
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, FeedForward, GQAttention, Linear, RoPE
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.param_init import (
    depth_scaled_std,
    PerLayer,
    resolve_per_layer,
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


def _expand_layer_configs(configs: dict) -> dict:
    """Expand the layer template into per-layer configs for each model config.

    Handles MoE vs. dense feed-forward selection based on moe_enabled flag.
    Mutates configs in place and returns the same dict.
    """
    for config in configs.values():
        assert isinstance(config.layer, Qwen3TransformerBlock.Config)
        layers = []
        for layer_id in range(config.n_layers):
            cfg = deepcopy(config.layer)
            if cfg.moe_enabled:
                cfg = replace(cfg, feed_forward=None)
            else:
                cfg = replace(cfg, moe=None)
            resolve_per_layer(cfg, layer_id)
            layers.append(cfg)
        config.layers = layers
    return configs


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_LINEAR_DEPTH_INIT = PerLayer(
    lambda layer_id: {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }
)
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
_EXPERTS_DEPTH_INIT = PerLayer(
    lambda layer_id: {
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

qwen3_configs = {
    "debugmodel": Qwen3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=8,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(256)),
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "debugmodel_flex": Qwen3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=8,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(256)),
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="flex",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "0.6B": Qwen3Model.Config(
        vocab_size=151936,
        dim=1024,
        n_layers=28,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(1024)),
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "1.7B": Qwen3Model.Config(
        vocab_size=151936,
        dim=2048,
        n_layers=28,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(2048)),
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "4B": Qwen3Model.Config(
        vocab_size=151936,
        dim=2560,
        n_layers=36,
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(2560)),
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "8B": Qwen3Model.Config(
        vocab_size=151936,
        dim=4096,
        n_layers=36,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(4096)),
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "14B": Qwen3Model.Config(
        vocab_size=151936,
        dim=5120,
        n_layers=40,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(5120)),
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "32B": Qwen3Model.Config(
        vocab_size=151936,
        dim=5120,
        n_layers=64,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(5120)),
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    # Qwen3-MoE models
    "debugmodel_moe": Qwen3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=8,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(256)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=64,
                num_shared_experts=0,
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "30B-A3B": Qwen3Model.Config(
        vocab_size=151936,
        dim=2048,
        n_layers=48,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(2048)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=128,
                num_shared_experts=0,
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=262144,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "235B-A22B": Qwen3Model.Config(
        vocab_size=151936,
        dim=4096,
        n_layers=94,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(4096)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=1536,
                num_experts=128,
                num_shared_experts=0,
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
                head_dim=128,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=5000000.0,
            backend="cos_sin",
        ),
    ),
}


_expand_layer_configs(qwen3_configs)


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=qwen3_configs[flavor],
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
