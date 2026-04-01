# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from copy import deepcopy
from dataclasses import replace
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.config import DeferredCallable
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, FeedForward, GQAttention, Linear, RoPE
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.config_expand import (
    fill_decoder_fields,
    fill_ffn_fields,
    fill_gqa_fields,
    fill_moe_fields,
)
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
    dim = config.dim
    fill_decoder_fields(config)
    assert isinstance(config.layer, Qwen3TransformerBlock.Config)
    config.layer.attention_norm.normalized_shape = dim
    config.layer.ffn_norm.normalized_shape = dim
    layers = []
    for layer_id in range(config.n_layers):
        cfg = deepcopy(config.layer)
        if cfg.moe_enabled:
            cfg = replace(cfg, feed_forward=None)
        else:
            cfg = replace(cfg, moe=None)
        resolve_deferred(cfg, layer_id)
        fill_gqa_fields(cfg.attention, dim)  # pyrefly: ignore [bad-argument-type]
        if cfg.feed_forward is not None:
            fill_ffn_fields(cfg.feed_forward, dim)
        if cfg.moe is not None:
            fill_moe_fields(cfg.moe, dim)
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


def _debugmodel() -> Qwen3Model.Config:
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


def _debugmodel_flex() -> Qwen3Model.Config:
    config = _debugmodel()
    config.layer.attention.inner_attention = FlexAttention.Config()
    config.layer.attention.mask_type = "block_causal"
    return config


def _debugmodel_flex_flash() -> Qwen3Model.Config:
    from torchtitan.tools.utils import has_cuda_capability

    if has_cuda_capability(10, 0):
        # NOTE: On NVIDIA Blackwell, to use FLASH backend we need
        # block size at least (256, 128) due to how the kernel works.
        block_size = (256, 128)
    elif has_cuda_capability(9, 0):
        block_size = (128, 128)
    else:
        raise ValueError(
            "Flash backend of FlexAttention is only supported on Hopper or Blackwell"
        )
    config = _debugmodel()
    config.layer.attention.inner_attention = FlexAttention.Config(
        block_size=block_size, kernel_options={"BACKEND": "FLASH"}
    )
    config.layer.attention.mask_type = "block_causal"
    return config


def _debugmodel_varlen() -> Qwen3Model.Config:
    config = _debugmodel()
    config.layer.attention.inner_attention = VarlenAttention.Config()
    config.layer.attention.mask_type = "block_causal"
    return config


def _0_6b() -> Qwen3Model.Config:
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


def _1_7b() -> Qwen3Model.Config:
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


def _4b() -> Qwen3Model.Config:
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


def _0_6b_varlen() -> Qwen3Model.Config:
    config = _0_6b()
    config.layer.attention.inner_attention = VarlenAttention.Config()
    config.layer.attention.mask_type = "block_causal"
    return config


def _1_7b_varlen() -> Qwen3Model.Config:
    config = _1_7b()
    config.layer.attention.inner_attention = VarlenAttention.Config()
    config.layer.attention.mask_type = "block_causal"
    return config


def _8b() -> Qwen3Model.Config:
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


def _8b_varlen() -> Qwen3Model.Config:
    config = _8b()
    config.layer.attention.inner_attention = VarlenAttention.Config()
    config.layer.attention.mask_type = "block_causal"
    return config


def _14b() -> Qwen3Model.Config:
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


def _32b() -> Qwen3Model.Config:
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


def _debugmodel_moe() -> Qwen3Model.Config:
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


def _30b_a3b() -> Qwen3Model.Config:
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


def _235b_a22b() -> Qwen3Model.Config:
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
    "debugmodel_flex_flash": _debugmodel_flex_flash,
    "debugmodel_varlen": _debugmodel_varlen,
    "0.6B": _0_6b,
    "0.6B_varlen": _0_6b_varlen,
    "1.7B": _1_7b,
    "1.7B_varlen": _1_7b_varlen,
    "4B": _4b,
    "8B": _8b,
    "8B_varlen": _8b_varlen,
    "14B": _14b,
    "32B": _32b,
    "debugmodel_moe": _debugmodel_moe,
    "30B-A3B": _30b_a3b,
    "235B-A22B": _235b_a22b,
}


def model_registry(flavor: str) -> ModelSpec:
    config = qwen3_configs[flavor]()
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
