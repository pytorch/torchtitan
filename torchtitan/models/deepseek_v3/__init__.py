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
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.config import Function
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, FeedForward, Linear, RMSNorm, RoPE
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.config_expand import (
    fill_decoder_fields,
    fill_ffn_fields,
    fill_moe_fields,
)
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.param_init import depth_scaled_std, resolve_deferred
from torchtitan.protocols.model_spec import ModelSpec

from .model import Attention, DeepSeekV3Model, DeepSeekV3TransformerBlock

from .parallelize import parallelize_deepseekv3
from .state_dict_adapter import DeepSeekV3StateDictAdapter

__all__ = [
    "parallelize_deepseekv3",
    "DeepSeekV3Model",
    "deepseekv3_configs",
]


def _fill_dsv3_attn_fields(attn: Attention.Config, dim: int) -> None:
    """Fill expanded fields on a DeepSeek V3 MLA Attention.Config."""
    attn.dim = dim
    n_heads = attn.n_heads
    qk_head_dim = attn.qk_nope_head_dim + attn.qk_rope_head_dim

    if attn.q_lora_rank == 0:
        if attn.wq is not None:
            attn.wq.in_features = dim
            attn.wq.out_features = n_heads * qk_head_dim
    else:
        if attn.wq_a is not None:
            attn.wq_a.in_features = dim
            attn.wq_a.out_features = attn.q_lora_rank
        attn.q_norm.normalized_shape = attn.q_lora_rank
        if attn.wq_b is not None:
            attn.wq_b.in_features = attn.q_lora_rank
            attn.wq_b.out_features = n_heads * qk_head_dim

    attn.wkv_a.in_features = dim
    attn.wkv_a.out_features = attn.kv_lora_rank + attn.qk_rope_head_dim
    attn.kv_norm.normalized_shape = attn.kv_lora_rank
    attn.wkv_b.in_features = attn.kv_lora_rank
    attn.wkv_b.out_features = n_heads * (attn.qk_nope_head_dim + attn.v_head_dim)
    attn.wo.in_features = n_heads * attn.v_head_dim
    attn.wo.out_features = dim


def expand_layer_configs(config) -> None:
    """Expand the layer template into per-layer configs for a single model config.

    Handles dense vs. MoE layer selection based on n_dense_layers.
    Mutates config in place.
    """
    dim = config.dim
    fill_decoder_fields(config)
    assert isinstance(config.layer, DeepSeekV3TransformerBlock.Config)
    config.layer.attention_norm.normalized_shape = dim
    config.layer.ffn_norm.normalized_shape = dim
    n_dense = config.layer.n_dense_layers
    layers = []
    for layer_id in range(config.n_layers):
        cfg = deepcopy(config.layer)
        if layer_id >= n_dense:
            cfg = replace(cfg, feed_forward=None)
        else:
            cfg = replace(cfg, moe=None)
        resolve_deferred(cfg, layer_id)
        _fill_dsv3_attn_fields(
            cfg.attention, dim  # pyrefly: ignore [bad-argument-type]
        )
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
_LINEAR_DEPTH_INIT = Function.Config(
    fn=lambda layer_id: {  # pyrefly: ignore [bad-argument-type]
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }
)
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EXPERTS_DEPTH_INIT = Function.Config(
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


def _debugmodel() -> DeepSeekV3Model.Config:
    dim = 256
    n_layers = 6
    vocab_size = 2048
    n_heads = 16
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    rope_dim = 64
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                hidden_dim=moe_hidden_dim,
                num_experts=8,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=3,
                    score_func="softmax",
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
                shared_experts=FeedForward.Config(
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(param_init=_NORM_INIT),
                kv_norm=RMSNorm.Config(param_init=_NORM_INIT),
                n_heads=n_heads,
                q_lora_rank=0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=rope_dim,
                v_head_dim=128,
                mscale=0.70,
                wq=Linear.Config(param_init=_LINEAR_INIT),
                wkv_a=Linear.Config(param_init=_LINEAR_INIT),
                wkv_b=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=dense_hidden_dim,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )


def _debugmodel_flex_attn() -> DeepSeekV3Model.Config:
    config = _debugmodel()
    config.layer.attention.inner_attention = FlexAttention.Config()
    config.layer.attention.mask_type = "block_causal"
    return config


def _16b() -> DeepSeekV3Model.Config:
    dim = 2048
    n_layers = 27
    vocab_size = 102400
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                hidden_dim=moe_hidden_dim,
                num_experts=64,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=6,
                    score_func="softmax",
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
                shared_experts=FeedForward.Config(
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(param_init=_NORM_INIT),
                kv_norm=RMSNorm.Config(param_init=_NORM_INIT),
                n_heads=n_heads,
                q_lora_rank=0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=rope_dim,
                v_head_dim=128,
                mscale=0.70,
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                wq=Linear.Config(param_init=_LINEAR_INIT),
                wkv_a=Linear.Config(param_init=_LINEAR_INIT),
                wkv_b=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=dense_hidden_dim,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )


def _236b() -> DeepSeekV3Model.Config:
    dim = 5120
    n_layers = 60
    vocab_size = 102400
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 1536
    num_shared_experts = 2
    dense_hidden_dim = 12288
    rope_dim = 64
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                hidden_dim=moe_hidden_dim,
                num_experts=160,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=6,
                    num_expert_groups=8,
                    num_limited_groups=3,
                    score_func="softmax",
                    route_scale=16.0,
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
                shared_experts=FeedForward.Config(
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(param_init=_NORM_INIT),
                kv_norm=RMSNorm.Config(param_init=_NORM_INIT),
                n_heads=n_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=rope_dim,
                v_head_dim=128,
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                wq_a=Linear.Config(param_init=_LINEAR_INIT),
                wq_b=Linear.Config(param_init=_LINEAR_INIT),
                wkv_a=Linear.Config(param_init=_LINEAR_INIT),
                wkv_b=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=dense_hidden_dim,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )


def _671b() -> DeepSeekV3Model.Config:
    dim = 7168
    n_layers = 61
    vocab_size = 129280
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 2048
    num_shared_experts = 1
    dense_hidden_dim = 18432
    rope_dim = 64
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=3,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                hidden_dim=moe_hidden_dim,
                num_experts=256,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=8,
                    num_expert_groups=8,
                    num_limited_groups=4,
                    score_func="sigmoid",
                    route_norm=True,
                    route_scale=2.5,
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
                shared_experts=FeedForward.Config(
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(param_init=_NORM_INIT),
                kv_norm=RMSNorm.Config(param_init=_NORM_INIT),
                n_heads=n_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=rope_dim,
                v_head_dim=128,
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                wq_a=Linear.Config(param_init=_LINEAR_INIT),
                wq_b=Linear.Config(param_init=_LINEAR_INIT),
                wkv_a=Linear.Config(param_init=_LINEAR_INIT),
                wkv_b=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=dense_hidden_dim,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )


deepseekv3_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_flex_attn": _debugmodel_flex_attn,
    "16B": _16b,
    "236B": _236b,
    "671B": _671b,
}


def model_registry(flavor: str) -> ModelSpec:
    config = deepseekv3_configs[flavor]()
    expand_layer_configs(config)
    return ModelSpec(
        name="deepseek_v3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
