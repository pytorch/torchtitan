# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, FeedForward, Linear, RMSNorm, RoPE
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
from torchtitan.models.common.param_init import (
    init_decoder_common,
    init_feed_forward,
    init_moe,
)
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import ParamInitializer, set_param_init

from .model import Attention, DeepSeekV3Model, DeepSeekV3TransformerBlock

from .parallelize import parallelize_deepseekv3
from .state_dict_adapter import DeepSeekV3StateDictAdapter

__all__ = [
    "parallelize_deepseekv3",
    "DeepSeekV3Model",
    "deepseekv3_configs",
]


def _init_dsv3_attention(
    attn: Attention, *, default: ParamInitializer, depth: ParamInitializer
) -> None:
    set_param_init(attn.wkv_a, {"weight": default})
    set_param_init(attn.wkv_b, {"weight": default})
    if attn.q_lora_rank > 0:
        set_param_init(attn.wq_a, {"weight": default})
        set_param_init(attn.wq_b, {"weight": default})
        set_param_init(attn.q_norm, {"weight": nn.init.ones_})
    else:
        set_param_init(attn.wq, {"weight": default})
    set_param_init(attn.wo, {"weight": depth})
    set_param_init(attn.kv_norm, {"weight": nn.init.ones_})


def setup_deepseekv3_param_init(model: DeepSeekV3Model) -> None:
    base_std: float = 0.02
    default: ParamInitializer = partial(nn.init.trunc_normal_, std=base_std)
    init_decoder_common(model, base_std=base_std)
    for i, layer in enumerate(model.layers.values()):
        std = base_std / (2 * (i + 1)) ** 0.5
        depth: ParamInitializer = partial(nn.init.trunc_normal_, std=std)
        _init_dsv3_attention(
            layer.attention,  # pyrefly: ignore [bad-argument-type]
            default=default,
            depth=depth,
        )
        if layer.moe_enabled:
            init_moe(
                layer.moe,  # pyrefly: ignore [bad-argument-type]
                default=default,
                depth=depth,
            )
        else:
            init_feed_forward(
                layer.feed_forward,  # pyrefly: ignore [bad-argument-type]
                default=default,
                depth=depth,
            )
        set_param_init(
            layer.attention_norm,  # pyrefly: ignore [bad-argument-type]
            {"weight": nn.init.ones_},
        )
        set_param_init(
            layer.ffn_norm,  # pyrefly: ignore [bad-argument-type]
            {"weight": nn.init.ones_},
        )


deepseekv3_configs = {
    "debugmodel": DeepSeekV3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=6,
        param_init_fn=setup_deepseekv3_param_init,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            moe=MoE.Config(
                hidden_dim=256,
                num_experts=8,
                num_shared_experts=2,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=3,
                    score_func="softmax",
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(),
                kv_norm=RMSNorm.Config(),
                n_heads=16,
                q_lora_rank=0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                mscale=0.70,
                wq=Linear.Config(),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=1024,
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
    "debugmodel_flex_attn": DeepSeekV3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=6,
        param_init_fn=setup_deepseekv3_param_init,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            moe=MoE.Config(
                hidden_dim=256,
                num_experts=8,
                num_shared_experts=2,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=3,
                    score_func="softmax",
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(),
                kv_norm=RMSNorm.Config(),
                n_heads=16,
                q_lora_rank=0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                mscale=0.70,
                attn_backend="flex",
                attn_mask_type="block_causal",
                wq=Linear.Config(),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=1024,
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
    "16B": DeepSeekV3Model.Config(
        vocab_size=102400,
        dim=2048,
        n_layers=27,
        param_init_fn=setup_deepseekv3_param_init,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            moe=MoE.Config(
                hidden_dim=1408,
                num_experts=64,
                num_shared_experts=2,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=6,
                    score_func="softmax",
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(),
                kv_norm=RMSNorm.Config(),
                n_heads=16,
                q_lora_rank=0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                mscale=0.70,
                attn_backend="flex",
                attn_mask_type="block_causal",
                wq=Linear.Config(),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=10944,
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
    "236B": DeepSeekV3Model.Config(
        vocab_size=102400,
        dim=5120,
        n_layers=60,
        param_init_fn=setup_deepseekv3_param_init,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            moe=MoE.Config(
                hidden_dim=1536,
                num_experts=160,
                num_shared_experts=2,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=6,
                    num_expert_groups=8,
                    num_limited_groups=3,
                    score_func="softmax",
                    route_scale=16.0,
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(),
                kv_norm=RMSNorm.Config(),
                n_heads=128,
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                attn_backend="flex",
                attn_mask_type="block_causal",
                wq_a=Linear.Config(),
                wq_b=Linear.Config(),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=12288,
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
    "671B": DeepSeekV3Model.Config(
        vocab_size=129280,
        dim=7168,
        n_layers=61,
        param_init_fn=setup_deepseekv3_param_init,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=3,
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            moe=MoE.Config(
                hidden_dim=2048,
                num_experts=256,
                num_shared_experts=1,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=8,
                    num_expert_groups=8,
                    num_limited_groups=4,
                    score_func="sigmoid",
                    route_norm=True,
                    route_scale=2.5,
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(),
                kv_norm=RMSNorm.Config(),
                n_heads=128,
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                attn_backend="flex",
                attn_mask_type="block_causal",
                wq_a=Linear.Config(),
                wq_b=Linear.Config(),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=18432,
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="deepseek_v3",
        flavor=flavor,
        model=deepseekv3_configs[flavor],
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
