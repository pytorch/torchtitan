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
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.param_init import depth_scaled_std, PerLayer
from torchtitan.protocols.model_spec import ModelSpec

from .model import Attention, DeepSeekV3Model, DeepSeekV3TransformerBlock

from .parallelize import parallelize_deepseekv3
from .state_dict_adapter import DeepSeekV3StateDictAdapter

__all__ = [
    "parallelize_deepseekv3",
    "DeepSeekV3Model",
    "deepseekv3_configs",
]

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


def _output_linear_init(dim: int):
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


deepseekv3_configs = {
    "debugmodel": DeepSeekV3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=6,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(256)),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                hidden_dim=256,
                num_experts=8,
                num_shared_experts=2,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=3,
                    score_func="softmax",
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(
                    param_init=PerLayer(
                        lambda layer_id: {
                            "w1": partial(nn.init.trunc_normal_, std=0.02),
                            "w2": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                            "w3": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                        }
                    )
                ),
                shared_experts=FeedForward.Config(
                    hidden_dim=512,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(param_init=_NORM_INIT),
                kv_norm=RMSNorm.Config(param_init=_NORM_INIT),
                n_heads=16,
                q_lora_rank=0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                mscale=0.70,
                wq=Linear.Config(param_init=_LINEAR_INIT),
                wkv_a=Linear.Config(param_init=_LINEAR_INIT),
                wkv_b=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=1024,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
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
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(256)),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                hidden_dim=256,
                num_experts=8,
                num_shared_experts=2,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=3,
                    score_func="softmax",
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(
                    param_init=PerLayer(
                        lambda layer_id: {
                            "w1": partial(nn.init.trunc_normal_, std=0.02),
                            "w2": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                            "w3": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                        }
                    )
                ),
                shared_experts=FeedForward.Config(
                    hidden_dim=512,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(param_init=_NORM_INIT),
                kv_norm=RMSNorm.Config(param_init=_NORM_INIT),
                n_heads=16,
                q_lora_rank=0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                mscale=0.70,
                attn_backend="flex",
                attn_mask_type="block_causal",
                wq=Linear.Config(param_init=_LINEAR_INIT),
                wkv_a=Linear.Config(param_init=_LINEAR_INIT),
                wkv_b=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=1024,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
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
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(2048)),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                hidden_dim=1408,
                num_experts=64,
                num_shared_experts=2,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=6,
                    score_func="softmax",
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(
                    param_init=PerLayer(
                        lambda layer_id: {
                            "w1": partial(nn.init.trunc_normal_, std=0.02),
                            "w2": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                            "w3": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                        }
                    )
                ),
                shared_experts=FeedForward.Config(
                    hidden_dim=2816,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(param_init=_NORM_INIT),
                kv_norm=RMSNorm.Config(param_init=_NORM_INIT),
                n_heads=16,
                q_lora_rank=0,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                mscale=0.70,
                attn_backend="flex",
                attn_mask_type="block_causal",
                wq=Linear.Config(param_init=_LINEAR_INIT),
                wkv_a=Linear.Config(param_init=_LINEAR_INIT),
                wkv_b=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=10944,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
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
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(5120)),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=1,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
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
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(
                    param_init=PerLayer(
                        lambda layer_id: {
                            "w1": partial(nn.init.trunc_normal_, std=0.02),
                            "w2": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                            "w3": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                        }
                    )
                ),
                shared_experts=FeedForward.Config(
                    hidden_dim=3072,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(param_init=_NORM_INIT),
                kv_norm=RMSNorm.Config(param_init=_NORM_INIT),
                n_heads=128,
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                attn_backend="flex",
                attn_mask_type="block_causal",
                wq_a=Linear.Config(param_init=_LINEAR_INIT),
                wq_b=Linear.Config(param_init=_LINEAR_INIT),
                wkv_a=Linear.Config(param_init=_LINEAR_INIT),
                wkv_b=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=12288,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
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
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(7168)),
        layer=DeepSeekV3TransformerBlock.Config(
            n_dense_layers=3,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
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
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(
                    param_init=PerLayer(
                        lambda layer_id: {
                            "w1": partial(nn.init.trunc_normal_, std=0.02),
                            "w2": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                            "w3": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                        }
                    )
                ),
                shared_experts=FeedForward.Config(
                    hidden_dim=2048,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            attention=Attention.Config(
                q_norm=RMSNorm.Config(param_init=_NORM_INIT),
                kv_norm=RMSNorm.Config(param_init=_NORM_INIT),
                n_heads=128,
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                attn_backend="flex",
                attn_mask_type="block_causal",
                wq_a=Linear.Config(param_init=_LINEAR_INIT),
                wq_b=Linear.Config(param_init=_LINEAR_INIT),
                wkv_a=Linear.Config(param_init=_LINEAR_INIT),
                wkv_b=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=18432,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
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
