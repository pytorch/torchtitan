# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE
from torchtitan.models.common.moe import TokenChoiceTopKRouter

from torchtitan.models.common.param_init import DepthScaledTruncNormal, RegexInitializer
from torchtitan.protocols.model_spec import ModelSpec

from .model import Attention, GptOssModel, GptOssTransformerBlock

from .moe import GptOssMoE
from .parallelize import parallelize_gptoss
from .state_dict_adapter import GptOssStateDictAdapter

__all__ = [
    "parallelize_gptoss",
    "GptOssModel",
    "gptoss_configs",
]


def _gptoss_param_init(dim: int, n_layers: int) -> RegexInitializer:
    depth_std = DepthScaledTruncNormal(n_layers=n_layers)
    final_out_std = dim**-0.5
    return RegexInitializer(
        {
            # Token embeddings
            r"tok_embeddings\.weight": partial(nn.init.normal_, std=0.02),
            # All attention linears get depth-scaled (GPT-OSS-specific)
            r"layers\..+\.attention\.w[qkvo]\.weight": depth_std,
            # Attention sinks parameter
            r"layers\..+\.attention\.sinks": depth_std,
            # MoE expert weights and biases (all depth-scaled)
            r"layers\..+\.moe\.experts\.mlp[12]_weight": depth_std,
            r"layers\..+\.moe\.experts\.mlp[12]_bias": depth_std,
            # MoE router gate
            r"layers\..+\.moe\.router\.gate\.weight": depth_std,
            # Norm weights
            r".*norm.*\.weight": nn.init.ones_,
            # Output projection
            r"output\.weight": partial(
                nn.init.trunc_normal_,
                std=final_out_std,
                a=-3 * final_out_std,
                b=3 * final_out_std,
            ),
            # Default weights
            r".*\.weight": partial(nn.init.trunc_normal_, std=0.02),
            # Biases
            r".*\.bias": nn.init.zeros_,
            # Catch-all
            r".*": partial(nn.init.trunc_normal_, std=0.02),
        }
    )


gptoss_configs = {
    "debugmodel": GptOssModel.Config(
        vocab_size=2048,
        dim=256,
        n_layers=4,
        param_init=_gptoss_param_init(256, 4),
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=GptOssTransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            moe=GptOssMoE.Config(
                hidden_dim=2880,
                num_experts=8,
                num_shared_experts=0,
                score_before_experts=False,
                load_balance_coeff=1e-3,
                router=TokenChoiceTopKRouter.Config(
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(bias=True),
                    top_k=4,
                ),
            ),
            attention=Attention.Config(
                linear_bias=True,
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=131072,
            theta=150000.0,
            backend="cos_sin",
            scaling="yarn",
            rope_factor=32,
            beta_slow=32.0,
            beta_fast=1.0,
            original_seq_len=4096,
        ),
    ),
    "20b": GptOssModel.Config(
        n_layers=24,
        param_init=_gptoss_param_init(2880, 24),
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=GptOssTransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            moe=GptOssMoE.Config(
                hidden_dim=2880,
                num_experts=32,
                num_shared_experts=0,
                score_before_experts=False,
                load_balance_coeff=1e-3,
                router=TokenChoiceTopKRouter.Config(
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(bias=True),
                    top_k=4,
                ),
            ),
            attention=Attention.Config(
                linear_bias=True,
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=131072,
            theta=150000.0,
            backend="cos_sin",
            scaling="yarn",
            rope_factor=32,
            beta_slow=32.0,
            beta_fast=1.0,
            original_seq_len=4096,
        ),
    ),
    "120b": GptOssModel.Config(
        n_layers=36,
        param_init=_gptoss_param_init(2880, 36),
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=GptOssTransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            moe=GptOssMoE.Config(
                hidden_dim=2880,
                num_experts=128,
                num_shared_experts=0,
                score_before_experts=False,
                load_balance_coeff=1e-3,
                router=TokenChoiceTopKRouter.Config(
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(bias=True),
                    top_k=4,
                ),
            ),
            attention=Attention.Config(
                linear_bias=True,
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=131072,
            theta=150000.0,
            backend="cos_sin",
            scaling="yarn",
            rope_factor=32,
            beta_slow=32.0,
            beta_fast=1.0,
            original_seq_len=4096,
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="gpt_oss",
        flavor=flavor,
        model=gptoss_configs[flavor],
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=GptOssStateDictAdapter,
    )
