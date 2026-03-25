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
from torchtitan.models.common.param_init import depth_scaled_std, PerLayer
from torchtitan.protocols.model_spec import ModelSpec

from .model import Attention, GptOssModel, GptOssTransformerBlock

from .moe import GptOssGroupedExperts, GptOssMoE
from .parallelize import parallelize_gptoss
from .state_dict_adapter import GptOssStateDictAdapter

__all__ = [
    "parallelize_gptoss",
    "GptOssModel",
    "gptoss_configs",
]

_LINEAR_DEPTH_INIT = PerLayer(
    lambda layer_id: {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }
)
_NORM_INIT = {"weight": nn.init.ones_}
# GPT-OSS uses std=0.02 for embeddings (model-specific)
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}


def _output_linear_init(dim: int):
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


_SINKS_INIT = PerLayer(
    lambda layer_id: {
        "sinks": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id))
    }
)

_GPTOSS_EXPERT_INIT = PerLayer(
    lambda layer_id: {
        "mlp1_weight": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
        ),
        "mlp1_bias": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
        ),
        "mlp2_weight": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
        ),
        "mlp2_bias": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
        ),
    }
)

# pyrefly: ignore [bad-argument-type]
_GPTOSS_EXPERTS_CONFIG = GptOssGroupedExperts.Config(param_init=_GPTOSS_EXPERT_INIT)


gptoss_configs = {
    "debugmodel": GptOssModel.Config(
        vocab_size=2048,
        dim=256,
        n_layers=4,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(256)),
        layer=GptOssTransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=GptOssMoE.Config(
                hidden_dim=2880,
                num_experts=8,
                num_shared_experts=0,
                score_before_experts=False,
                load_balance_coeff=1e-3,
                experts=_GPTOSS_EXPERTS_CONFIG,  # pyrefly: ignore [bad-argument-type]
                router=TokenChoiceTopKRouter.Config(
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(bias=True, param_init=_LINEAR_DEPTH_INIT),
                    top_k=4,
                ),
            ),
            attention=Attention.Config(
                wqkv=Linear.Config(bias=True, param_init=_LINEAR_DEPTH_INIT),
                wo=Linear.Config(bias=True, param_init=_LINEAR_DEPTH_INIT),
                param_init=_SINKS_INIT,
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
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(2880)),
        layer=GptOssTransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=GptOssMoE.Config(
                hidden_dim=2880,
                num_experts=32,
                num_shared_experts=0,
                score_before_experts=False,
                load_balance_coeff=1e-3,
                experts=_GPTOSS_EXPERTS_CONFIG,  # pyrefly: ignore [bad-argument-type]
                router=TokenChoiceTopKRouter.Config(
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(bias=True, param_init=_LINEAR_DEPTH_INIT),
                    top_k=4,
                ),
            ),
            attention=Attention.Config(
                wqkv=Linear.Config(bias=True, param_init=_LINEAR_DEPTH_INIT),
                wo=Linear.Config(bias=True, param_init=_LINEAR_DEPTH_INIT),
                param_init=_SINKS_INIT,
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
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(2880)),
        layer=GptOssTransformerBlock.Config(
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=GptOssMoE.Config(
                hidden_dim=2880,
                num_experts=128,
                num_shared_experts=0,
                score_before_experts=False,
                load_balance_coeff=1e-3,
                experts=_GPTOSS_EXPERTS_CONFIG,  # pyrefly: ignore [bad-argument-type]
                router=TokenChoiceTopKRouter.Config(
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(bias=True, param_init=_LINEAR_DEPTH_INIT),
                    top_k=4,
                ),
            ),
            attention=Attention.Config(
                wqkv=Linear.Config(bias=True, param_init=_LINEAR_DEPTH_INIT),
                wo=Linear.Config(bias=True, param_init=_LINEAR_DEPTH_INIT),
                param_init=_SINKS_INIT,
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
