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
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import ParamInitializer, set_param_init

from .model import Attention, GptOssModel, GptOssTransformerBlock

from .moe import GptOssMoE
from .parallelize import parallelize_gptoss
from .state_dict_adapter import GptOssStateDictAdapter

__all__ = [
    "parallelize_gptoss",
    "GptOssModel",
    "gptoss_configs",
]


def setup_gptoss_param_init(model: GptOssModel) -> None:
    base_std: float = 0.02
    dim: int = model.config.dim
    final_out_std = dim**-0.5
    # GPT-OSS uses std=0.02 for tok_embeddings (not std=1.0 like other decoders)
    set_param_init(model.tok_embeddings, {"weight": partial(nn.init.normal_, std=base_std)})
    set_param_init(model.norm, {"weight": nn.init.ones_})
    set_param_init(
        model.output,
        {
            "weight": partial(
                nn.init.trunc_normal_,
                std=final_out_std,
                a=-3 * final_out_std,
                b=3 * final_out_std,
            )
        },
    )
    for i, layer in enumerate(model.layers.values()):
        std = base_std / (2 * (i + 1)) ** 0.5
        depth: ParamInitializer = partial(nn.init.trunc_normal_, std=std)
        # GPT-OSS: all attention linears + sinks are depth-scaled
        for proj in (
            layer.attention.wq,  # pyrefly: ignore [missing-attribute]
            layer.attention.wk,  # pyrefly: ignore [missing-attribute]
            layer.attention.wv,  # pyrefly: ignore [missing-attribute]
            layer.attention.wo,  # pyrefly: ignore [missing-attribute]
        ):
            set_param_init(
                proj,  # pyrefly: ignore [bad-argument-type]
                {"weight": depth, "bias": nn.init.zeros_},
            )
        # sinks is a bare nn.Parameter on the Attention module
        set_param_init(
            layer.attention,  # pyrefly: ignore [bad-argument-type]
            {"sinks": depth},
        )
        # GptOss MoE experts
        set_param_init(
            layer.moe.experts,  # pyrefly: ignore [missing-attribute]
            {
                "mlp1_weight": depth,
                "mlp1_bias": depth,
                "mlp2_weight": depth,
                "mlp2_bias": depth,
            },
        )
        set_param_init(
            layer.moe.router.gate,  # pyrefly: ignore [missing-attribute]
            {"weight": depth, "bias": nn.init.zeros_},
        )
        set_param_init(
            layer.attention_norm,  # pyrefly: ignore [bad-argument-type]
            {"weight": nn.init.ones_},
        )
        set_param_init(
            layer.ffn_norm,  # pyrefly: ignore [bad-argument-type]
            {"weight": nn.init.ones_},
        )


gptoss_configs = {
    "debugmodel": GptOssModel.Config(
        vocab_size=2048,
        dim=256,
        n_layers=4,
        param_init_fn=setup_gptoss_param_init,
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
        param_init_fn=setup_gptoss_param_init,
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
        param_init_fn=setup_gptoss_param_init,
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
