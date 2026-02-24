# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import RoPE
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


gptoss_configs = {
    "debugmodel": GptOssModel.Config(
        vocab_size=2048,
        dim=256,
        n_layers=4,
        layer=GptOssTransformerBlock.Config(
            moe=GptOssMoE.Config(
                hidden_dim=2880,
                num_experts=8,
                num_shared_experts=0,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                gate_bias=True,
                score_before_experts=False,
                top_k=4,
                use_grouped_mm=True,
                load_balance_coeff=1e-3,
            ),
            attention=Attention.Config(),
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
        layer=GptOssTransformerBlock.Config(
            moe=GptOssMoE.Config(
                hidden_dim=2880,
                num_experts=32,
                num_shared_experts=0,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                gate_bias=True,
                score_before_experts=False,
                top_k=4,
                use_grouped_mm=True,
                load_balance_coeff=1e-3,
            ),
            attention=Attention.Config(),
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
        layer=GptOssTransformerBlock.Config(
            moe=GptOssMoE.Config(
                hidden_dim=2880,
                num_experts=128,
                num_shared_experts=0,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                gate_bias=True,
                score_before_experts=False,
                top_k=4,
                use_grouped_mm=True,
                load_balance_coeff=1e-3,
            ),
            attention=Attention.Config(),
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
