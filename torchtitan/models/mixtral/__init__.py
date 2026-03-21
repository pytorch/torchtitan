# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, GQAttention, Linear, RoPE
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.model_spec import ModelSpec

from .model import MixtralModel, MixtralTransformerBlock
from .parallelize import parallelize_mixtral
from .state_dict_adapter import MixtralStateDictAdapter

__all__ = [
    "MixtralModel",
    "mixtral_configs",
]

mixtral_configs = {
    "debugmodel": MixtralModel.Config(
        dim=256,
        n_layers=4,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        norm=RMSNorm.Config(),
        layer=MixtralTransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            attention=GQAttention.Config(
                n_heads=8,
                n_kv_heads=2,
                head_dim=32,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
            moe=MoE.Config(
                hidden_dim=512,
                num_experts=4,
                num_shared_experts=0,
                score_before_experts=False,
                experts=GroupedExperts.Config(use_grouped_mm=False),
                router=TokenChoiceTopKRouter.Config(
                    top_k=2,
                    score_func="softmax",
                    route_norm=True,
                ),
            ),
        ),
        rope=RoPE.Config(
            dim=32,
            max_seq_len=4096,
            theta=1_000_000.0,
            backend="cos_sin",
        ),
    ),
    "8x7b": MixtralModel.Config(
        dim=4096,
        n_layers=32,
        vocab_size=32000,
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        norm=RMSNorm.Config(eps=1e-5),
        layer=MixtralTransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-5),
            ffn_norm=RMSNorm.Config(eps=1e-5),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
            moe=MoE.Config(
                hidden_dim=14336,
                num_experts=8,
                num_shared_experts=0,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=2,
                    score_func="softmax",
                    route_norm=True,
                ),
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=32768,
            theta=1_000_000.0,
            backend="cos_sin",
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="mixtral",
        flavor=flavor,
        model=mixtral_configs[flavor],
        parallelize_fn=parallelize_mixtral,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=MixtralStateDictAdapter,
    )
