# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import copy

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    Embedding,
    FeedForward,
    FlexAttention,
    GQAttention,
    Linear,
    RoPE,
    VarlenAttention,
)
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
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

# Adding different variants of the model

qwen3_configs = {
    "debugmodel": Qwen3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=8,
        norm=RMSNorm.Config(eps=1e-6),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(
                hidden_dim=3072,
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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
        norm=RMSNorm.Config(eps=1e-6),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(
                hidden_dim=3072,
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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
        norm=RMSNorm.Config(eps=1e-6),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(
                hidden_dim=6144,
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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
        norm=RMSNorm.Config(eps=1e-6),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(
                hidden_dim=9728,
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        norm=RMSNorm.Config(eps=1e-6),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(
                hidden_dim=12288,
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        norm=RMSNorm.Config(eps=1e-6),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(
                hidden_dim=17408,
            ),
            attention=GQAttention.Config(
                n_heads=40,
                n_kv_heads=8,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        norm=RMSNorm.Config(eps=1e-6),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(
                hidden_dim=25600,
            ),
            attention=GQAttention.Config(
                n_heads=64,
                n_kv_heads=8,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        norm=RMSNorm.Config(eps=1e-6),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
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
                ),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=3072,
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        norm=RMSNorm.Config(eps=1e-6),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
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
                ),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=6144,
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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
        tok_embeddings=Embedding.Config(),
        output=Linear.Config(),
        norm=RMSNorm.Config(eps=1e-6),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
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
                ),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=12288,
            ),
            attention=GQAttention.Config(
                n_heads=64,
                n_kv_heads=4,
                head_dim=128,
                q_norm=RMSNorm.Config(eps=1e-6),
                k_norm=RMSNorm.Config(eps=1e-6),
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


def model_registry(flavor: str, attn_backend_override: str | None = None) -> ModelSpec:
    model = copy.deepcopy(qwen3_configs[flavor])
    if attn_backend_override is not None:
        from torchtitan.models.common import ScaledDotProductAttention

        match attn_backend_override:
            case "sdpa":
                model.layer.attention.inner_attention = (
                    ScaledDotProductAttention.Config()
                )
            case "flex":
                model.layer.attention.inner_attention = FlexAttention.Config()
                model.layer.attention.mask_type = "block_causal"
            case "flex_flash":
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
                model.layer.attention.inner_attention = FlexAttention.Config(
                    block_size=block_size, kernel_options={"BACKEND": "FLASH"}
                )
                model.layer.attention.mask_type = "block_causal"
            case "varlen":
                model.layer.attention.inner_attention = VarlenAttention.Config()
                model.layer.attention.mask_type = "block_causal"
            case _:
                raise ValueError(
                    f"Invalid attn_backend_override: {attn_backend_override}"
                )

    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=model,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
