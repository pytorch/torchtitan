# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import FeedForward, MoE, RoPE
from torchtitan.protocols.model_spec import ModelSpec

from .model import Attention, GatedDeltaNet, Model, TransformerBlock
from .parallelize import parallelize_qwen3_5_moe
from .state_dict_adapter import Qwen35MoEStateDictAdapter

__all__ = [
    "parallelize_qwen3_5_moe",
    "Model",
    "qwen35_moe_configs",
]

qwen35_moe_configs = {
    "debugmodel": Model.Config(
        dim=256,
        n_layers=8,
        vocab_size=2048,
        norm_eps=1e-6,
        rope=RoPE.Config(
            dim=16,
            max_seq_len=1024,
            theta=10000.0,
            backend="cos_sin",
        ),
        layer=TransformerBlock.Config(
            attention=Attention.Config(
                n_heads=4,
                n_kv_heads=2,
                head_dim=64,
                rotary_dim=16,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNet.Config(
                n_key_heads=2,
                n_value_heads=4,
                key_head_dim=64,
                value_head_dim=64,
            ),
            moe=MoE.Config(
                num_experts=8,
                num_shared_experts=0,
                top_k=2,
                hidden_dim=256,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
                use_grouped_mm=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=256),
        ),
        full_attention_interval=4,
    ),
    "35b-a3b": Model.Config(
        dim=2048,
        n_layers=40,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RoPE.Config(
            dim=64,
            max_seq_len=262144,
            theta=10_000_000.0,
            backend="cos_sin",
        ),
        layer=TransformerBlock.Config(
            attention=Attention.Config(
                n_heads=16,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNet.Config(
                n_key_heads=16,
                n_value_heads=32,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoE.Config(
                num_experts=256,
                num_shared_experts=0,
                top_k=8,
                hidden_dim=512,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=512),
        ),
        full_attention_interval=4,
    ),
    "35b-a3b-varlen": Model.Config(
        dim=2048,
        n_layers=40,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RoPE.Config(
            dim=64,
            max_seq_len=262144,
            theta=10_000_000.0,
            backend="cos_sin",
        ),
        layer=TransformerBlock.Config(
            attention=Attention.Config(
                n_heads=16,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="varlen",
                attn_mask_type="block_causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNet.Config(
                n_key_heads=16,
                n_value_heads=32,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoE.Config(
                num_experts=256,
                num_shared_experts=0,
                top_k=8,
                hidden_dim=512,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=512),
        ),
        full_attention_interval=4,
    ),
    "122b-a10b": Model.Config(
        dim=3072,
        n_layers=48,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RoPE.Config(
            dim=64,
            max_seq_len=262144,
            theta=10_000_000.0,
            backend="cos_sin",
        ),
        layer=TransformerBlock.Config(
            attention=Attention.Config(
                n_heads=32,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNet.Config(
                n_key_heads=16,
                n_value_heads=64,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoE.Config(
                num_experts=256,
                num_shared_experts=0,
                top_k=8,
                hidden_dim=1024,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=1024),
        ),
        full_attention_interval=4,
    ),
    "397b-a17b": Model.Config(
        dim=4096,
        n_layers=60,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RoPE.Config(
            dim=64,
            max_seq_len=262144,
            theta=10_000_000.0,
            backend="cos_sin",
        ),
        layer=TransformerBlock.Config(
            attention=Attention.Config(
                n_heads=32,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNet.Config(
                n_key_heads=16,
                n_value_heads=64,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoE.Config(
                num_experts=512,
                num_shared_experts=0,
                top_k=10,
                hidden_dim=1024,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=1024),
        ),
        full_attention_interval=4,
    ),
    "397B_A19B": Model.Config(
        dim=4096,
        n_layers=60,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RoPE.Config(
            dim=64,
            max_seq_len=1_000_000,
            theta=10_000_000.0,
            backend="cos_sin",
            scaling="yarn",
            rope_factor=3.0,
            original_seq_len=262144,
        ),
        layer=TransformerBlock.Config(
            attention=Attention.Config(
                n_heads=32,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNet.Config(
                n_key_heads=16,
                n_value_heads=64,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoE.Config(
                num_experts=512,
                num_shared_experts=0,
                top_k=10,
                hidden_dim=1024,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=1024),
        ),
        full_attention_interval=4,
    ),
}


def build(flavor: str) -> Model:
    """Build a Qwen3.5 MoE model by flavor name."""
    if flavor not in qwen35_moe_configs:
        raise ValueError(
            f"Unknown flavor '{flavor}'. Available: {list(qwen35_moe_configs.keys())}"
        )
    return qwen35_moe_configs[flavor].build()


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="qwen3_5_moe",
        flavor=flavor,
        model=qwen35_moe_configs[flavor],
        parallelize_fn=parallelize_qwen3_5_moe,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Qwen35MoEStateDictAdapter,
    )
