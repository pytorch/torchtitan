# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.models.common import FeedForward, GQAttention, RoPE
from torchtitan.models.common.moe import MoE
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
        enable_weight_tying=True,
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=3072),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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
        enable_weight_tying=True,
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=3072),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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
        enable_weight_tying=True,
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=6144),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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
        enable_weight_tying=True,
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=9728),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=12288),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=17408),
            attention=GQAttention.Config(
                n_heads=40,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=25600),
            attention=GQAttention.Config(
                n_heads=64,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=64,
                num_shared_experts=0,
                top_k=8,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=3072),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=128,
                num_shared_experts=0,
                top_k=8,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=6144),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=1536,
                num_experts=128,
                num_shared_experts=0,
                top_k=8,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=12288),
            attention=GQAttention.Config(
                n_heads=64,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
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


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=qwen3_configs[flavor],
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
