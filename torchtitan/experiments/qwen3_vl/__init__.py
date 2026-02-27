# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL: A Vision-Language Model based on Qwen3.

This module provides the Qwen3-VL implementation for multimodal training
with images and videos. Key features include:
- DeepStack: Visual features from intermediate ViT layers are added to
  early LLM hidden states for better multimodal understanding
- MRoPE: Multi-dimensional RoPE with interleaved temporal/height/width
  position encoding for vision tokens
- 2D RoPE + bilinear position interpolation in the vision encoder
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.models.common import FeedForward, GQAttention, RoPE
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3.model import Qwen3TransformerBlock
from torchtitan.protocols.model_spec import ModelSpec

from .model import Qwen3VLModel
from .parallelize import parallelize_qwen3_vl
from .state_dict_adapter import Qwen3VLStateDictAdapter
from .vision_encoder import Qwen3VLVisionEncoder

__all__ = [
    "parallelize_qwen3_vl",
    "Qwen3VLModel",
    "qwen3_vl_configs",
]


# Model configurations for different Qwen3-VL variants
qwen3_vl_configs = {
    # Debug model for testing
    "debugmodel": Qwen3VLModel.Config(
        vocab_size=151936,
        dim=256,
        n_layers=4,
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=512),
            attention=GQAttention.Config(
                n_heads=4,
                n_kv_heads=2,
                head_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        encoder=Qwen3VLVisionEncoder.Config(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
            deepstack_visual_indicies=[1, 2, 3],
        ),
        mrope_section=[8, 8, 8],
    ),
    # Debug MoE model for testing
    "debugmodel_moe": Qwen3VLModel.Config(
        vocab_size=151936,
        dim=256,
        n_layers=1,
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
            feed_forward=FeedForward.Config(hidden_dim=512),
            attention=GQAttention.Config(
                n_heads=4,
                n_kv_heads=2,
                head_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        encoder=Qwen3VLVisionEncoder.Config(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
            deepstack_visual_indicies=[1, 2, 3],
        ),
        mrope_section=[8, 8, 8],
    ),
    # Qwen3-VL 2B variant (based on Qwen3 1.7B LLM + ViT)
    "2B": Qwen3VLModel.Config(
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
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        encoder=Qwen3VLVisionEncoder.Config(
            dim=1024,
            ffn_dim=4096,
            n_layers=24,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            deepstack_visual_indicies=[5, 11, 17],
        ),
        mrope_section=[24, 20, 20],
    ),
    # Qwen3-VL 8B variant (based on Qwen3 8B LLM + ViT)
    "8B": Qwen3VLModel.Config(
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
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        encoder=Qwen3VLVisionEncoder.Config(
            dim=1152,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
            deepstack_visual_indicies=[8, 16, 24],
        ),
        mrope_section=[24, 20, 20],
    ),
    # Qwen3-VL 30B-A3B MoE variant
    "30B-A3B": Qwen3VLModel.Config(
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
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        encoder=Qwen3VLVisionEncoder.Config(
            dim=1152,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            deepstack_visual_indicies=[8, 16, 24],
        ),
        mrope_section=[24, 20, 20],
    ),
    # Qwen3-VL 235B-A22B MoE variant
    "235B-A22B": Qwen3VLModel.Config(
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
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        encoder=Qwen3VLVisionEncoder.Config(
            dim=1152,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
            deepstack_visual_indicies=[8, 16, 24],
        ),
        mrope_section=[24, 20, 20],
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="qwen3_vl",
        flavor=flavor,
        model=qwen3_vl_configs[flavor],
        parallelize_fn=parallelize_qwen3_vl,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3VLStateDictAdapter,
    )
