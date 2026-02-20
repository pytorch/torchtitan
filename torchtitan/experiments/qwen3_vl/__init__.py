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

from dataclasses import fields
from typing import Any

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .datasets.mm_datasets import build_mm_dataloader
from .infra.parallelize import parallelize_qwen3_vl
from .model.args import Qwen3VLModelArgs, Qwen3VLVisionEncoderArgs, Qwen3VLTextConfig
from .model.model import Qwen3VLModel

__all__ = [
    "parallelize_qwen3_vl",
    "Qwen3VLModelArgs",
    "Qwen3VLVisionEncoderArgs",
    "Qwen3VLModel",
    "qwen3_vl_args",
]


# Model configurations for different Qwen3-VL variants
# Reference: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
qwen3_vl_args = {
    # Debug model for testing
    "debugmodel": Qwen3VLModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=64,
        dim=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=512,
        rope_theta=1000000,
        encoder=Qwen3VLVisionEncoderArgs(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,  # 32x32 grid
            deepstack_visual_indicies=[1, 2, 3],
        ),
        text_config=Qwen3VLTextConfig(mrope_section=[8, 8, 8]),
    ),
    # Debug MoE model for testing
    "debugmodel_moe": Qwen3VLModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=64,
        dim=256,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=512,
        rope_theta=1000000,
        moe_enabled=True,
        moe_inter_dim=768,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
        ),
        encoder=Qwen3VLVisionEncoderArgs(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,  # 32x32 grid
            deepstack_visual_indicies=[1, 2, 3],
        ),
        text_config=Qwen3VLTextConfig(mrope_section=[8, 8, 8]),
    ),
    # Qwen3-VL 2B variant (based on Qwen3 0.6B LLM + ViT)
    "2B": Qwen3VLModelArgs(
        vocab_size=151936,
        max_seq_len=32768,
        head_dim=128,
        dim=1536,
        n_layers=28,
        n_heads=12,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=8960,
        rope_theta=1000000,
        encoder=Qwen3VLVisionEncoderArgs(
            dim=1280,
            ffn_dim=5120,
            n_layers=32,
            n_heads=16,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=1536,
            num_position_embeddings=4096,  # 64x64 grid
            deepstack_visual_indicies=[7, 15, 23],
        ),
        text_config=Qwen3VLTextConfig(mrope_section=[24, 20, 20]),
    ),
    # Qwen3-VL 8B variant (based on Qwen3 8B LLM + ViT)
    "8B": Qwen3VLModelArgs(
        vocab_size=151936,
        max_seq_len=32768,
        head_dim=128,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=14336,
        rope_theta=1000000,
        encoder=Qwen3VLVisionEncoderArgs(
            dim=1280,
            ffn_dim=5120,
            n_layers=32,
            n_heads=16,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=4096,  # 64x64 grid
            deepstack_visual_indicies=[7, 15, 23],
        ),
        text_config=Qwen3VLTextConfig(mrope_section=[24, 20, 20]),
    ),
    # Qwen3-VL 72B variant (based on Qwen3 72B LLM + ViT)
    "72B": Qwen3VLModelArgs(
        vocab_size=151936,
        max_seq_len=32768,
        head_dim=128,
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=29568,
        rope_theta=1000000,
        encoder=Qwen3VLVisionEncoderArgs(
            dim=1280,
            ffn_dim=5120,
            n_layers=32,
            n_heads=16,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=8192,
            num_position_embeddings=4096,  # 64x64 grid
            deepstack_visual_indicies=[7, 15, 23],
        ),
        text_config=Qwen3VLTextConfig(mrope_section=[24, 20, 20]),
    ),
}


def get_train_spec() -> TrainSpec:
    """Return the training specification for Qwen3-VL."""
    return TrainSpec(
        model_cls=Qwen3VLModel,
        model_args=qwen3_vl_args,
        parallelize_fn=parallelize_qwen3_vl,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
