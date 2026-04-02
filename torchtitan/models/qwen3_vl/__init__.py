# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.config import Function
from torchtitan.models.common import Embedding, FeedForward, GQAttention, Linear, RoPE
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.qwen3 import expand_layer_configs
from torchtitan.models.qwen3.model import Qwen3TransformerBlock
from torchtitan.protocols.model_spec import ModelSpec

from .model import Qwen3VLModel
from .parallelize import parallelize_qwen3_vl
from .special_tokens import Qwen3VLSpecialTokens
from .state_dict_adapter import Qwen3VLStateDictAdapter
from .vision_encoder import Qwen3VLVisionEncoder

__all__ = [
    "parallelize_qwen3_vl",
    "Qwen3VLModel",
    "Qwen3VLSpecialTokens",
    "qwen3_vl_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_LINEAR_DEPTH_INIT = Function.Config(
    fn=lambda layer_id: {  # pyrefly: ignore [bad-argument-type]
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }
)
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)}
_EXPERTS_DEPTH_INIT = Function.Config(
    fn=lambda layer_id: {  # pyrefly: ignore [bad-argument-type]
        "w1": partial(nn.init.trunc_normal_, std=0.02),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }
)


def _output_linear_init(dim: int):
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _debugmodel():
    dim = 256
    head_dim = 64
    return Qwen3VLModel.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=4,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_LINEAR_INIT),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=512,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=4,
                n_kv_heads=2,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        vision_encoder=Qwen3VLVisionEncoder.Config(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
            deepstack_visual_indices=[1, 2, 3],
            linear=Linear.Config(bias=True, param_init=_LINEAR_INIT),
            param_init=_POS_EMBED_INIT,
        ),
        mrope_section=[8, 8, 8],
    )


def _debugmodel_moe():
    dim = 256
    head_dim = 64
    return Qwen3VLModel.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=1,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_LINEAR_INIT),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=64,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=8,
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=512,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=4,
                n_kv_heads=2,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        vision_encoder=Qwen3VLVisionEncoder.Config(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
            deepstack_visual_indices=[1, 2, 3],
            linear=Linear.Config(bias=True, param_init=_LINEAR_INIT),
            param_init=_POS_EMBED_INIT,
        ),
        mrope_section=[8, 8, 8],
    )


def _2b():
    dim = 2048
    head_dim = 128
    return Qwen3VLModel.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=28,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_SKIP_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=6144,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        vision_encoder=Qwen3VLVisionEncoder.Config(
            dim=1024,
            ffn_dim=4096,
            n_layers=24,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            deepstack_visual_indices=[5, 11, 17],
            linear=Linear.Config(bias=True, param_init=_LINEAR_INIT),
            param_init=_POS_EMBED_INIT,
        ),
        mrope_section=[24, 20, 20],
    )


def _8b():
    dim = 4096
    head_dim = 128
    return Qwen3VLModel.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=36,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=12288,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        vision_encoder=Qwen3VLVisionEncoder.Config(
            dim=1152,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
            deepstack_visual_indices=[8, 16, 24],
            linear=Linear.Config(bias=True, param_init=_LINEAR_INIT),
            param_init=_POS_EMBED_INIT,
        ),
        mrope_section=[24, 20, 20],
    )


# Qwen3-VL MoE models


def _30b_a3b():
    dim = 2048
    head_dim = 128
    return Qwen3VLModel.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=48,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=128,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=8,
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=6144,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=4,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        vision_encoder=Qwen3VLVisionEncoder.Config(
            dim=1152,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            deepstack_visual_indices=[8, 16, 24],
            linear=Linear.Config(bias=True, param_init=_LINEAR_INIT),
            param_init=_POS_EMBED_INIT,
        ),
        mrope_section=[24, 20, 20],
    )


def _235b_a22b():
    dim = 4096
    head_dim = 128
    return Qwen3VLModel.Config(
        vocab_size=151936,
        dim=dim,
        n_layers=94,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
        layer=Qwen3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=1536,
                num_experts=128,
                score_before_experts=False,
                router=TokenChoiceTopKRouter.Config(
                    top_k=8,
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=12288,
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=64,
                n_kv_heads=4,
                head_dim=head_dim,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                q_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                k_norm=RMSNorm.Config(eps=1e-6, param_init=_NORM_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=32768,
            theta=5000000.0,
            backend="cos_sin",
        ),
        vision_encoder=Qwen3VLVisionEncoder.Config(
            dim=1152,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
            deepstack_visual_indices=[8, 16, 24],
            linear=Linear.Config(bias=True, param_init=_LINEAR_INIT),
            param_init=_POS_EMBED_INIT,
        ),
        mrope_section=[24, 20, 20],
    )


qwen3_vl_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_moe": _debugmodel_moe,
    "2B": _2b,
    "8B": _8b,
    "30B-A3B": _30b_a3b,
    "235B-A22B": _235b_a22b,
}


def model_registry(flavor: str) -> ModelSpec:
    config = qwen3_vl_configs[flavor]()
    expand_layer_configs(config)
    return ModelSpec(
        name="qwen3_vl",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_vl,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3VLStateDictAdapter,
    )
