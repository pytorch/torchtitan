# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.models.common import Embedding, Linear, RoPE, TransformerBlock
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_experts_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.nn_modules import LayerNorm, RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.qwen3.model import Qwen3TransformerBlock
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec
from .model import Qwen3VLModel
from .parallelize import parallelize_qwen3_vl
from .rope import MRoPE
from .state_dict_adapter import Qwen3VLStateDictAdapter
from .vision_encoder import (
    PatchEmbed,
    PatchMerger,
    Qwen3VLVisionEncoder,
    VisionAttention,
    VisionMLP,
    VisionRotaryEmbedding,
    VisionTransformerBlock,
)

__all__ = [
    "parallelize_qwen3_vl",
    "Qwen3VLModel",
    "qwen3_vl_configs",
    "QWEN3_VL_SPECIAL_TOKENS",
]

QWEN3_VL_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)}

_EPS = 1e-6


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _vl_linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
    )


def _qwen3_vl_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _vl_layer_norm(normalized_shape: int) -> LayerNorm.Config:
    return LayerNorm.Config(normalized_shape=normalized_shape, eps=_EPS)


def _build_qwen3_vl_vision_block(
    *, dim: int, ffn_dim: int, n_heads: int
) -> VisionTransformerBlock.Config:
    return VisionTransformerBlock.Config(
        attn=VisionAttention.Config(
            dim=dim,
            n_heads=n_heads,
            qkv=_vl_linear(dim, dim * 3),
            proj=_vl_linear(dim, dim),
        ),
        mlp=VisionMLP.Config(
            fc1=_vl_linear(dim, ffn_dim),
            fc2=_vl_linear(ffn_dim, dim),
        ),
        norm1=_vl_layer_norm(dim),
        norm2=_vl_layer_norm(dim),
    )


def _build_qwen3_vl_merger(
    *,
    dim: int,
    out_hidden_size: int,
    spatial_merge_size: int,
    use_postshuffle_norm: bool,
) -> PatchMerger.Config:
    merged_hidden_size = dim * (spatial_merge_size**2)
    norm_dim = merged_hidden_size if use_postshuffle_norm else dim
    return PatchMerger.Config(
        hidden_size=dim,
        spatial_merge_size=spatial_merge_size,
        fc1=_vl_linear(merged_hidden_size, merged_hidden_size),
        fc2=_vl_linear(merged_hidden_size, out_hidden_size),
        norm=_vl_layer_norm(norm_dim),
        use_postshuffle_norm=use_postshuffle_norm,
    )


def _vl_vision_encoder_config(
    *,
    dim: int,
    ffn_dim: int,
    n_layers: int,
    n_heads: int,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    out_hidden_size: int,
    num_position_embeddings: int,
    deepstack_visual_indices: list[int],
    in_channels: int = 3,
) -> Qwen3VLVisionEncoder.Config:
    """Build a fully-specified Qwen3VLVisionEncoder.Config."""
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    head_dim = dim // n_heads
    return Qwen3VLVisionEncoder.Config(
        dim=dim,
        n_heads=n_heads,
        spatial_merge_size=spatial_merge_size,
        num_position_embeddings=num_position_embeddings,
        deepstack_visual_indices=deepstack_visual_indices,
        patch_embed=PatchEmbed.Config(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            proj=_vl_linear(patch_dim, dim),
        ),
        rotary_pos_emb=VisionRotaryEmbedding.Config(dim=head_dim // 2),
        layers=[
            _build_qwen3_vl_vision_block(dim=dim, ffn_dim=ffn_dim, n_heads=n_heads)
            for _ in range(n_layers)
        ],
        merger=_build_qwen3_vl_merger(
            dim=dim,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            use_postshuffle_norm=False,
        ),
        deepstack_mergers=[
            _build_qwen3_vl_merger(
                dim=dim,
                out_hidden_size=out_hidden_size,
                spatial_merge_size=spatial_merge_size,
                use_postshuffle_norm=True,
            )
            for _ in range(len(deepstack_visual_indices))
        ],
        param_init=_POS_EMBED_INIT,
    )


def _build_qwen3_vl_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    attn_backend: str,
    rope: RoPE.Config,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for dense Qwen3-VL models with depth-scaled inits."""
    inner_attention = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen3_vl_norm(dim),
                ffn_norm=_qwen3_vl_norm(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    rope=rope,
                    qk_norm=_qwen3_vl_norm(head_dim),
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _build_qwen3_vl_moe_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    rope: RoPE.Config,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for MoE Qwen3-VL models with depth-scaled inits."""
    inner_attention = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen3_vl_norm(dim),
                ffn_norm=_qwen3_vl_norm(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    rope=rope,
                    qk_norm=_qwen3_vl_norm(head_dim),
                ),
                moe=make_moe_config(
                    num_experts=num_experts,
                    router=make_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=_depth_init(layer_id),
                        top_k=top_k,
                        score_func="softmax",
                        route_norm=True,
                    ),
                    experts=make_experts_config(
                        dim=dim,
                        hidden_dim=moe_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        param_init=_depth_experts_init(layer_id),
                        score_before_experts=False,
                        comm_backend=moe_comm_backend,
                        non_blocking_capacity_factor=non_blocking_capacity_factor,
                    ),
                ),
            )
        )
    return layers


def _debugmodel(attn_backend: str) -> Qwen3VLModel.Config:
    dim = 256
    head_dim = 64
    n_layers = 4
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_vl_layers(
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            hidden_dim=512,
            rope=MRoPE.Config(
                dim=head_dim,
                max_seq_len=4096,
                theta=1000000.0,
                mrope_section=[8, 8, 8],
            ),
        ),
        vision_encoder=_vl_vision_encoder_config(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
            deepstack_visual_indices=[1, 2, 3],
        ),
    )


def _debugmodel_moe(
    attn_backend: str,
    moe_comm_backend: str = "standard",
) -> Qwen3VLModel.Config:
    dim = 256
    head_dim = 64
    n_layers = 1
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_vl_moe_layers(
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=8,
            top_k=4,
            moe_comm_backend=moe_comm_backend,
            rope=MRoPE.Config(
                dim=head_dim,
                max_seq_len=4096,
                theta=1000000.0,
                mrope_section=[8, 8, 8],
            ),
        ),
        vision_encoder=_vl_vision_encoder_config(
            dim=256,
            ffn_dim=512,
            n_layers=4,
            n_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
            deepstack_visual_indices=[1, 2, 3],
        ),
    )


def _2b(attn_backend: str) -> Qwen3VLModel.Config:
    dim = 2048
    head_dim = 128
    n_layers = 28
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_vl_layers(
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=6144,
            rope=MRoPE.Config(
                dim=head_dim,
                max_seq_len=32768,
                theta=5000000.0,
                mrope_section=[24, 20, 20],
            ),
        ),
        vision_encoder=_vl_vision_encoder_config(
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
        ),
    )


def _8b(attn_backend: str) -> Qwen3VLModel.Config:
    dim = 4096
    head_dim = 128
    n_layers = 36
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_vl_layers(
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=12288,
            rope=MRoPE.Config(
                dim=head_dim,
                max_seq_len=32768,
                theta=5000000.0,
                mrope_section=[24, 20, 20],
            ),
        ),
        vision_encoder=_vl_vision_encoder_config(
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
        ),
    )


# Qwen3-VL MoE models


def _30b_a3b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
) -> Qwen3VLModel.Config:
    dim = 2048
    head_dim = 128
    n_layers = 48
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_vl_moe_layers(
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=128,
            top_k=8,
            moe_comm_backend=moe_comm_backend,
            rope=MRoPE.Config(
                dim=head_dim,
                max_seq_len=32768,
                theta=5000000.0,
                mrope_section=[24, 20, 20],
            ),
        ),
        vision_encoder=_vl_vision_encoder_config(
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
        ),
    )


def _235b_a22b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
) -> Qwen3VLModel.Config:
    dim = 4096
    head_dim = 128
    n_layers = 94
    vocab_size = 151936
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_vl_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_vl_moe_layers(
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=1536,
            num_experts=128,
            top_k=8,
            moe_comm_backend=moe_comm_backend,
            rope=MRoPE.Config(
                dim=head_dim,
                max_seq_len=32768,
                theta=5000000.0,
                mrope_section=[24, 20, 20],
            ),
        ),
        vision_encoder=_vl_vision_encoder_config(
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
        ),
    )


qwen3_vl_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_moe": _debugmodel_moe,
    "2B": _2b,
    "8B": _8b,
    "30B-A3B": _30b_a3b,
    "235B-A22B": _235b_a22b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    kwargs = dict(attn_backend=attn_backend)
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    config = qwen3_vl_configs[flavor](**kwargs)
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            c.build().convert(config)
    return ModelSpec(
        name="qwen3_vl",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_vl,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3VLStateDictAdapter,
    )
