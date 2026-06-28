# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook

from torchtitan.distributed.pipeline_parallel import pipeline_vlm
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    Linear,
    RMSNorm,
    TransformerBlock,
)
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.deepseek_v3 import build_mla_moe_layers
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import KimiK25Model
from .parallelize import parallelize_kimi_k2_5
from .state_dict_adapter import KimiK25StateDictAdapter

from .vision_encoder import (
    KimiK25VisionEncoder,
    VisionProjector,
    VisionRotaryEmbedding2D,
)

__all__ = [
    "parallelize_kimi_k2_5",
    "KimiK25Model",
    "KimiK25StateDictAdapter",
    "KimiK25VisionEncoder",
    "VisionProjector",
    "VisionRotaryEmbedding2D",
    "model_registry",
    "kimi_k2_5_configs",
    "KIMI_K2_5_SPECIAL_TOKENS",
]


KIMI_K2_5_SPECIAL_TOKENS = {
    "image_token": "<|media_pad|>",
    "video_token": "<|media_pad|>",
    "vision_start_token": "<|media_begin|>",
    "vision_end_token": "<|media_end|>",
    "pad_token": "[PAD]",
}


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_POS_EMB_INIT = {"pos_embed": partial(nn.init.normal_, std=1.0)}


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


def _vl_layernorm(dim: int, eps: float = 1e-5) -> LayerNorm.Config:
    return LayerNorm.Config(normalized_shape=dim, eps=eps)


def _vision_encoder_config(
    *,
    dim: int,
    ffn_dim: int,
    num_layers: int,
    num_heads: int,
    patch_size: int = 14,
    in_channels: int = 3,
    init_pos_emb_height: int = 64,
    init_pos_emb_width: int = 64,
    rope_theta: float = 10000.0,
    merge_kernel_size: list[int] | None = None,
    text_hidden_size: int = 7168,
) -> KimiK25VisionEncoder.Config:
    """Build a fully-specified KimiK25VisionEncoder.Config (MoonViT3d)."""
    if merge_kernel_size is None:
        merge_kernel_size = [2, 2]
    patch_dim = in_channels * patch_size * patch_size
    head_dim = dim // num_heads
    merged_dim = dim * merge_kernel_size[0] * merge_kernel_size[1]

    block = VisionTransformerBlock.Config(
        norm1=_vl_layernorm(dim),
        norm2=_vl_layernorm(dim),
        attn=VisionAttention.Config(
            dim=dim,
            num_heads=num_heads,
            wq=_vl_linear(dim, dim),
            wk=_vl_linear(dim, dim),
            wv=_vl_linear(dim, dim),
            proj=_vl_linear(dim, dim),
        ),
        mlp=VisionMLP.Config(
            fc1=_vl_linear(dim, ffn_dim),
            fc2=_vl_linear(ffn_dim, dim),
        ),
    )

    return KimiK25VisionEncoder.Config(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_size=patch_size,
        in_channels=in_channels,
        merge_kernel_size=merge_kernel_size,
        text_hidden_size=text_hidden_size,
        init_pos_emb_height=init_pos_emb_height,
        init_pos_emb_width=init_pos_emb_width,
        param_init=_POS_EMB_INIT,
        patch_embed_proj=_vl_linear(patch_dim, dim),
        rotary_pos_emb=VisionRotaryEmbedding2D.Config(
            head_dim=head_dim, theta=rope_theta
        ),
        block=block,
        final_norm=_vl_layernorm(dim),
        projector=VisionProjector.Config(
            vt_hidden_size=dim,
            merged_dim=merged_dim,
            pre_norm=_vl_layernorm(dim),
            linear_1=_vl_linear(merged_dim, merged_dim),
            linear_2=_vl_linear(merged_dim, text_hidden_size),
        ),
    )


def _build_kimi_layers(**kwargs) -> list[TransformerBlock.Config]:
    """Thin wrapper: ``build_mla_moe_layers`` with Kimi K2.5's own inits."""
    return build_mla_moe_layers(
        **kwargs,
        linear_init=_LINEAR_INIT,
        norm_init=_NORM_INIT,
        depth_init=_depth_init,
        depth_experts_init=_depth_experts_init,
    )


def _debugmodel(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> KimiK25Model.Config:
    dim = 256
    n_layers = 6
    vocab_size = 2048
    n_heads = 16
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    rope_dim = 64
    num_experts = 8
    n_dense_layers = 1

    layers = _build_kimi_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    config = KimiK25Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        vision_encoder=_vision_encoder_config(
            dim=256,
            ffn_dim=512,
            num_layers=4,
            num_heads=4,
            patch_size=14,
            init_pos_emb_height=16,
            init_pos_emb_width=16,
            text_hidden_size=dim,
        ),
    )
    return config


def _moonlight_16b_a3b_config(
    *,
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    rope_theta: float,
    max_seq_len: int,
    vision_encoder: "KimiK25VisionEncoder.Config | None" = None,
) -> KimiK25Model.Config:
    """Shared Moonshot 16B-A3B DeepSeekV3 text tower (MLA + sigmoid-routed MoE).

    Used by both Moonlight (text-only) and Kimi-VL (with a vision encoder): they
    share the architecture (no q-LoRA, no RoPE scaling, 64 experts top-6); only
    the RoPE ``theta`` / ``max_seq_len`` and the vision tower differ.
    """
    dim = 2048
    vocab_size = 163840
    layers = _build_kimi_layers(
        n_layers=27,
        n_dense_layers=1,
        dim=dim,
        n_heads=16,
        q_lora_rank=0,  # q_lora_rank null -> separate wq
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=11264,
        moe_hidden_dim=1408,
        num_experts=64,
        num_shared_experts=2,
        router_top_k=6,
        router_score_func="sigmoid",
        router_num_expert_groups=None,
        router_num_limited_groups=None,
        router_route_scale=2.446,
        router_route_norm=True,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=ComplexRoPE.Config(dim=64, max_seq_len=max_seq_len, theta=rope_theta),
    )
    return KimiK25Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        vision_encoder=vision_encoder,
    )


def _moonlight_16b_a3b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> KimiK25Model.Config:
    """Moonlight 16B-A3B: the text-only DeepSeekV3 sibling (no vision tower).

    Mirrors the released ``moonlight_16b_a3b`` config (``model_type
    "deepseek_v3"``, rope_theta 50000). ``vision_encoder=None`` makes this a pure
    text config of ``KimiK25Model``.
    """
    return _moonlight_16b_a3b_config(
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope_theta=50000.0,
        max_seq_len=8192,
        vision_encoder=None,
    )


def _kimi_vl_a3b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> KimiK25Model.Config:
    """Kimi-VL 16B-A3B: Moonlight 16B-A3B text tower (rope_theta 800000) + 2D MoonViT vision.

    Kimi-VL uses the original 2D MoonViT, which is the ``t=1`` case of MoonViT3d
    It reuses ``KimiK25VisionEncoder``; with image inputs (``t=1``) the temporal
    embedding is never applied.
    """
    config = _moonlight_16b_a3b_config(
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope_theta=800000.0,
        max_seq_len=131072,
        vision_encoder=_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=14,
            init_pos_emb_height=64,
            init_pos_emb_width=64,
            text_hidden_size=2048,
        ),
    )
    return config


def _1t_a32b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> KimiK25Model.Config:
    """Full Kimi K2.5: ~1T-total / ~32B-active DeepSeekV3-style text tower
    (384 routed experts, top-8) + MoonViT3d vision tower.

    Values mirror the released bf16 ``KimiK25Config`` (text_config is a
    DeepSeekV3 config with model_type "kimi_k2").
    """
    dim = 7168
    n_layers = 61
    vocab_size = 163840
    n_heads = 64
    q_lora_rank = 1536
    moe_hidden_dim = 2048
    num_shared_experts = 1
    dense_hidden_dim = 18432
    rope_dim = 64  # qk_rope_head_dim
    num_experts = 384
    n_dense_layers = 1

    layers = _build_kimi_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=8,
        router_score_func="sigmoid",
        router_num_expert_groups=None,
        router_num_limited_groups=None,
        router_route_scale=2.827,
        router_route_norm=True,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=262144,
            theta=50000.0,
            scaling="yarn",
            rope_factor=64.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return KimiK25Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        vision_encoder=_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=14,
            init_pos_emb_height=64,
            init_pos_emb_width=64,
            text_hidden_size=dim,
        ),
    )


kimi_k2_5_configs = {
    "debugmodel": _debugmodel,
    "moonlight-16B-A3B": _moonlight_16b_a3b,
    "Kimi-VL-A3B": _kimi_vl_a3b,
    "1T-A32B": _1t_a32b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = kimi_k2_5_configs[flavor](
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            c.build().convert(config)
    return ModelSpec(
        name="kimi_k2_5",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_kimi_k2_5,
        pipelining_fn=pipeline_vlm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=KimiK25StateDictAdapter,
    )
