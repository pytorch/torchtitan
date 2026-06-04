# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE, TransformerBlock
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.deepseek_v3 import build_mla_moe_layers
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import KimiK25Model
from .parallelize import parallelize_kimi_k2_5, pipeline_kimi_k2_5
from .state_dict_adapter import KimiK25StateDictAdapter
from .vision_encoder import (
    KimiK25VisionEncoder,
    Learnable2DInterpPosEmb,
    MultiModalProjector,
    VisionAttention,
    VisionEncoderBlock,
    VisionMLP,
    VisionRotaryEmbedding2D,
)

__all__ = [
    "parallelize_kimi_k2_5",
    "KimiK25Model",
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
_VL_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}


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
        param_init=_VL_LINEAR_INIT,
    )


def _vl_layernorm(dim: int, eps: float = 1e-5) -> LayerNorm.Config:
    # LayerNorm uses its built-in reset_parameters (weight=1, bias=0); no
    # param_init needed.
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
    init_pos_emb_time: int = 4,
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

    block = VisionEncoderBlock.Config(
        norm0=_vl_layernorm(dim),
        norm1=_vl_layernorm(dim),
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
        patch_embed_proj=_vl_linear(patch_dim, dim),
        pos_emb=Learnable2DInterpPosEmb.Config(
            height=init_pos_emb_height,
            width=init_pos_emb_width,
            num_frames=init_pos_emb_time,
            dim=dim,
        ),
        rotary_pos_emb=VisionRotaryEmbedding2D.Config(
            head_dim=head_dim, theta=rope_theta
        ),
        block=block,
        final_norm=_vl_layernorm(dim),
        projector=MultiModalProjector.Config(
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
        score_before_experts=False,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
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
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
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
            init_pos_emb_time=4,
            text_hidden_size=dim,
        ),
        # Debug vocab is tiny; keep the placeholder id within range.
        media_placeholder_token_id=vocab_size - 1,
    )


def _debugmodel_mm(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> KimiK25Model.Config:
    """Debug model fed by the shared (block-order) multimodal collator.

    Same as ``debugmodel`` but the vision encoder reorders incoming block-order
    patches to raster on entry (``input_patch_order="block"``), so the shared
    ``MMDataLoader`` can drive the raster-faithful MoonViT encoder.
    """
    config = _debugmodel(attn_backend, moe_comm_backend, non_blocking_capacity_factor)
    config.vision_encoder.input_patch_order = "block"
    return config


def _1t_a32b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> KimiK25Model.Config:
    """Full Kimi K2.5: ~1T-total / ~32B-active DeepSeekV3-style text tower
    (384 routed experts, top-8) + MoonViT3d vision tower.

    Values mirror the released ``KimiK25Config`` (text_config is a DeepSeekV3
    config with model_type "kimi_k2"). The released checkpoint ships int4
    weight quantization for serving; that is a serving artifact and is not
    reflected here — training quantization (if any) is chosen by the trainer
    config.
    """
    dim = 7168
    n_layers = 61
    vocab_size = 163840
    n_heads = 64
    q_lora_rank = 1536
    moe_hidden_dim = 2048  # moe_intermediate_size
    num_shared_experts = 1
    dense_hidden_dim = 18432  # intermediate_size
    rope_dim = 64  # qk_rope_head_dim
    num_experts = 384  # n_routed_experts
    n_dense_layers = 1  # first_k_dense_replace

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
        router_top_k=8,  # num_experts_per_tok
        router_score_func="sigmoid",
        # n_group == 1 and topk_group == 1 -> no node-limited routing.
        router_num_expert_groups=None,
        router_num_limited_groups=None,
        router_route_scale=2.827,  # routed_scaling_factor
        router_route_norm=True,  # norm_topk_prob
        score_before_experts=False,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
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
        rope=RoPE.Config(
            dim=rope_dim,
            # max_position_embeddings; overridden by training.seq_len.
            max_seq_len=262144,
            theta=50000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=64.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
        vision_encoder=_vision_encoder_config(
            dim=1152,  # vt_hidden_size
            ffn_dim=4304,  # vt_intermediate_size
            num_layers=27,  # vt_num_hidden_layers
            num_heads=16,  # vt_num_attention_heads
            patch_size=14,
            init_pos_emb_height=64,
            init_pos_emb_width=64,
            init_pos_emb_time=4,
            text_hidden_size=dim,
        ),
        media_placeholder_token_id=163605,
    )


kimi_k2_5_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_mm": _debugmodel_mm,
    "1T-A32B": _1t_a32b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
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
        pipelining_fn=pipeline_kimi_k2_5,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=KimiK25StateDictAdapter,
    )
