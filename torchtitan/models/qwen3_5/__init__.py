# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from typing import Literal

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook

from torchtitan.models.common import (  # noqa: F401
    Conv1d,
    Embedding,
    Linear,
    SigmoidGatedFeedForward,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.param_init import depth_scaled_std  # noqa: F401
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter

from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    GatedDeltaKernel,
    GatedDeltaNet,
    OffsetRMSNorm,
    Qwen35Attention,
    Qwen35Model,
    Qwen35TransformerBlock,
    RMSNormGated,
)
from .parallelize import parallelize_qwen3_5, pipeline_qwen3_5
from .rope import MRoPE
from .state_dict_adapter import Qwen35StateDictAdapter
from .vision_encoder import (
    PatchMerger,
    Qwen35VisionEncoder,
    VisionAttention,
    VisionMLP,
    VisionRotaryEmbedding,
    VisionTransformerBlock,
)

__all__ = [
    "parallelize_qwen3_5",
    "Qwen35Model",
    "qwen3_5_configs",
    "QWEN3_5_SPECIAL_TOKENS",
]

QWEN3_5_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02, a=-3 * 0.02, b=3 * 0.02),
    "bias": nn.init.zeros_,
}
_OFFSET_NORM_INIT = {"weight": nn.init.zeros_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)}

_EPS = 1e-6


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    std = depth_scaled_std(0.02, layer_id)
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    # Only w2 (output projection) gets the depth-scaled std; w1/w3 are inputs.
    std = depth_scaled_std(0.02, layer_id)
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02, a=-3 * 0.02, b=3 * 0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "w3_EFD": partial(nn.init.trunc_normal_, std=0.02, a=-3 * 0.02, b=3 * 0.02),
    }


def _a_log_init(param: nn.Parameter) -> None:
    param.data.uniform_(1e-6, 16.0).log_()


def _linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
    )


def _offset_norm(dim: int) -> OffsetRMSNorm.Config:
    return OffsetRMSNorm.Config(dim=dim, eps=_EPS, param_init=_OFFSET_NORM_INIT)


def _shared_experts_config(
    *, dim: int, hidden_dim: int, layer_id: int
) -> SigmoidGatedFeedForward.Config:
    """Build Qwen3.5's sigmoid-gated shared-expert config (SwiGLU FFN + gate)."""
    ffn = make_ffn_config(
        dim=dim,
        hidden_dim=hidden_dim,
        w1w3_param_init=_LINEAR_INIT,
        w2_param_init=_depth_init(layer_id),
    )
    return SigmoidGatedFeedForward.Config(
        w1=ffn.w1,
        w2=ffn.w2,
        w3=ffn.w3,
        gate=Linear.Config(in_features=dim, out_features=1, param_init=_LINEAR_INIT),
    )


def _qwen35_vision_encoder_config(
    *,
    dim: int,
    ffn_dim: int,
    num_layers: int,
    num_heads: int,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    out_hidden_size: int,
    num_position_embeddings: int,
    layer_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    in_channels: int = 3,
) -> Qwen35VisionEncoder.Config:
    """Build a fully-specified Qwen35VisionEncoder.Config."""
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    merged_hidden_size = dim * (spatial_merge_size**2)
    head_dim = dim // num_heads
    _norm = LayerNorm.Config(normalized_shape=dim, eps=layer_norm_eps)
    return Qwen35VisionEncoder.Config(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        spatial_merge_size=spatial_merge_size,
        num_position_embeddings=num_position_embeddings,
        patch_embed_proj=_linear(patch_dim, dim),
        block=VisionTransformerBlock.Config(
            norm1=_norm,
            norm2=_norm,
            attn=VisionAttention.Config(
                dim=dim,
                num_heads=num_heads,
                wq=_linear(dim, dim),
                wk=_linear(dim, dim),
                wv=_linear(dim, dim),
                proj=_linear(dim, dim),
            ),
            mlp=VisionMLP.Config(
                fc1=_linear(dim, ffn_dim),
                fc2=_linear(ffn_dim, dim),
            ),
        ),
        rotary_pos_emb=VisionRotaryEmbedding.Config(
            dim=head_dim // 2, theta=rope_theta
        ),
        merger=PatchMerger.Config(
            spatial_merge_size=spatial_merge_size,
            merged_hidden_size=merged_hidden_size,
            norm=LayerNorm.Config(normalized_shape=dim, eps=layer_norm_eps),
            fc1=_linear(merged_hidden_size, merged_hidden_size),
            fc2=_linear(merged_hidden_size, out_hidden_size),
        ),
        param_init=_POS_EMBED_INIT,
    )


def _qwen35_attention_config(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope: MRoPE.Config,
    attn_backend: str,
    layer_id: int,
) -> Qwen35Attention.Config:
    """Build a fully-specified Qwen35Attention.Config."""
    inner_attention = get_attention_config(attn_backend)
    return Qwen35Attention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope=rope,
        wq=Linear.Config(
            in_features=dim,
            out_features=n_heads * head_dim * 2,
            param_init=_LINEAR_INIT,
        ),
        wk=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        wv=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        wo=Linear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        q_norm=_offset_norm(head_dim),
        k_norm=_offset_norm(head_dim),
        inner_attention=inner_attention,
    )


def _qwen35_deltanet_config(
    *,
    dim: int,
    n_key_heads: int,
    n_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    layer_id: int,
    conv_kernel_size: int = 4,
    fla_backend: Literal[
        "fla_chunked", "fla_fused_recurrent", "torch_native"
    ] = "fla_chunked",
) -> GatedDeltaNet.Config:
    """Build a fully-specified GatedDeltaNet.Config."""
    key_dim = n_key_heads * key_head_dim
    value_dim = n_value_heads * value_head_dim

    def _proj(in_f: int, out_f: int, init: dict) -> Linear.Config:
        return Linear.Config(
            in_features=in_f, out_features=out_f, bias=False, param_init=init
        )

    def _conv(channels: int) -> Conv1d.Config:
        # Depthwise causal conv (groups == channels). Causal left-padding is
        # applied in the forward, so padding=0 here.
        return Conv1d.Config(
            in_channels=channels,
            out_channels=channels,
            kernel_size=conv_kernel_size,
            groups=channels,
            padding=0,
            bias=False,
        )

    return GatedDeltaNet.Config(
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel_size=conv_kernel_size,
        in_proj_q=_proj(dim, key_dim, _LINEAR_INIT),
        in_proj_k=_proj(dim, key_dim, _LINEAR_INIT),
        in_proj_v=_proj(dim, value_dim, _LINEAR_INIT),
        in_proj_z=_proj(dim, value_dim, _LINEAR_INIT),
        in_proj_a=_proj(dim, n_value_heads, _LINEAR_INIT),
        in_proj_b=_proj(dim, n_value_heads, _LINEAR_INIT),
        conv_q=_conv(key_dim),
        conv_k=_conv(key_dim),
        conv_v=_conv(value_dim),
        kernel=GatedDeltaKernel.Config(backend=fla_backend),
        norm=RMSNormGated.Config(
            dim=value_head_dim,
            eps=1e-6,
            param_init={"weight": nn.init.ones_},
        ),
        out_proj=_proj(value_dim, dim, _depth_init(layer_id)),
        param_init={
            "A_log": _a_log_init,
            "dt_bias": nn.init.ones_,
        },
    )


def _build_qwen35_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope: MRoPE.Config,
    hidden_dim: int,
    n_key_heads: int,
    n_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    full_attention_interval: int = 4,
    attn_backend: str,
    fla_backend: Literal[
        "fla_chunked", "fla_fused_recurrent", "torch_native"
    ] = "fla_chunked",
) -> list[Qwen35TransformerBlock.Config]:
    """Build per-layer configs for dense Qwen3.5 models."""
    layers = []
    for layer_id in range(n_layers):
        is_full = (layer_id + 1) % full_attention_interval == 0

        attention = (
            _qwen35_attention_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rope=rope,
                attn_backend=attn_backend,
                layer_id=layer_id,
            )
            if is_full
            else None
        )
        deltanet = (
            _qwen35_deltanet_config(
                dim=dim,
                n_key_heads=n_key_heads,
                n_value_heads=n_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                layer_id=layer_id,
                fla_backend=fla_backend,
            )
            if not is_full
            else None
        )

        layers.append(
            Qwen35TransformerBlock.Config(
                attention=attention,
                delta_net=deltanet,
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1w3_param_init=_LINEAR_INIT,
                    w2_param_init=_depth_init(layer_id),
                ),
                attention_norm=_offset_norm(dim),
                ffn_norm=_offset_norm(dim),
            )
        )
    return layers


def _build_qwen35_moe_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope: MRoPE.Config,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    shared_expert_hidden_dim: int,
    n_key_heads: int,
    n_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    full_attention_interval: int = 4,
    attn_backend: str,
    fla_backend: Literal[
        "fla_chunked", "fla_fused_recurrent", "torch_native"
    ] = "fla_chunked",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
) -> list[Qwen35TransformerBlock.Config]:
    """Build per-layer configs for MoE Qwen3.5 models with shared expert."""
    layers = []
    for layer_id in range(n_layers):
        is_full = (layer_id + 1) % full_attention_interval == 0

        attention = (
            _qwen35_attention_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rope=rope,
                attn_backend=attn_backend,
                layer_id=layer_id,
            )
            if is_full
            else None
        )
        deltanet = (
            _qwen35_deltanet_config(
                dim=dim,
                n_key_heads=n_key_heads,
                n_value_heads=n_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                layer_id=layer_id,
                fla_backend=fla_backend,
            )
            if not is_full
            else None
        )

        layers.append(
            Qwen35TransformerBlock.Config(
                attention=attention,
                delta_net=deltanet,
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
                    routed_experts=make_routed_experts_config(
                        dim=dim,
                        hidden_dim=moe_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        param_init=_depth_experts_init(layer_id),
                        comm_backend=moe_comm_backend,
                        non_blocking_capacity_factor=non_blocking_capacity_factor,
                    ),
                    shared_experts=_shared_experts_config(
                        dim=dim,
                        hidden_dim=shared_expert_hidden_dim,
                        layer_id=layer_id,
                    ),
                ),
                attention_norm=_offset_norm(dim),
                ffn_norm=_offset_norm(dim),
            )
        )
    return layers


def _debugmodel(attn_backend: str) -> Qwen35Model.Config:
    """Debug config for Qwen3.5 with vision encoder."""
    dim = 256
    head_dim = 64
    rotary_dim = 16
    n_layers = 8
    vocab_size = 248320
    # mrope_section sum must equal rotary_dim / 2 (8 for rotary_dim=16).
    # Real models use [11, 11, 10] with rotary_dim=64.
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=4096,
                theta=10_000_000.0,
                mrope_section=[3, 3, 2],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=512,
            n_key_heads=2,
            n_value_heads=4,
            key_head_dim=64,
            value_head_dim=64,
            fla_backend="fla_chunked",
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=256,
            ffn_dim=512,
            num_layers=4,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
        ),
    )


def _debugmodel_moe(
    attn_backend: str,
    moe_comm_backend: str = "standard",
) -> Qwen35Model.Config:
    """Debug MoE config for Qwen3.5 with shared expert."""
    dim = 256
    head_dim = 64
    rotary_dim = 16
    n_layers = 4
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=4096,
                theta=10_000_000.0,
                mrope_section=[3, 3, 2],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=256,
            num_experts=8,
            top_k=2,
            shared_expert_hidden_dim=256,
            n_key_heads=2,
            n_value_heads=4,
            key_head_dim=64,
            value_head_dim=64,
            moe_comm_backend=moe_comm_backend,
            fla_backend="fla_chunked",
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=256,
            ffn_dim=512,
            num_layers=2,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
        ),
    )


def _0_8b(attn_backend: str) -> Qwen35Model.Config:
    """Qwen3.5-0.8B dense config with vision encoder.

    NOTE: HF config has tie_word_embeddings=true. Torchtitan doesn't support
    tied embeddings yet, so we use a separate lm_head. Checkpoint conversion
    must handle this.
    """
    dim = 1024
    head_dim = 256
    rotary_dim = 64  # partial_rotary_factor=0.25 → head_dim * 0.25
    n_layers = 24
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=8,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=3584,
            n_key_heads=16,
            n_value_heads=16,
            key_head_dim=128,
            value_head_dim=128,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=768,
            ffn_dim=3072,
            num_layers=12,
            num_heads=12,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=1024,
            num_position_embeddings=2304,
        ),
    )


def _2b(attn_backend: str) -> Qwen35Model.Config:
    """Qwen3.5-2B dense config with vision encoder.

    NOTE: HF config has tie_word_embeddings=true. Torchtitan doesn't support
    tied embeddings yet, so we use a separate lm_head. Checkpoint conversion
    must handle this.
    """
    dim = 2048
    head_dim = 256
    rotary_dim = 64  # partial_rotary_factor=0.25
    n_layers = 24
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=8,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=6144,
            n_key_heads=16,
            n_value_heads=16,
            key_head_dim=128,
            value_head_dim=128,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1024,
            ffn_dim=4096,
            num_layers=24,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
        ),
    )


def _4b(attn_backend: str) -> Qwen35Model.Config:
    """Qwen3.5-4B dense config with vision encoder.

    NOTE: HF config has tie_word_embeddings=true. Torchtitan doesn't support
    tied embeddings yet, so we use a separate lm_head.
    """
    dim = 2560
    head_dim = 256
    rotary_dim = 64
    n_layers = 32
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=4,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=9216,
            n_key_heads=16,
            n_value_heads=32,
            key_head_dim=128,
            value_head_dim=128,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1024,
            ffn_dim=4096,
            num_layers=24,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2560,
            num_position_embeddings=2304,
        ),
    )


def _9b(attn_backend: str) -> Qwen35Model.Config:
    """Qwen3.5-9B dense config with vision encoder."""
    dim = 4096
    head_dim = 256
    rotary_dim = 64
    n_layers = 32
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=4,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=12288,
            n_key_heads=16,
            n_value_heads=32,
            key_head_dim=128,
            value_head_dim=128,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
        ),
    )


def _27b(attn_backend: str) -> Qwen35Model.Config:
    """Qwen3.5-27B dense config with vision encoder."""
    dim = 5120
    head_dim = 256
    rotary_dim = 64
    n_layers = 64
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=24,
            n_kv_heads=4,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=17408,
            n_key_heads=16,
            n_value_heads=48,
            key_head_dim=128,
            value_head_dim=128,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=5120,
            num_position_embeddings=2304,
        ),
    )


def _35b_a3b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
) -> Qwen35Model.Config:
    """Qwen3.5-35B-A3B MoE config with vision encoder."""
    dim = 2048
    head_dim = 256
    rotary_dim = 64
    n_layers = 40
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=512,
            num_experts=256,
            top_k=8,
            shared_expert_hidden_dim=512,
            n_key_heads=16,
            n_value_heads=32,
            key_head_dim=128,
            value_head_dim=128,
            moe_comm_backend=moe_comm_backend,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
        ),
    )


def _122b_a10b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
) -> Qwen35Model.Config:
    """Qwen3.5-122B-A10B MoE config with vision encoder."""
    dim = 3072
    head_dim = 256
    rotary_dim = 64
    n_layers = 48
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=1024,
            num_experts=256,
            top_k=8,
            shared_expert_hidden_dim=1024,
            n_key_heads=16,
            n_value_heads=64,
            key_head_dim=128,
            value_head_dim=128,
            moe_comm_backend=moe_comm_backend,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=3072,
            num_position_embeddings=2304,
        ),
    )


def _397b_a17b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
) -> Qwen35Model.Config:
    """Qwen3.5-397B-A17B MoE config with vision encoder."""
    dim = 4096
    head_dim = 256
    rotary_dim = 64
    n_layers = 60
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
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
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_seq_len=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=1024,
            num_experts=512,
            top_k=10,
            shared_expert_hidden_dim=1024,
            n_key_heads=16,
            n_value_heads=64,
            key_head_dim=128,
            value_head_dim=128,
            moe_comm_backend=moe_comm_backend,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
        ),
    )


qwen3_5_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_moe": _debugmodel_moe,
    "0.8B": _0_8b,
    "2B": _2b,
    "4B": _4b,
    "9B": _9b,
    "27B": _27b,
    "35B-A3B": _35b_a3b,
    "122B-A10B": _122b_a10b,
    "397B-A17B": _397b_a17b,
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
    config = qwen3_5_configs[flavor](**kwargs)
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)

    return ModelSpec(
        name="qwen3_5",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=pipeline_qwen3_5,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Qwen35StateDictAdapter,
    )
