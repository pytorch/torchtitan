# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import Conv1d, Embedding, Linear
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_token_dispatcher_config,
)
from torchtitan.models.common.moe import RoutedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.kimi_k2_7.vision_encoder import VisionRotaryEmbedding2D
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .kda import InnerKDA, KDA, KDAKernel, KimiRMSNormGated
from .model import KimiK3Model, KimiK3TransformerBlock, KimiMLAAttention
from .moe import KimiFeedForward, KimiGroupedExperts, KimiLatentMoE
from .parallelize import parallelize_kimi_k3
from .state_dict_adapter import KimiK3StateDictAdapter
from .vision_encoder import KimiK3VisionEncoder, KimiK3VisionProjector

__all__ = [
    "KIMI_K3_SPECIAL_TOKENS",
    "KimiK3Model",
    "KimiK3StateDictAdapter",
    "KimiK3VisionEncoder",
    "kimi_k3_configs",
    "model_registry",
    "parallelize_kimi_k3",
]


KIMI_K3_SPECIAL_TOKENS = {
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
_CONV_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    scale = dim**-0.5
    return {
        "weight": partial(
            nn.init.trunc_normal_,
            std=scale,
            a=-3 * scale,
            b=3 * scale,
        )
    }


def _fan_in_linear_init(in_features: int) -> dict[str, Callable]:
    return {
        "weight": partial(
            nn.init.trunc_normal_,
            std=(2.0 / in_features) ** 0.5,
        ),
        "bias": nn.init.zeros_,
    }


def _a_log_init(parameter: nn.Parameter) -> None:
    with torch.no_grad():
        nn.init.uniform_(parameter, 1.0, 16.0)
        parameter.log_()


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
    param_init: dict[str, Callable] | None = None,
) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        param_init=param_init or _LINEAR_INIT,
    )


def _norm(dim: int, eps: float = 1e-5) -> RMSNorm.Config:
    return RMSNorm.Config(
        normalized_shape=dim,
        eps=eps,
        param_init=_NORM_INIT,
    )


def _feed_forward_config(
    *,
    dim: int,
    hidden_dim: int,
) -> KimiFeedForward.Config:
    return KimiFeedForward.Config(
        w1=_linear(dim, hidden_dim),
        w2=_linear(hidden_dim, dim),
        w3=_linear(dim, hidden_dim),
        beta=4.0,
        linear_beta=25.0,
    )


def _mla_config(
    *,
    dim: int,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    attn_backend: str,
) -> KimiMLAAttention.Config:
    inner_attention = get_attention_config(attn_backend)

    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    return KimiMLAAttention.Config(
        dim=dim,
        n_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        wq_a=_linear(dim, q_lora_rank),
        q_norm=_norm(q_lora_rank),
        wq_b=_linear(q_lora_rank, num_heads * q_head_dim),
        wkv_a=_linear(dim, kv_lora_rank + qk_rope_head_dim),
        kv_norm=_norm(kv_lora_rank),
        wkv_b=_linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
        ),
        gate=_linear(dim, num_heads * v_head_dim),
        wo=_linear(num_heads * v_head_dim, dim),
        inner_attention=inner_attention,
    )


def _kda_config(
    *,
    dim: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
) -> KDA.Config:
    projection_dim = num_heads * head_dim

    def conv() -> Conv1d.Config:
        return Conv1d.Config(
            in_channels=projection_dim,
            out_channels=projection_dim,
            kernel_size=conv_kernel_size,
            groups=projection_dim,
            bias=False,
            param_init=_CONV_INIT,
        )

    return KDA.Config(
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
        q_proj=_linear(dim, projection_dim),
        k_proj=_linear(dim, projection_dim),
        v_proj=_linear(dim, projection_dim),
        q_conv=conv(),
        k_conv=conv(),
        v_conv=conv(),
        forget_a=_linear(dim, head_dim),
        forget_b=_linear(head_dim, projection_dim),
        beta=_linear(dim, num_heads),
        output_gate=_linear(dim, projection_dim),
        inner_kda=InnerKDA.Config(
            head_dim=head_dim,
            kernel=KDAKernel.Config(),
        ),
        output_norm=KimiRMSNormGated.Config(
            dim=head_dim,
            eps=1e-5,
            param_init=_NORM_INIT,
        ),
        output_proj=_linear(projection_dim, dim),
        param_init={
            "A_log": _a_log_init,
            "dt_bias": nn.init.zeros_,
        },
    )


def _latent_moe_config(
    *,
    dim: int,
    latent_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    num_shared_experts: int,
    moe_comm_backend: str,
) -> KimiLatentMoE.Config:
    return KimiLatentMoE.Config(
        num_experts=num_experts,
        router=TokenChoiceTopKRouter.Config(
            num_experts=num_experts,
            top_k=top_k,
            gate=_linear(dim, num_experts),
            score_func="sigmoid",
            route_norm=True,
            route_scale=1.0,
        ),
        routed_down=_linear(dim, latent_dim),
        routed_experts=RoutedExperts.Config(
            inner_experts=KimiGroupedExperts.Config(
                dim=latent_dim,
                hidden_dim=expert_hidden_dim,
                num_experts=num_experts,
                beta=4.0,
                linear_beta=25.0,
                param_init={
                    "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
                    "w2_EDF": partial(nn.init.trunc_normal_, std=0.02),
                    "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
                },
            ),
            # core's dispatcher factory: standard / deepep / hybridep /
            # minimal_async_ep per spec, as deepseek_v3; falls back to local
            # dispatch when the ep mesh is None.
            token_dispatcher=make_token_dispatcher_config(
                num_experts=num_experts,
                top_k=top_k,
                comm_backend=moe_comm_backend,
                # The routed experts consume the LATENT stream, so the
                # dispatcher buffers size by latent_dim, not model dim.
                hidden_dim=latent_dim,
            ),
        ),
        routed_norm=_norm(latent_dim),
        routed_up=_linear(latent_dim, dim),
        shared_experts=_feed_forward_config(
            dim=dim,
            hidden_dim=num_shared_experts * expert_hidden_dim,
        ),
        load_balance_coeff=1e-3,
    )


def _vision_encoder_config(
    *,
    text_dim: int,
    dim: int,
    qkv_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    patch_size: int = 14,
    in_channels: int = 3,
    merge_kernel_size: tuple[int, int] = (2, 2),
    init_pos_emb_height: int = 16,
    init_pos_emb_width: int = 16,
    max_num_frames: int = 4,
) -> KimiK3VisionEncoder.Config:
    patch_dim = in_channels * patch_size * patch_size
    head_dim = qkv_dim // num_heads
    merged_dim = dim * merge_kernel_size[0] * merge_kernel_size[1]
    vision_norm = RMSNorm.Config(
        normalized_shape=dim,
        eps=1e-5,
        param_init=_NORM_INIT,
    )
    block = VisionTransformerBlock.Config(
        norm1=vision_norm,
        norm2=vision_norm,
        attn=VisionAttention.Config(
            dim=qkv_dim,
            num_heads=num_heads,
            wq=_linear(dim, qkv_dim),
            wk=_linear(dim, qkv_dim),
            wv=_linear(dim, qkv_dim),
            proj=_linear(qkv_dim, dim),
        ),
        mlp=VisionMLP.Config(
            fc1=_linear(
                dim,
                hidden_dim,
                param_init=_fan_in_linear_init(dim),
            ),
            fc2=_linear(
                hidden_dim,
                dim,
                param_init=_fan_in_linear_init(hidden_dim),
            ),
            act_fn=GELU.Config(approximate="tanh"),
        ),
    )
    return KimiK3VisionEncoder.Config(
        dim=dim,
        num_layers=num_layers,
        patch_size=patch_size,
        in_channels=in_channels,
        merge_kernel_size=merge_kernel_size,
        init_pos_emb_height=init_pos_emb_height,
        init_pos_emb_width=init_pos_emb_width,
        max_num_frames=max_num_frames,
        interpolation_mode="bilinear",
        patch_embed_proj=_linear(patch_dim, dim),
        rotary_pos_emb=VisionRotaryEmbedding2D.Config(head_dim=head_dim),
        block=block,
        final_norm=vision_norm,
        projector=KimiK3VisionProjector.Config(
            linear_1=_linear(
                merged_dim,
                merged_dim,
                param_init=_fan_in_linear_init(merged_dim),
            ),
            linear_2=_linear(
                merged_dim,
                text_dim,
                param_init=_fan_in_linear_init(merged_dim),
            ),
            post_norm=RMSNorm.Config(
                normalized_shape=text_dim,
                eps=1e-5,
                param_init=_NORM_INIT,
            ),
            activation=GELU.Config(),
        ),
        param_init=_POS_EMBED_INIT,
    )


def _kimi_k3_config(
    *,
    dim: int,
    vocab_size: int,
    num_layers: int,
    full_attention_layers: set[int],
    attn_res_block_size: int,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kda_head_dim: int,
    conv_kernel_size: int,
    dense_hidden_dim: int,
    latent_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    num_shared_experts: int,
    vision_encoder: KimiK3VisionEncoder.Config,
    attn_backend: str,
    moe_comm_backend: str = "standard",
) -> KimiK3Model.Config:
    """Assemble a Kimi K3 config from the released topology's free parameters.

    ``full_attention_layers`` holds zero-based layer indices. Every other layer
    is KDA. Layer 0 is the single dense FFN layer (released
    ``first_k_dense_replace=1``); the rest are LatentMoE.
    """
    layers = []
    for layer_idx in range(num_layers):
        is_full_attention = layer_idx in full_attention_layers
        layers.append(
            KimiK3TransformerBlock.Config(
                layer_id=layer_idx,
                attn_res_block_size=attn_res_block_size,
                attention=(
                    _mla_config(
                        dim=dim,
                        num_heads=num_heads,
                        q_lora_rank=q_lora_rank,
                        kv_lora_rank=kv_lora_rank,
                        qk_nope_head_dim=qk_nope_head_dim,
                        qk_rope_head_dim=qk_rope_head_dim,
                        v_head_dim=v_head_dim,
                        attn_backend=attn_backend,
                    )
                    if is_full_attention
                    else None
                ),
                delta_attention=(
                    None
                    if is_full_attention
                    else _kda_config(
                        dim=dim,
                        num_heads=num_heads,
                        head_dim=kda_head_dim,
                        conv_kernel_size=conv_kernel_size,
                    )
                ),
                feed_forward=(
                    _feed_forward_config(dim=dim, hidden_dim=dense_hidden_dim)
                    if layer_idx == 0
                    else None
                ),
                moe=(
                    None
                    if layer_idx == 0
                    else _latent_moe_config(
                        dim=dim,
                        latent_dim=latent_dim,
                        expert_hidden_dim=expert_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        num_shared_experts=num_shared_experts,
                        moe_comm_backend=moe_comm_backend,
                    )
                ),
                attention_norm=_norm(dim),
                ffn_norm=_norm(dim),
                attention_res_norm=None if layer_idx == 0 else _norm(dim),
                attention_res_proj=None if layer_idx == 0 else _linear(dim, 1),
                ffn_res_norm=_norm(dim),
                ffn_res_proj=_linear(dim, 1),
            )
        )

    return KimiK3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        layers=layers,
        norm=_norm(dim),
        lm_head=_linear(
            dim,
            vocab_size,
            param_init=_output_linear_init(dim),
        ),
        output_res_norm=_norm(dim),
        output_res_proj=_linear(dim, 1),
        vision_encoder=vision_encoder,
    )


def _debugmodel(attn_backend: str, moe_comm_backend: str) -> KimiK3Model.Config:
    dim = 1024
    return _kimi_k3_config(
        dim=dim,
        moe_comm_backend=moe_comm_backend,
        vocab_size=163840,
        num_layers=24,
        full_attention_layers={3, 7, 11, 15, 19, 23},
        attn_res_block_size=12,
        num_heads=16,
        q_lora_rank=512,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=4096,
        latent_dim=512,
        expert_hidden_dim=384,
        num_experts=32,
        top_k=4,
        num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=512,
            qkv_dim=768,
            hidden_dim=2048,
            num_layers=8,
            num_heads=6,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend=attn_backend,
    )


def _kimi_k3(attn_backend: str, moe_comm_backend: str) -> KimiK3Model.Config:
    dim = 7168
    return _kimi_k3_config(
        dim=dim,
        moe_comm_backend=moe_comm_backend,
        vocab_size=163840,
        num_layers=93,
        full_attention_layers=set(range(3, 92, 4)) | {92},
        attn_res_block_size=12,
        num_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=33792,
        latent_dim=3584,
        expert_hidden_dim=3072,
        num_experts=896,
        top_k=16,
        num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=1024,
            qkv_dim=1536,
            hidden_dim=4096,
            num_layers=27,
            num_heads=12,
            init_pos_emb_height=64,
            init_pos_emb_width=64,
        ),
        attn_backend=attn_backend,
    )


kimi_k3_configs = {
    "debugmodel": (_debugmodel, 16384),
    "Kimi-K3": (_kimi_k3, 262144),
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
    moe_comm_backend: str = "standard",
    *,
    seq_len: int | None = None,
) -> ModelSpec:
    # The KDA / MLA layers build their own RoPE, so seq_len is not a builder
    # argument here -- it only reports the context length on the ModelSpec.
    get_config, max_context_len = kimi_k3_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(attn_backend=attn_backend, moe_comm_backend=moe_comm_backend)
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            config = converter.build().convert(config)
    return ModelSpec(
        name="kimi_k3",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=None,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=KimiK3StateDictAdapter,
    )
