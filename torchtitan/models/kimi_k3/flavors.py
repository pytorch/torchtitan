# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 model flavors."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import spmd_types as spmd
import torch
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common import (
    Conv1d,
    DenseFeedForward,
    Embedding,
    GatedRMSNorm,
    HiMidLoLinear,
    Linear,
    RowParallelLinear,
    Sigmoid,
    SiTUGLU,
)
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.kda import InnerKDA, KDA, KDAKernel
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_routed_experts_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_input_colwise_sharding_config,
    replicated_state_sharding_config,
    residual_activation_placement,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe import QuantileBalancedTopKRouter
from torchtitan.models.common.moe.sharding import (
    moe_sharding_config,
    router_gate_sharding_config,
    router_sharding_config,
)
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.vision_encoder import (
    InvariantRowParallelLinear,
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_sharding_config,
    multimodal_token_embedding_sharding_config,
    vision_attention_sharding_config,
    vision_colwise_sharding_config,
    vision_inner_attention_sharding_config,
    vision_invariant_linear_sharding_config,
    vision_rowwise_sharding_config,
)
from torchtitan.models.kimi_k2_7.sharding import (
    moonvit_encoder_sharding_config,
    moonvit_projector_sharding_config,
    moonvit_rotary_sharding_config,
)
from torchtitan.models.kimi_k2_7.vision_encoder import VisionRotaryEmbedding2D
from .model import (
    AttentionResidual,
    KimiK3Model,
    KimiK3TransformerBlock,
    KimiMLAAttention,
)
from .moe import KimiLatentMoE
from .sharding import (
    inner_kda_sharding_config,
    kda_conv_sharding_config,
    kda_sharding_config,
    latent_routed_down_sharding_config,
    latent_routed_up_sharding_config,
    mla_attention_sharding_config,
    multimodal_first_layer_sharding_config,
    tp_unsharded_weight_sharding_config,
)
from .vision_encoder import KimiK3VisionEncoder, KimiK3VisionProjector

__all__ = [
    "KIMI_K3_SPECIAL_TOKENS",
    "KimiK3Model",
    "KimiK3VisionEncoder",
    "MODEL_FLAVORS",
    "build_model_config",
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
# Zero init gives the uniform initial depth weights the report requires.
_RES_PROJ_INIT = {"weight": nn.init.zeros_}
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
    sharding_config: ShardingConfig,
) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        param_init=param_init or _LINEAR_INIT,
        sharding_config=sharding_config,
    )


def _vision_row_parallel_linear(
    in_features: int,
    out_features: int,
    *,
    param_init: dict[str, Callable] | None = None,
) -> InvariantRowParallelLinear.Config:
    return InvariantRowParallelLinear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        param_init=param_init or _LINEAR_INIT,
        sharding_config=vision_rowwise_sharding_config(),
    )


def _norm(
    dim: int, eps: float = 1e-5, *, sharding_config: ShardingConfig
) -> RMSNorm.Config:
    return RMSNorm.Config(
        normalized_shape=dim,
        eps=eps,
        param_init=_NORM_INIT,
        sharding_config=sharding_config,
    )


def _feed_forward_config(
    *,
    dim: int,
    hidden_dim: int,
    enable_sp: bool,
) -> DenseFeedForward.Config:
    return replace(
        make_ffn_config(
            dim=dim,
            hidden_dim=hidden_dim,
            w13_param_init=_LINEAR_INIT,
            w2_param_init=_LINEAR_INIT,
            enable_sp=enable_sp,
        ),
        activation_fn=SiTUGLU.Config(beta=4.0, linear_beta=25.0),
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
    enable_sp: bool,
) -> KimiMLAAttention.Config:
    """Low-rank projections and norms stay replicated; up-projections and the
    gate are column-parallel; ``wo`` is row-parallel to the residual layout."""
    inner_attention = replace(
        get_attention_config(attn_backend),
        sharding_config=inner_attention_local_spmd_sharding_config(),
    )

    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    return KimiMLAAttention.Config(
        dim=dim,
        n_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        wq_a=_linear(
            dim, q_lora_rank, sharding_config=replicated_state_sharding_config("weight")
        ),
        q_norm=_norm(
            q_lora_rank, sharding_config=replicated_state_sharding_config("weight")
        ),
        wq_b=_linear(
            q_lora_rank,
            num_heads * q_head_dim,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        wkv_a=_linear(
            dim,
            kv_lora_rank + qk_rope_head_dim,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        kv_norm=_norm(
            kv_lora_rank, sharding_config=replicated_state_sharding_config("weight")
        ),
        wkv_b=_linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        gate=_linear(
            dim,
            num_heads * v_head_dim,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        wo=RowParallelLinear.Config(
            in_features=num_heads * v_head_dim,
            out_features=dim,
            param_init=_LINEAR_INIT,
            sharding_config=rowwise_sharding_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        inner_attention=inner_attention,
        sharding_config=mla_attention_sharding_config(enable_sp=enable_sp),
    )


def _kda_config(
    *,
    dim: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    enable_sp: bool,
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
            sharding_config=kda_conv_sharding_config(),
        )

    def colwise_linear(in_features: int, out_features: int) -> Linear.Config:
        return _linear(
            in_features,
            out_features,
            sharding_config=replicated_input_colwise_sharding_config(),
        )

    return KDA.Config(
        n_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
        q_proj=colwise_linear(dim, projection_dim),
        k_proj=colwise_linear(dim, projection_dim),
        v_proj=colwise_linear(dim, projection_dim),
        q_conv=conv(),
        k_conv=conv(),
        v_conv=conv(),
        forget_a=_linear(
            dim, head_dim, sharding_config=replicated_state_sharding_config("weight")
        ),
        forget_b=colwise_linear(head_dim, projection_dim),
        beta=colwise_linear(dim, num_heads),
        output_gate=colwise_linear(dim, projection_dim),
        inner_attention=InnerKDA.Config(
            head_dim=head_dim,
            conv_kernel_size=conv_kernel_size,
            kernel=KDAKernel.Config(),
            sharding_config=inner_kda_sharding_config(),
        ),
        output_norm=GatedRMSNorm.Config(
            dim=head_dim,
            eps=1e-5,
            activation_fn=Sigmoid.Config(),
            param_init=_NORM_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        output_proj=RowParallelLinear.Config(
            in_features=projection_dim,
            out_features=dim,
            param_init=_LINEAR_INIT,
            sharding_config=rowwise_sharding_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        param_init={
            "A_log": _a_log_init,
            "dt_bias": nn.init.zeros_,
        },
        sharding_config=kda_sharding_config(enable_sp=enable_sp),
    )


def _latent_moe_config(
    *,
    dim: int,
    latent_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    num_shared_experts: int,
    enable_sp: bool,
    enable_ep: bool,
) -> KimiLatentMoE.Config:
    """Core's MoE sharding, plus the latent projections around the routed
    experts, which run on token-sharded activations."""
    return KimiLatentMoE.Config(
        num_experts=num_experts,
        router=QuantileBalancedTopKRouter.Config(
            num_experts=num_experts,
            top_k=top_k,
            gate=HiMidLoLinear.Config(
                in_features=dim,
                out_features=num_experts,
                backward_mode="hi_mid_lo",
                bias=False,
                param_init=_LINEAR_INIT,
                sharding_config=router_gate_sharding_config(),
            ),
            score_func=Sigmoid.Config(),
            route_norm=True,
            route_scale=1.0,
            num_bins=1000,
            sharding_config=router_sharding_config(),
        ),
        routed_down=_linear(
            dim, latent_dim, sharding_config=latent_routed_down_sharding_config()
        ),
        routed_experts=replace(
            make_routed_experts_config(
                dim=latent_dim,
                hidden_dim=expert_hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                param_init={
                    "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
                    "w2_EDF": partial(nn.init.trunc_normal_, std=0.02),
                    "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
                },
                enable_ep=enable_ep,
            ),
            activation_fn=SiTUGLU.Config(beta=4.0, linear_beta=25.0),
        ),
        routed_norm=_norm(
            latent_dim, sharding_config=norm_sharding_config(enable_sp=True)
        ),
        routed_up=_linear(
            latent_dim, dim, sharding_config=latent_routed_up_sharding_config()
        ),
        shared_experts=replace(
            make_shared_expert_ffn_config(
                dim=dim,
                hidden_dim=num_shared_experts * expert_hidden_dim,
                w13_param_init=_LINEAR_INIT,
                w2_param_init=_LINEAR_INIT,
                enable_sp=enable_sp,
            ),
            activation_fn=SiTUGLU.Config(beta=4.0, linear_beta=25.0),
        ),
        load_balance_coeff=None,
        sharding_config=moe_sharding_config(enable_sp=enable_sp),
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
        sharding_config=invariant_norm_sharding_config(),
    )
    block = VisionTransformerBlock.Config(
        norm1=vision_norm,
        norm2=vision_norm,
        attn=VisionAttention.Config(
            dim=qkv_dim,
            num_heads=num_heads,
            wq=_linear(
                dim,
                qkv_dim,
                sharding_config=vision_colwise_sharding_config(input_tp=spmd.R),
            ),
            wk=_linear(
                dim,
                qkv_dim,
                sharding_config=vision_colwise_sharding_config(input_tp=spmd.R),
            ),
            wv=_linear(
                dim,
                qkv_dim,
                sharding_config=vision_colwise_sharding_config(input_tp=spmd.R),
            ),
            proj=_vision_row_parallel_linear(qkv_dim, dim),
            inner_attention=FlexInnerAttention.Config(
                sharding_config=vision_inner_attention_sharding_config()
            ),
            sharding_config=vision_attention_sharding_config(rope_cache_dp=spmd.V),
        ),
        mlp=VisionMLP.Config(
            fc1=_linear(
                dim,
                hidden_dim,
                param_init=_fan_in_linear_init(dim),
                sharding_config=vision_colwise_sharding_config(),
            ),
            fc2=_vision_row_parallel_linear(
                hidden_dim, dim, param_init=_fan_in_linear_init(hidden_dim)
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
        patch_embed_proj=_linear(
            patch_dim, dim, sharding_config=vision_invariant_linear_sharding_config()
        ),
        rotary_pos_emb=VisionRotaryEmbedding2D.Config(
            head_dim=head_dim, sharding_config=moonvit_rotary_sharding_config()
        ),
        block=block,
        final_norm=vision_norm,
        projector=KimiK3VisionProjector.Config(
            linear_1=_linear(
                merged_dim,
                merged_dim,
                param_init=_fan_in_linear_init(merged_dim),
                sharding_config=vision_colwise_sharding_config(),
            ),
            linear_2=_vision_row_parallel_linear(
                merged_dim,
                text_dim,
                param_init=_fan_in_linear_init(merged_dim),
            ),
            post_norm=RMSNorm.Config(
                normalized_shape=text_dim,
                eps=1e-5,
                param_init=_NORM_INIT,
                sharding_config=invariant_norm_sharding_config(),
            ),
            activation=GELU.Config(),
            sharding_config=moonvit_projector_sharding_config(),
        ),
        param_init=_POS_EMBED_INIT,
        sharding_config=moonvit_encoder_sharding_config(),
    )


def _kimi_k3_config(
    *,
    max_context_length: int,
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
    parallelism: ParallelismConfig,
) -> KimiK3Model.Config:
    """Assemble a Kimi K3 config from the released topology's free parameters.

    ``full_attention_layers`` holds zero-based layer indices. Every other layer
    is KDA. Layer 0 is the single dense FFN layer (released
    ``first_k_dense_replace=1``); the rest are LatentMoE.

    With a vision encoder, ``tok_embeddings`` keeps a TP-replicated output for
    the vision scatter and layer 0's input boundary restores the decoder's
    layout.
    """
    enable_sp = parallelism.enable_sequence_parallel
    enable_ep = parallelism.expert_parallel_degree > 1
    multimodal = vision_encoder is not None
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)

    def residual_norm() -> RMSNorm.Config:
        return _norm(
            dim,
            sharding_config=tp_unsharded_weight_sharding_config(
                token_sharded=enable_sp
            ),
        )

    def residual_proj() -> Linear.Config:
        return _linear(
            dim,
            1,
            param_init=_RES_PROJ_INIT,
            sharding_config=tp_unsharded_weight_sharding_config(
                token_sharded=enable_sp
            ),
        )

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
                        enable_sp=enable_sp,
                    )
                    if is_full_attention
                    else _kda_config(
                        dim=dim,
                        num_heads=num_heads,
                        head_dim=kda_head_dim,
                        conv_kernel_size=conv_kernel_size,
                        enable_sp=enable_sp,
                    )
                ),
                feed_forward=(
                    _feed_forward_config(
                        dim=dim, hidden_dim=dense_hidden_dim, enable_sp=enable_sp
                    )
                    if layer_idx == 0
                    else _latent_moe_config(
                        dim=dim,
                        latent_dim=latent_dim,
                        expert_hidden_dim=expert_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        num_shared_experts=num_shared_experts,
                        enable_sp=enable_sp,
                        enable_ep=enable_ep,
                    )
                ),
                attention_norm=_norm(dim, sharding_config=norm_sharding),
                ffn_norm=_norm(dim, sharding_config=norm_sharding),
                # Layer 0 has no attention residual.
                attention_res_norm=None if layer_idx == 0 else residual_norm(),
                attention_res_proj=None if layer_idx == 0 else residual_proj(),
                ffn_res_norm=residual_norm(),
                ffn_res_proj=residual_proj(),
                attention_residual=AttentionResidual.Config(),
                sharding_config=(
                    multimodal_first_layer_sharding_config(enable_sp=enable_sp)
                    if multimodal and layer_idx == 0
                    else None
                ),
            )
        )

    return KimiK3Model.Config(
        max_context_length=max_context_length,
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=(
                multimodal_token_embedding_sharding_config()
                if multimodal
                else token_embedding_sharding_config(enable_sp=enable_sp)
            ),
        ),
        layers=layers,
        norm=_norm(
            dim, sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp)
        ),
        lm_head=_linear(
            dim,
            vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        output_res_norm=residual_norm(),
        output_res_proj=residual_proj(),
        vision_encoder=vision_encoder,
    )


def _debugmodel(
    attn_backend: str,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> KimiK3Model.Config:
    dim = 256
    return _kimi_k3_config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=2048,
        num_layers=17,
        full_attention_layers={3, 7, 11, 15, 16},
        attn_res_block_size=4,
        num_heads=4,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=512,
        latent_dim=128,
        expert_hidden_dim=128,
        num_experts=8,
        top_k=2,
        num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=256,
            qkv_dim=512,
            hidden_dim=512,
            num_layers=2,
            num_heads=4,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend=attn_backend,
        parallelism=parallelism,
    )


def _kimi_k3(
    attn_backend: str,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> KimiK3Model.Config:
    dim = 7168
    return _kimi_k3_config(
        max_context_length=seq_len,
        dim=dim,
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
        parallelism=parallelism,
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 16384),
    "Kimi-K3": (_kimi_k3, 262144),
}


def build_model_config(
    flavor: str,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
) -> KimiK3Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        seq_len=context_len,
        parallelism=parallelism,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            config = converter.build().convert(config)
    return config
