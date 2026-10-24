# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from collections.abc import Callable
from functools import partial
from typing import cast, Literal

import spmd_types as spmd
import torch
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    QKVLinear,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import (
    fused_grouped_gate_up_param_init,
    fused_qkv_param_init,
    get_attention_config,
)
from torchtitan.models.common.decoder_sharding import (
    colwise_sharding_config,
    dense_sequence_parallel_placement,
    gqa_attention_sharding_config,
    head_norm_sharding_config,
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import RoutedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.moe.sharding import (
    moe_sharding_config,
    routed_expert_linear_sharding_config,
    routed_experts_sharding_config,
    router_gate_sharding_config,
    router_sharding_config,
)
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.common.nn_modules import Conv1d, GELU, LayerNorm, RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import (
    InvariantRowParallelLinear,
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_sharding_config,
    vision_attention_sharding_config,
    vision_colwise_sharding_config,
    vision_inner_attention_sharding_config,
    vision_invariant_linear_sharding_config,
    vision_rowwise_sharding_config,
)

from .audio_encoder import (
    _AudioBlock,
    _AudioPatchifier,
    _AudioStage,
    _DepthwiseCausalConv,
    MuseSparkAudioEncoder,
)
from .model import (
    MuseSparkAttention,
    MuseSparkEmbeddingWithNorm,
    MuseSparkLMHead,
    MuseSparkModel,
    MuseSparkNormPreservingResidualConnection,
    MuseSparkRMSNorm,
    MuseSparkRMSNormWithInputScale,
    MuseSparkTransformerBlock,
)
from .moe import MuseSparkMoE, MuseSparkRouterSigmoid

from .sharding import (
    audio_encoder_sharding_config,
    audio_invariant_state_sharding_config,
    audio_wrapper_sharding_config,
    compressed_moe_linear_sharding_config,
    muse_spark_activation_placement,
    perception_norm_sharding_config,
    post_expert_norm_sharding_config,
    residual_gate_sharding_config,
    transformer_block_sharding_config,
)
from .vision_encoder import MuseSparkNaViTAdapter, NaViTEncoder, VisionRopeFreq
from .vision_encoder_sharding import (
    cp_replicated_vision_sharding_config,
    navit_encoder_sharding_config,
    vision_rope_freq_sharding_config,
)


# The query is scaled by this tuned constant divided by sqrt(head_dim) after
# q-norm.
_SCALE_QUERY_NUMERATOR = 43.7840518911

_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
}
_EMBEDDING_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=1.0),
}
_GAIN_NORM_INIT: dict[str, Callable] = {"weight": nn.init.zeros_}
_VISION_POSITION_INIT: dict[str, Callable] = {
    "positional_embedding_vlm": partial(nn.init.trunc_normal_, std=0.02),
}

_AUDIO_NORM_EPS = 1e-6
_NORM_EPS = 1e-5
_POST_NORM_EPS = 1e-8
_ROPE_THETA = 500_000.0
_RESIDUAL_GATE_TEMPERATURE = 0.3
_RESIDUAL_GATE_EPS = 1e-3
_DEBUG_MODEL_DIM = 256
_MUSE_SPARK_816B_DIM = 8192
_MUSE_SPARK_816B_VOCAB_SIZE = 202_048
_MUSE_SPARK_816B_VISION_ADAPTER_DIM = 4096


def _audio_block_config(*, hidden_size: int, kernel_size: int) -> _AudioBlock.Config:
    return _AudioBlock.Config(
        conv=_DepthwiseCausalConv.Config(
            kernel_size=kernel_size,
            conv=Conv1d.Config(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                groups=hidden_size,
                bias=True,
                sharding_config=audio_invariant_state_sharding_config(has_bias=True),
            ),
            sharding_config=audio_wrapper_sharding_config("value_LD"),
        ),
        norm=RMSNorm.Config(
            normalized_shape=hidden_size,
            eps=_AUDIO_NORM_EPS,
            sharding_config=audio_invariant_state_sharding_config(),
        ),
        linear1=Linear.Config(
            in_features=hidden_size,
            out_features=hidden_size * 4,
            bias=True,
            sharding_config=audio_invariant_state_sharding_config(has_bias=True),
        ),
        linear2=Linear.Config(
            in_features=hidden_size * 4,
            out_features=hidden_size,
            bias=True,
            sharding_config=audio_invariant_state_sharding_config(has_bias=True),
        ),
        sharding_config=audio_wrapper_sharding_config("value_LD"),
    )


def muse_spark_audio_encoder_config(
    output_size: int,
    *,
    patch_size: int = 1920,
    kernel_size: int = 7,
    input_channels: int = 1,
    patchify_sizes: tuple[int, ...] = (16, 6, 5, 4),
    hidden_sizes: tuple[int, ...] = (256, 512, 1024, 2048),
) -> MuseSparkAudioEncoder.Config:
    if len(patchify_sizes) != len(hidden_sizes):
        raise ValueError("audio stage settings must have equal lengths")
    if patch_size != math.prod(patchify_sizes):
        raise ValueError("patch_size must equal the product of patchify_sizes")

    layers = []
    stage_input = input_channels
    for stage_patch_size, hidden_size in zip(
        patchify_sizes,
        hidden_sizes,
        strict=True,
    ):
        layers.append(
            _AudioStage.Config(
                patchifier=_AudioPatchifier.Config(
                    patch_size=stage_patch_size,
                    linear=Linear.Config(
                        in_features=stage_input * stage_patch_size,
                        out_features=hidden_size,
                        bias=True,
                        sharding_config=audio_invariant_state_sharding_config(
                            has_bias=True
                        ),
                    ),
                    norm=RMSNorm.Config(
                        normalized_shape=hidden_size,
                        eps=_AUDIO_NORM_EPS,
                        sharding_config=audio_invariant_state_sharding_config(),
                    ),
                    sharding_config=audio_wrapper_sharding_config("value_LC"),
                ),
                block1=_audio_block_config(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                ),
                block2=_audio_block_config(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                ),
                sharding_config=audio_wrapper_sharding_config("value_LC"),
            )
        )
        stage_input = hidden_size

    return MuseSparkAudioEncoder.Config(
        output_size=output_size,
        patch_size=patch_size,
        input_channels=input_channels,
        layers=layers,
        norm=RMSNorm.Config(
            normalized_shape=stage_input,
            eps=_AUDIO_NORM_EPS,
            sharding_config=audio_invariant_state_sharding_config(),
        ),
        o_proj=Linear.Config(
            in_features=stage_input,
            out_features=output_size,
            bias=True,
            sharding_config=audio_invariant_state_sharding_config(has_bias=True),
        ),
        dual_norm=RMSNorm.Config(
            normalized_shape=output_size,
            eps=_AUDIO_NORM_EPS,
            sharding_config=audio_invariant_state_sharding_config(),
        ),
        sharding_config=audio_encoder_sharding_config(),
    )


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(
            nn.init.trunc_normal_,
            std=depth_scaled_std(0.02, layer_id),
        ),
    }


def _output_linear_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(
            nn.init.trunc_normal_,
            std=std,
            a=-3 * std,
            b=3 * std,
        ),
    }


def _grouped_linear_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(
            nn.init.trunc_normal_,
            std=depth_scaled_std(0.02, layer_id),
        ),
        "w3_EFD": partial(
            nn.init.trunc_normal_,
            std=depth_scaled_std(0.02, layer_id),
        ),
    }


def _gain_norm(
    dim: int,
    *,
    elementwise_affine: bool = True,
    gain_center: float = 1.0,
    eps: float = _NORM_EPS,
    sharding_config: ShardingConfig | None = None,
) -> MuseSparkRMSNorm.Config:
    return MuseSparkRMSNorm.Config(
        normalized_shape=dim,
        eps=eps,
        elementwise_affine=elementwise_affine,
        gain_center=gain_center,
        param_init=_GAIN_NORM_INIT if elementwise_affine else None,
        sharding_config=sharding_config,
    )


def _input_scaled_gain_norm(
    dim: int,
    *,
    gain_center: float,
    eps: float,
    sharding_config: ShardingConfig | None = None,
) -> MuseSparkRMSNormWithInputScale.Config:
    return MuseSparkRMSNormWithInputScale.Config(
        normalized_shape=dim,
        eps=eps,
        elementwise_affine=True,
        gain_center=gain_center,
        param_init=_GAIN_NORM_INIT,
        sharding_config=sharding_config,
    )


def _residual_gate(
    dim: int,
    *,
    layer_id: int,
    enable_sp: bool,
) -> MuseSparkNormPreservingResidualConnection.Config:
    target_beta = 1.0 / (2.0 * (layer_id + 1))
    target_beta = max(
        _RESIDUAL_GATE_EPS,
        min(1.0 - _RESIDUAL_GATE_EPS, target_beta),
    )
    gate_init = _RESIDUAL_GATE_TEMPERATURE * math.log(target_beta / (1.0 - target_beta))
    return MuseSparkNormPreservingResidualConnection.Config(
        dim=dim,
        temperature=_RESIDUAL_GATE_TEMPERATURE,
        eps=_RESIDUAL_GATE_EPS,
        param_init={"gate": partial(nn.init.constant_, val=gate_init)},
        sharding_config=residual_gate_sharding_config(enable_sp=enable_sp),
    )


def _attention_config(
    *,
    layer_id: int,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_context_length: int,
    use_rope: bool,
    window_size: int | None,
    query_scale: float,
    position_temperature_floor_scale: float | None,
    position_temperature_query_scale: float | None,
    attn_backend: str,
    enable_sp: bool,
) -> MuseSparkAttention.Config:
    activation = muse_spark_activation_placement(enable_sp=enable_sp)
    inner_attention = get_attention_config(attn_backend)
    inner_attention.sharding_config = inner_attention_local_spmd_sharding_config()
    if window_size is not None:
        if isinstance(inner_attention, FlexInnerAttention.Config):
            inner_attention = SlidingWindowFlexInnerAttention.Config(
                block_size=inner_attention.block_size,
                kernel_options=inner_attention.kernel_options,
                sliding_window_size=window_size,
                sharding_config=inner_attention_local_spmd_sharding_config(),
            )
        elif isinstance(inner_attention, VarlenInnerAttention.Config):
            inner_attention = SlidingWindowVarlenInnerAttention.Config(
                **{
                    f.name: getattr(inner_attention, f.name)
                    for f in dataclasses.fields(inner_attention)
                },
                sliding_window_size=window_size,
            )
    return MuseSparkAttention.Config(
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=QKVLinear.Config(
            head_dim=head_dim,
            n_heads=num_heads,
            n_kv_heads=num_kv_heads,
            wqkv=ColumnParallelLinear.Config(
                in_features=dim,
                out_features=(num_heads + 2 * num_kv_heads) * head_dim,
                bias=False,
                param_init=fused_qkv_param_init(
                    _LINEAR_INIT,
                    n_heads=num_heads,
                    n_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                ),
                sharding_config=colwise_sharding_config(input_layout=activation),
            ),
        ),
        wo=RowParallelLinear.Config(
            in_features=num_heads * head_dim,
            out_features=dim,
            bias=False,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_sharding_config(output_layout=activation),
        ),
        qk_norm=RMSNorm.Config(
            normalized_shape=head_dim,
            eps=_NORM_EPS,
            elementwise_affine=False,
            sharding_config=head_norm_sharding_config(),
        ),
        inner_attention=inner_attention,
        rope=(
            ComplexRoPE.Config(
                dim=head_dim,
                max_context_length=max_context_length,
                theta=_ROPE_THETA,
                sharding_config=rope_cache_sharding_config(),
            )
            if use_rope
            else None
        ),
        use_rope=use_rope,
        scale_query_by=query_scale,
        window_size=window_size,
        position_temperature_floor_scale=position_temperature_floor_scale,
        position_temperature_query_scale=position_temperature_query_scale,
        output_norm=RMSNorm.Config(
            normalized_shape=head_dim,
            eps=_NORM_EPS,
            elementwise_affine=False,
            sharding_config=head_norm_sharding_config(),
        ),
        output_gate=ColumnParallelLinear.Config(
            in_features=dim,
            out_features=num_heads * head_dim,
            bias=False,
            param_init=_LINEAR_INIT,
            sharding_config=colwise_sharding_config(input_layout=activation),
        ),
        sharding_config=gqa_attention_sharding_config(enable_sp=enable_sp),
    )


def _moe_config(
    *,
    layer_id: int,
    dim: int,
    compressed_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    post_norm_gain_center: float,
    route_scale: float,
    router_logit_multiplier: float,
    route_norm_epsilon: float,
    load_balance_coeff: float | None,
    expert_bias_init: float,
    enable_sp: bool,
    enable_ep: bool,
) -> MuseSparkMoE.Config:
    grouped_linear_init = _grouped_linear_init(layer_id)
    experts_sharding = routed_experts_sharding_config()
    experts_sharding.out_dst_shardings = dense_sequence_parallel_placement()
    return MuseSparkMoE.Config(
        num_experts=num_experts,
        router=TokenChoiceTopKRouter.Config(
            num_experts=num_experts,
            gate=HiMidLoLinear.Config(
                in_features=dim,
                out_features=num_experts,
                backward_mode="hi_mid_lo",
                bias=False,
                param_init=_depth_init(layer_id),
                sharding_config=router_gate_sharding_config(),
            ),
            top_k=top_k,
            score_func=MuseSparkRouterSigmoid.Config(
                logit_multiplier=router_logit_multiplier,
            ),
            route_norm=True,
            route_scale=route_scale,
            route_norm_epsilon=route_norm_epsilon,
            sharding_config=router_sharding_config(),
        ),
        routed_experts=RoutedExperts.Config(
            w13=GroupedLinear.Config(
                group_size=num_experts,
                in_features=compressed_dim,
                out_features=expert_hidden_dim,
                num_linears=2,
                param_init=fused_grouped_gate_up_param_init(grouped_linear_init),
                sharding_config=routed_expert_linear_sharding_config(
                    enable_ep=enable_ep
                ),
            ),
            w2=GroupedLinear.Config(
                group_size=num_experts,
                in_features=expert_hidden_dim,
                out_features=compressed_dim,
                param_init={"weight": grouped_linear_init["w2_EDF"]},
                sharding_config=routed_expert_linear_sharding_config(
                    enable_ep=enable_ep
                ),
            ),
            token_dispatcher=AllToAllTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
            ),
            output_postprocess=_input_scaled_gain_norm(
                compressed_dim,
                gain_center=post_norm_gain_center,
                eps=_POST_NORM_EPS,
                sharding_config=post_expert_norm_sharding_config(enable_ep=enable_ep),
            ),
            sharding_config=experts_sharding,
        ),
        load_balance_coeff=load_balance_coeff,
        expert_bias_init=expert_bias_init,
        pooler=Linear.Config(
            in_features=dim,
            out_features=compressed_dim,
            bias=False,
            param_init=_LINEAR_INIT,
            sharding_config=compressed_moe_linear_sharding_config(has_bias=True),
        ),
        pre_expert_norm=_gain_norm(
            compressed_dim,
            sharding_config=compressed_moe_linear_sharding_config(),
        ),
        unpooler=Linear.Config(
            in_features=compressed_dim,
            out_features=dim,
            bias=False,
            param_init=_depth_init(layer_id),
            sharding_config=compressed_moe_linear_sharding_config(has_bias=True),
        ),
        sharding_config=moe_sharding_config(enable_sp=enable_sp),
    )


def _muse_spark_navit_attention_config(
    *,
    latent_dim: int,
    num_heads: int,
) -> VisionAttention.Config:
    return VisionAttention.Config(
        dim=latent_dim,
        num_heads=num_heads,
        inner_attention=FlexInnerAttention.Config(
            sharding_config=vision_inner_attention_sharding_config(),
        ),
        wq=Linear.Config(
            in_features=latent_dim,
            out_features=latent_dim,
            bias=True,
            sharding_config=vision_colwise_sharding_config(input_tp=spmd.R),
        ),
        wk=Linear.Config(
            in_features=latent_dim,
            out_features=latent_dim,
            bias=True,
            sharding_config=vision_colwise_sharding_config(input_tp=spmd.R),
        ),
        wv=Linear.Config(
            in_features=latent_dim,
            out_features=latent_dim,
            bias=True,
            sharding_config=vision_colwise_sharding_config(input_tp=spmd.R),
        ),
        proj=InvariantRowParallelLinear.Config(
            in_features=latent_dim,
            out_features=latent_dim,
            bias=True,
            sharding_config=vision_rowwise_sharding_config(),
        ),
        sharding_config=vision_attention_sharding_config(rope_cache_dp=spmd.V),
    )


def _muse_spark_navit_mlp_config(
    *,
    latent_dim: int,
    mlp_hidden_dim: int,
) -> VisionMLP.Config:
    return VisionMLP.Config(
        fc1=Linear.Config(
            in_features=latent_dim,
            out_features=mlp_hidden_dim,
            bias=True,
            sharding_config=vision_colwise_sharding_config(),
        ),
        fc2=InvariantRowParallelLinear.Config(
            in_features=mlp_hidden_dim,
            out_features=latent_dim,
            bias=True,
            sharding_config=vision_rowwise_sharding_config(),
        ),
        act_fn=GELU.Config(),
    )


def muse_spark_navit_encoder_config(
    *,
    latent_dim: int = 1536,
    num_layers: int = 50,
    num_heads: int = 16,
    num_channels: int = 3,
    mlp_hidden_dim: int = 8960,
    patch_size: int = 14,
    patch_temporal: int = 2,
    downsample_factor: int = 2,
    sparse_attention_factor: int = 4,
    pos_emb_grid_h: int = 32,
    pos_emb_grid_w: int = 32,
    use_rope: bool = True,
    rope_theta: float = 10000.0,
    grid_sample_mode: str = "bilinear",
) -> NaViTEncoder.Config:
    head_dim = latent_dim // num_heads
    output_dim = latent_dim * downsample_factor**2
    patch_input_dim = num_channels * patch_size * patch_size * patch_temporal
    return NaViTEncoder.Config(
        latent_dim=latent_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        patch_size=patch_size,
        patch_temporal=patch_temporal,
        downsample_factor=downsample_factor,
        sparse_attention_factor=sparse_attention_factor,
        pos_emb_grid_h=pos_emb_grid_h,
        pos_emb_grid_w=pos_emb_grid_w,
        rope_freq=VisionRopeFreq.Config(
            head_dim=head_dim,
            rope_theta=rope_theta,
            sharding_config=vision_rope_freq_sharding_config(),
        ),
        conv1=Linear.Config(
            in_features=patch_input_dim,
            out_features=latent_dim,
            bias=False,
            sharding_config=vision_invariant_linear_sharding_config(),
        ),
        ln_pre=LayerNorm.Config(
            normalized_shape=latent_dim,
            sharding_config=invariant_norm_sharding_config(),
        ),
        block=VisionTransformerBlock.Config(
            norm1=LayerNorm.Config(
                normalized_shape=latent_dim,
                sharding_config=invariant_norm_sharding_config(),
            ),
            norm2=LayerNorm.Config(
                normalized_shape=latent_dim,
                sharding_config=invariant_norm_sharding_config(),
            ),
            attn=_muse_spark_navit_attention_config(
                latent_dim=latent_dim,
                num_heads=num_heads,
            ),
            mlp=_muse_spark_navit_mlp_config(
                latent_dim=latent_dim,
                mlp_hidden_dim=mlp_hidden_dim,
            ),
        ),
        ln_post=LayerNorm.Config(
            normalized_shape=latent_dim,
            sharding_config=invariant_norm_sharding_config(),
        ),
        use_rope=use_rope,
        grid_sample_mode=grid_sample_mode,
        pos_emb_grid_order="legacy_spark",
        param_init=_VISION_POSITION_INIT,
        sharding_config=navit_encoder_sharding_config(),
    )


def muse_spark_navit_adapter_config(
    *,
    input_dim: int = 6144,
    output_dim: int = 4096,
) -> MuseSparkNaViTAdapter.Config:
    return MuseSparkNaViTAdapter.Config(
        c_fc=Linear.Config(
            in_features=input_dim,
            out_features=output_dim,
            bias=False,
            sharding_config=cp_replicated_vision_sharding_config(
                vision_invariant_linear_sharding_config()
            ),
        ),
        c_proj=Linear.Config(
            in_features=output_dim,
            out_features=output_dim,
            bias=False,
            sharding_config=cp_replicated_vision_sharding_config(
                vision_invariant_linear_sharding_config()
            ),
        ),
    )


def muse_spark_navit_configs(
    vision_adapter_dim: int,
) -> tuple[NaViTEncoder.Config, MuseSparkNaViTAdapter.Config]:
    """Build the production NaViT encoder and its dimension-matched adapter."""
    encoder = muse_spark_navit_encoder_config()
    adapter = muse_spark_navit_adapter_config(
        input_dim=encoder.output_dim,
        output_dim=vision_adapter_dim,
    )
    return encoder, adapter


def _muse_spark_config(
    *,
    dim: int,
    num_layers: int,
    vocab_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_context_length: int,
    global_attention_layers: set[int],
    sliding_window_size: int,
    compressed_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    position_temperature_floor_scale: float | None,
    position_temperature_query_scale: float | None,
    query_scale: float,
    post_norm_gain_centers: tuple[float, ...],
    post_attention_norm_has_weight: bool,
    final_norm_gain_center: float,
    embedding_multiplier: float,
    output_multiplier: float,
    output_soft_cap_temp: float,
    final_norm_input_dtype: torch.dtype,
    route_scale: float,
    router_logit_multiplier: float,
    route_norm_epsilon: float,
    load_balance_coeff: float | None,
    expert_bias_init: float,
    attn_backend: str,
    moe_comm_backend: str,
    parallelism: ParallelismConfig,
    dataset_modalities: tuple[Literal["vision", "audio"], ...],
    audio_encoder: MuseSparkAudioEncoder.Config,
    vision_encoder: NaViTEncoder.Config,
    vision_adapter: MuseSparkNaViTAdapter.Config,
) -> MuseSparkModel.Config:
    enable_sp = parallelism.enable_sequence_parallel
    enable_ep = parallelism.expert_parallel_degree > 1
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    if attn_backend not in {"flex", "varlen"}:
        raise ValueError("Muse Spark supports only attn_backend='flex' or 'varlen'")
    if moe_comm_backend != "standard":
        raise ValueError("Muse Spark supports only moe_comm_backend='standard'")

    assert len(post_norm_gain_centers) == num_layers

    if audio_encoder.output_size != dim:
        raise ValueError(
            "audio_encoder.output_size must match Muse Spark model dim "
            f"{dim}; got {audio_encoder.output_size}"
        )
    vision_projection = Linear.Config(
        in_features=vision_adapter.c_proj.out_features,
        out_features=dim,
        bias=False,
        param_init=_LINEAR_INIT,
        sharding_config=cp_replicated_vision_sharding_config(
            vision_invariant_linear_sharding_config()
        ),
    )
    perception_emb_norm = _gain_norm(
        dim,
        elementwise_affine=False,
        sharding_config=perception_norm_sharding_config(enable_sp=enable_sp),
    )

    layers = []
    for layer_id, post_norm_gain_center in enumerate(post_norm_gain_centers):
        is_global_attention = layer_id in global_attention_layers
        feed_forward = _moe_config(
            layer_id=layer_id,
            dim=dim,
            compressed_dim=compressed_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            post_norm_gain_center=post_norm_gain_center,
            route_scale=route_scale,
            router_logit_multiplier=router_logit_multiplier,
            route_norm_epsilon=route_norm_epsilon,
            load_balance_coeff=load_balance_coeff,
            expert_bias_init=expert_bias_init,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
        )
        layers.append(
            MuseSparkTransformerBlock.Config(
                attention=_attention_config(
                    layer_id=layer_id,
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    max_context_length=max_context_length,
                    use_rope=not is_global_attention,
                    window_size=(None if is_global_attention else sliding_window_size),
                    query_scale=query_scale,
                    position_temperature_floor_scale=(
                        position_temperature_floor_scale
                        if is_global_attention
                        else None
                    ),
                    position_temperature_query_scale=(
                        position_temperature_query_scale
                        if is_global_attention
                        else None
                    ),
                    attn_backend=attn_backend,
                    enable_sp=enable_sp,
                ),
                attention_norm=_gain_norm(
                    dim,
                    sharding_config=norm_sharding,
                ),
                post_attention_norm=_gain_norm(
                    dim,
                    elementwise_affine=post_attention_norm_has_weight,
                    gain_center=post_norm_gain_center,
                    eps=_POST_NORM_EPS,
                    sharding_config=norm_sharding,
                ),
                attention_residual=_residual_gate(
                    dim, layer_id=layer_id, enable_sp=enable_sp
                ),
                ffn_norm=_gain_norm(
                    dim,
                    sharding_config=norm_sharding,
                ),
                post_ffn_norm=_input_scaled_gain_norm(
                    dim,
                    gain_center=post_norm_gain_center,
                    eps=_POST_NORM_EPS,
                    sharding_config=norm_sharding,
                ),
                ffn_residual=_residual_gate(
                    dim, layer_id=layer_id, enable_sp=enable_sp
                ),
                feed_forward=feed_forward,
                branch_dtype=torch.bfloat16,
                sharding_config=transformer_block_sharding_config(enable_sp=enable_sp),
            )
        )

    return MuseSparkModel.Config(
        max_context_length=max_context_length,
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=MuseSparkEmbeddingWithNorm.Config(
            embedding=Embedding.Config(
                num_embeddings=vocab_size,
                embedding_dim=dim,
                param_init=_EMBEDDING_INIT,
                sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
            ),
            norm=MuseSparkRMSNorm.Config(
                normalized_shape=dim,
                eps=_NORM_EPS,
                elementwise_affine=False,
                sharding_config=norm_sharding,
            ),
            multiplier=embedding_multiplier,
        ),
        layers=layers,
        # Preserve the zero-centered checkpoint parameterization while starting
        # every final norm with unit effective scale.
        norm=dataclasses.replace(
            _gain_norm(
                dim, gain_center=final_norm_gain_center, sharding_config=norm_sharding
            ),
            param_init=(
                {"weight": nn.init.ones_}
                if final_norm_gain_center == 0.0
                else _GAIN_NORM_INIT
            ),
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=MuseSparkLMHead.Config(
            in_features=dim,
            out_features=vocab_size,
            bias=False,
            output_multiplier=output_multiplier,
            output_soft_cap_temp=output_soft_cap_temp,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        residual_dtype=torch.float32,
        final_norm_input_dtype=final_norm_input_dtype,
        dataset_modalities=dataset_modalities,
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        vision_adapter=vision_adapter,
        vision_projection=vision_projection,
        perception_emb_norm=perception_emb_norm,
    )


def _debugmodel(
    attn_backend: str,
    moe_comm_backend: str,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
    dataset_modalities: tuple[Literal["vision", "audio"], ...],
) -> MuseSparkModel.Config:
    audio_encoder = muse_spark_audio_encoder_config(
        output_size=_DEBUG_MODEL_DIM,
        patch_size=4,
        kernel_size=7,
        input_channels=1,
        patchify_sizes=(2, 2),
        hidden_sizes=(16, 32),
    )
    vision_encoder = muse_spark_navit_encoder_config(
        latent_dim=64,
        num_layers=3,
        num_heads=4,
        mlp_hidden_dim=128,
        patch_size=14,
        patch_temporal=2,
        downsample_factor=2,
        sparse_attention_factor=2,
        pos_emb_grid_h=2,
        pos_emb_grid_w=2,
    )
    vision_adapter = muse_spark_navit_adapter_config(
        input_dim=vision_encoder.output_dim,
        output_dim=128,
    )
    num_layers = 8
    return _muse_spark_config(
        dim=_DEBUG_MODEL_DIM,
        num_layers=num_layers,
        vocab_size=2048,
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        max_context_length=seq_len,
        global_attention_layers={3, 7},
        sliding_window_size=128,
        compressed_dim=128,
        expert_hidden_dim=256,
        num_experts=8,
        top_k=2,
        position_temperature_floor_scale=32.0,
        position_temperature_query_scale=0.125,
        query_scale=_SCALE_QUERY_NUMERATOR / math.sqrt(32),
        post_norm_gain_centers=tuple(
            1.0 / math.sqrt(2.0 * (layer_id + 1)) for layer_id in range(num_layers)
        ),
        post_attention_norm_has_weight=True,
        final_norm_gain_center=1.0,
        embedding_multiplier=1.25,
        output_multiplier=0.5,
        output_soft_cap_temp=30.0,
        final_norm_input_dtype=torch.float32,
        route_scale=1.5,
        router_logit_multiplier=0.75,
        route_norm_epsilon=1e-6,
        load_balance_coeff=None,
        expert_bias_init=0.125,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        dataset_modalities=dataset_modalities,
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        vision_adapter=vision_adapter,
        parallelism=parallelism,
    )


def _muse_spark_816b_a42b(
    attn_backend: str,
    moe_comm_backend: str,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
    dataset_modalities: tuple[Literal["vision", "audio"], ...],
) -> MuseSparkModel.Config:
    vision_encoder, vision_adapter = muse_spark_navit_configs(
        _MUSE_SPARK_816B_VISION_ADAPTER_DIM,
    )
    audio_encoder = muse_spark_audio_encoder_config(
        output_size=_MUSE_SPARK_816B_DIM,
    )
    num_layers = 62
    return _muse_spark_config(
        dim=_MUSE_SPARK_816B_DIM,
        num_layers=num_layers,
        vocab_size=_MUSE_SPARK_816B_VOCAB_SIZE,
        num_heads=128,
        num_kv_heads=16,
        head_dim=64,
        max_context_length=seq_len,
        global_attention_layers=set(range(1, num_layers, 4)),
        sliding_window_size=2048,
        compressed_dim=4096,
        expert_hidden_dim=4096,
        num_experts=256,
        top_k=8,
        position_temperature_floor_scale=None,
        position_temperature_query_scale=None,
        query_scale=_SCALE_QUERY_NUMERATOR / math.sqrt(64),
        post_norm_gain_centers=(1.0,) * num_layers,
        post_attention_norm_has_weight=False,
        final_norm_gain_center=0.0,
        embedding_multiplier=1.0,
        output_multiplier=1.0 / math.sqrt(32.0),
        output_soft_cap_temp=20.0,
        final_norm_input_dtype=torch.bfloat16,
        route_scale=1.0,
        router_logit_multiplier=1.0 / math.sqrt(32.0),
        route_norm_epsilon=1e-15,
        load_balance_coeff=1e-3,
        expert_bias_init=0.0,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        dataset_modalities=dataset_modalities,
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        vision_adapter=vision_adapter,
        parallelism=parallelism,
    )


# Each tuple contains the model factory and its default training context length.
MODEL_FLAVORS: dict[str, tuple[Callable[..., MuseSparkModel.Config], int]] = {
    "816B-A42B": (
        partial(_muse_spark_816b_a42b, dataset_modalities=()),
        32768,
    ),
    "816B-A42B_mm": (
        partial(_muse_spark_816b_a42b, dataset_modalities=("vision", "audio")),
        32768,
    ),
    "debugmodel": (
        partial(_debugmodel, dataset_modalities=()),
        512,
    ),
    "debugmodel_mm": (
        partial(_debugmodel, dataset_modalities=("vision", "audio")),
        512,
    ),
}

_MUSE_SPARK_MAX_CONTEXT_LENGTHS = {
    "816B-A42B": 1_048_576,
    "816B-A42B_mm": 1_048_576,
    "debugmodel": 32_768,
    "debugmodel_mm": 512,
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    moe_comm_backend: str = "standard",
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> MuseSparkModel.Config:
    if flavor not in MODEL_FLAVORS:
        available = ", ".join(sorted(MODEL_FLAVORS))
        raise ValueError(
            f"Unknown Muse Spark flavor {flavor!r}; available flavors: {available}"
        )

    get_config, default_context_len = MODEL_FLAVORS[flavor]
    max_context_len = _MUSE_SPARK_MAX_CONTEXT_LENGTHS[flavor]
    context_len = default_context_len if seq_len is None else seq_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        seq_len=context_len,
        parallelism=ParallelismConfig() if parallelism is None else parallelism,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            config = cast(
                MuseSparkModel.Config,
                converter.build().convert(config),
            )

    return config
