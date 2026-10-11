# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 model flavors."""

import dataclasses
from collections.abc import Callable
from functools import partial

import spmd_types as spmd
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)

from torchtitan.models.base.sharding import ShardingConfig

from torchtitan.models.common import (  # noqa: F401
    Conv1d,
    Embedding,
    GatedRMSNorm,
    Linear,
    RowParallelLinear,
    SiLU,
    Softmax,
)
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.gdn import GatedDeltaNet, GDNKernel, InnerGDN
from torchtitan.models.common.config_utils import (
    fused_gate_up_param_init,
    get_attention_config,
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.decoder_sharding import (
    head_norm_config,
    inner_attention_local_spmd_config,
    lm_head_config,
    norm_config,
    pre_lm_head_norm_config,
    replicated_input_colwise_config,
    residual_activation_placement,
    rope_cache_config,
    rowwise_config,
)
from torchtitan.models.common.moe_sharding import (
    shared_expert_rowwise_config,
    shared_experts_config,
    shared_experts_output_placement,
)
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.param_init import depth_scaled_std  # noqa: F401
from torchtitan.models.common.vision_encoder import (
    InvariantRowParallelLinear,
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_config,
    multimodal_token_embedding_config,
    vision_attention_config,
    vision_colwise_config,
    vision_inner_attention_config,
    vision_invariant_linear_config,
    vision_rowwise_config,
)
from .model import OffsetRMSNorm, Qwen35Attention, Qwen35Model, Qwen35TransformerBlock
from .moe import SigmoidGatedFeedForward
from .rope import MRoPE
from .sharding import (
    attention_config,
    conv_weight_config,
    decoder_norm_config,
    gated_delta_net_config,
    gated_rms_norm_config,
    inner_gated_delta_net_config,
    sigmoid_gate_config,
    sigmoid_gated_w13_config,
    transformer_block_config,
    vision_encoder_config,
    vision_rotary_embedding_config,
)

from .vision_encoder import PatchMerger, Qwen35VisionEncoder, VisionRotaryEmbedding

__all__ = [
    "Qwen35Model",
    "MODEL_FLAVORS",
    "QWEN3_5_SPECIAL_TOKENS",
    "build_model_config",
]

QWEN3_5_SPECIAL_TOKENS = {
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
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
    }


def _a_log_init(param: nn.Parameter) -> None:
    # Match https://github.com/huggingface/transformers/pull/47944 to avoid
    # near-zero decay heads under bf16 initialization.
    param.data.uniform_(0.01, 16.0).log_()


def _linear(
    in_features: int, out_features: int, *, sharding_config: ShardingConfig
) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
        sharding_config=sharding_config,
    )


def _vision_row_parallel_linear(
    in_features: int, out_features: int
) -> InvariantRowParallelLinear.Config:
    return InvariantRowParallelLinear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
        sharding_config=vision_rowwise_config(),
    )


def _offset_norm(dim: int, *, sharding_config: ShardingConfig) -> OffsetRMSNorm.Config:
    return OffsetRMSNorm.Config(
        dim=dim,
        eps=_EPS,
        param_init=_OFFSET_NORM_INIT,
        sharding_config=sharding_config,
    )


def _shared_experts_config(
    *, dim: int, hidden_dim: int, layer_id: int, enable_sp: bool
) -> SigmoidGatedFeedForward.Config:
    """Build Qwen3.5's sigmoid-gated shared-expert config (SwiGLU FFN + gate)."""
    depth_init = _depth_init(layer_id)
    return SigmoidGatedFeedForward.Config(
        # The shared expert gathers once because w13 and the sigmoid gate
        # consume the same input.
        w13=Linear.Config(
            in_features=dim,
            out_features=hidden_dim,
            num_linears=2,
            param_init=fused_gate_up_param_init(_LINEAR_INIT, _LINEAR_INIT),
            sharding_config=sigmoid_gated_w13_config(),
        ),
        w2=RowParallelLinear.Config(
            in_features=hidden_dim,
            out_features=dim,
            param_init=depth_init,
            sharding_config=shared_expert_rowwise_config(
                output_layout=shared_experts_output_placement(enable_sp=enable_sp)
            ),
        ),
        gate=Linear.Config(
            in_features=dim,
            out_features=1,
            param_init=_LINEAR_INIT,
            sharding_config=sigmoid_gate_config(),
        ),
        sharding_config=shared_experts_config(enable_sp=enable_sp),
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
    _norm = LayerNorm.Config(
        normalized_shape=dim,
        eps=layer_norm_eps,
        sharding_config=invariant_norm_config(),
    )
    return Qwen35VisionEncoder.Config(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        spatial_merge_size=spatial_merge_size,
        num_position_embeddings=num_position_embeddings,
        patch_embed_proj=_linear(
            patch_dim, dim, sharding_config=vision_invariant_linear_config()
        ),
        block=VisionTransformerBlock.Config(
            norm1=_norm,
            norm2=_norm,
            attn=VisionAttention.Config(
                dim=dim,
                num_heads=num_heads,
                wq=_linear(
                    dim, dim, sharding_config=vision_colwise_config(input_tp=spmd.R)
                ),
                wk=_linear(
                    dim, dim, sharding_config=vision_colwise_config(input_tp=spmd.R)
                ),
                wv=_linear(
                    dim, dim, sharding_config=vision_colwise_config(input_tp=spmd.R)
                ),
                proj=_vision_row_parallel_linear(dim, dim),
                inner_attention=FlexInnerAttention.Config(
                    sharding_config=vision_inner_attention_config()
                ),
                sharding_config=vision_attention_config(rope_cache_dp=spmd.V),
            ),
            mlp=VisionMLP.Config(
                fc1=_linear(dim, ffn_dim, sharding_config=vision_colwise_config()),
                fc2=_vision_row_parallel_linear(ffn_dim, dim),
            ),
        ),
        rotary_pos_emb=VisionRotaryEmbedding.Config(
            dim=head_dim // 2,
            theta=rope_theta,
            sharding_config=vision_rotary_embedding_config(),
        ),
        merger=PatchMerger.Config(
            spatial_merge_size=spatial_merge_size,
            merged_hidden_size=merged_hidden_size,
            norm=LayerNorm.Config(
                normalized_shape=dim,
                eps=layer_norm_eps,
                sharding_config=invariant_norm_config(),
            ),
            fc1=_linear(
                merged_hidden_size,
                merged_hidden_size,
                sharding_config=vision_colwise_config(),
            ),
            fc2=_vision_row_parallel_linear(merged_hidden_size, out_hidden_size),
        ),
        param_init=_POS_EMBED_INIT,
        sharding_config=vision_encoder_config(),
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
    enable_sp: bool,
) -> Qwen35Attention.Config:
    """Build a fully-specified, sharded Qwen35Attention.Config."""
    inner_attention = dataclasses.replace(
        get_attention_config(attn_backend),
        sharding_config=inner_attention_local_spmd_config(),
    )
    return Qwen35Attention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        # The per-layer rope ``cache`` buffer is replicated on TP.
        rope=dataclasses.replace(rope, sharding_config=rope_cache_config()),
        wq=Linear.Config(
            in_features=dim,
            out_features=n_heads * head_dim * 2,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_config(),
        ),
        wk=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_config(),
        ),
        wv=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_config(),
        ),
        # RowParallelLinear reduce-scatters with SP and all-reduces otherwise.
        wo=RowParallelLinear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        q_norm=_offset_norm(head_dim, sharding_config=head_norm_config()),
        k_norm=_offset_norm(head_dim, sharding_config=head_norm_config()),
        inner_attention=inner_attention,
        sharding_config=attention_config(enable_sp=enable_sp),
    )


def _qwen35_deltanet_config(
    *,
    dim: int,
    n_key_heads: int,
    n_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    layer_id: int,
    enable_sp: bool,
    conv_kernel_size: int = 4,
) -> GatedDeltaNet.Config:
    """Build a fully-specified, head-sharded GatedDeltaNet.Config.

    Input projections are colwise-sharded (head-sharded output) and out_proj is
    row-parallel. Conv weights and per-head A_log/dt_bias are Shard(0). The
    recurrence runs on rank-local heads via a single local SPMD boundary.
    """
    key_dim = n_key_heads * key_head_dim
    value_dim = n_value_heads * value_head_dim

    def _proj(in_f: int, out_f: int, init: dict) -> Linear.Config:
        return Linear.Config(
            in_features=in_f,
            out_features=out_f,
            bias=False,
            param_init=init,
            sharding_config=replicated_input_colwise_config(),
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
            sharding_config=conv_weight_config(),
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
        inner_gated_delta_net=InnerGDN.Config(
            conv_kernel_size=conv_kernel_size,
            kernel=GDNKernel.Config(),
            sharding_config=inner_gated_delta_net_config(),
        ),
        # Keep RMS normalization and gating in FP32 until the final output cast,
        # following the FLA behavior noted by Hugging Face:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L216-L218
        norm=GatedRMSNorm.Config(
            dim=value_head_dim,
            eps=1e-6,
            activation_fn=SiLU.Config(),
            param_init={"weight": nn.init.ones_},
            sharding_config=gated_rms_norm_config(),
        ),
        # RowParallelLinear reduce-scatters with SP and all-reduces otherwise.
        out_proj=RowParallelLinear.Config(
            in_features=value_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        param_init={
            "A_log": _a_log_init,
            "dt_bias": nn.init.ones_,
        },
        sharding_config=gated_delta_net_config(enable_sp=enable_sp),
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
    enable_sp: bool,
    has_vision_encoder: bool,
) -> list[Qwen35TransformerBlock.Config]:
    """Build sharded per-layer configs for dense Qwen3.5 models."""
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
                enable_sp=enable_sp,
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
                enable_sp=enable_sp,
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
                    w13_param_init=_LINEAR_INIT,
                    w2_param_init=_depth_init(layer_id),
                    enable_sp=enable_sp,
                ),
                attention_norm=_offset_norm(
                    dim, sharding_config=decoder_norm_config(enable_sp=enable_sp)
                ),
                ffn_norm=_offset_norm(
                    dim, sharding_config=norm_config(enable_sp=enable_sp)
                ),
                sharding_config=transformer_block_config(
                    enable_sp=enable_sp,
                    after_vision_scatter=has_vision_encoder and layer_id == 0,
                ),
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
    enable_sp: bool,
    enable_ep: bool,
    has_vision_encoder: bool,
) -> list[Qwen35TransformerBlock.Config]:
    """Build sharded per-layer configs for MoE Qwen3.5 models with shared expert."""
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
                enable_sp=enable_sp,
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
                enable_sp=enable_sp,
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
                        score_func=Softmax.Config(),
                        route_norm=True,
                    ),
                    routed_experts=make_routed_experts_config(
                        dim=dim,
                        hidden_dim=moe_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        param_init=_depth_experts_init(layer_id),
                        enable_ep=enable_ep,
                    ),
                    shared_experts=_shared_experts_config(
                        dim=dim,
                        hidden_dim=shared_expert_hidden_dim,
                        layer_id=layer_id,
                        enable_sp=enable_sp,
                    ),
                    enable_sp=enable_sp,
                ),
                attention_norm=_offset_norm(
                    dim, sharding_config=decoder_norm_config(enable_sp=enable_sp)
                ),
                ffn_norm=_offset_norm(
                    dim, sharding_config=norm_config(enable_sp=enable_sp)
                ),
                sharding_config=transformer_block_config(
                    enable_sp=enable_sp,
                    after_vision_scatter=has_vision_encoder and layer_id == 0,
                ),
            )
        )
    return layers


def _debugmodel(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> Qwen35Model.Config:
    """Debug config for Qwen3.5 with vision encoder."""
    dim = 256
    head_dim = 64
    rotary_dim = 16
    n_layers = 8
    vocab_size = 248320
    # mrope_section sum must equal rotary_dim / 2 (8 for rotary_dim=16).
    # Real models use [11, 11, 10] with rotary_dim=64.
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            # Attention Gym fused chunk GDN requires K=V=128 on SM80+.
            key_head_dim=128,
            value_head_dim=128,
            enable_sp=parallelism.enable_sequence_parallel,
            has_vision_encoder=True,
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
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> Qwen35Model.Config:
    """Debug MoE config for Qwen3.5 with shared expert."""
    dim = 256
    head_dim = 64
    rotary_dim = 16
    n_layers = 4
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            key_head_dim=128,
            value_head_dim=128,
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
            has_vision_encoder=True,
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


def _0_8b(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> Qwen35Model.Config:
    """Qwen3.5-0.8B dense config with vision encoder.

    The HF checkpoint ties the input embedding and lm_head weights
    (tie_word_embeddings=true), so this config ties them as well.
    """
    dim = 1024
    head_dim = 256
    rotary_dim = 64  # partial_rotary_factor=0.25 → head_dim * 0.25
    n_layers = 24
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            enable_sp=parallelism.enable_sequence_parallel,
            has_vision_encoder=True,
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


def _2b(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> Qwen35Model.Config:
    """Qwen3.5-2B dense config with vision encoder.

    The HF checkpoint ties the input embedding and lm_head weights
    (tie_word_embeddings=true), so this config ties them as well.
    """
    dim = 2048
    head_dim = 256
    rotary_dim = 64  # partial_rotary_factor=0.25
    n_layers = 24
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            enable_sp=parallelism.enable_sequence_parallel,
            has_vision_encoder=True,
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


def _4b(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> Qwen35Model.Config:
    """Qwen3.5-4B dense config with vision encoder.

    The HF checkpoint ties the input embedding and lm_head weights
    (tie_word_embeddings=true), so this config ties them as well.
    """
    dim = 2560
    head_dim = 256
    rotary_dim = 64
    n_layers = 32
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            enable_sp=parallelism.enable_sequence_parallel,
            has_vision_encoder=True,
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


def _9b(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> Qwen35Model.Config:
    """Qwen3.5-9B dense config with vision encoder."""
    dim = 4096
    head_dim = 256
    rotary_dim = 64
    n_layers = 32
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            enable_sp=parallelism.enable_sequence_parallel,
            has_vision_encoder=True,
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


def _27b(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> Qwen35Model.Config:
    """Qwen3.5-27B dense config with vision encoder."""
    dim = 5120
    head_dim = 256
    rotary_dim = 64
    n_layers = 64
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            enable_sp=parallelism.enable_sequence_parallel,
            has_vision_encoder=True,
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
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> Qwen35Model.Config:
    """Qwen3.5-35B-A3B MoE config with vision encoder."""
    dim = 2048
    head_dim = 256
    rotary_dim = 64
    n_layers = 40
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
            has_vision_encoder=True,
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
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> Qwen35Model.Config:
    """Qwen3.5-122B-A10B MoE config with vision encoder."""
    dim = 3072
    head_dim = 256
    rotary_dim = 64
    n_layers = 48
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
            has_vision_encoder=True,
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
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> Qwen35Model.Config:
    """Qwen3.5-397B-A17B MoE config with vision encoder."""
    dim = 4096
    head_dim = 256
    rotary_dim = 64
    n_layers = 60
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_config(),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
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
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
            has_vision_encoder=True,
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


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 4096),
    "debugmodel_moe": (_debugmodel_moe, 4096),
    "0.8B": (_0_8b, 262144),
    "2B": (_2b, 262144),
    "4B": (_4b, 262144),
    "9B": (_9b, 262144),
    "27B": (_27b, 262144),
    "35B-A3B": (_35b_a3b, 262144),
    "122B-A10B": (_122b_a10b, 262144),
    "397B-A17B": (_397b_a17b, 262144),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Qwen35Model.Config:
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
        parallelism=parallelism or ParallelismConfig(),
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)
    return config
