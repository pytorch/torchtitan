# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MiniMax M3 model flavors."""

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
from torchtitan.models.common import (
    ClampedSwiGLU,
    Embedding,
    Linear,
    OffsetRMSNorm,
    Sigmoid,
)
from torchtitan.models.common.attention import FlexInnerAttention, GQAttention
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    head_norm_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_input_colwise_sharding_config,
    replicated_state_sharding_config,
)
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.common.vision_encoder import (
    InvariantRowParallelLinear,
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_sharding_config,
    multimodal_token_embedding_sharding_config,
    transformer_block_sharding_config,
    vision_attention_sharding_config,
    vision_colwise_sharding_config,
    vision_inner_attention_sharding_config,
    vision_invariant_linear_sharding_config,
    vision_projector_sharding_config,
    vision_rowwise_sharding_config,
)

from .attention import (
    MiniMaxSparseAttention,
    MSAIndexer,
    MSAIndexerLoss,
    MSAInnerAttention,
)
from .model import MiniMaxM3Model, MiniMaxM3TransformerBlock
from .sharding import msa_attention_sharding_config, msa_inner_attention_sharding_config
from .vision_encoder import MiniMaxM3VisionEncoder, MiniMaxM3VisionProjector

__all__ = [
    "MiniMaxM3Model",
    "MINIMAX_M3_SPECIAL_TOKENS",
    "MODEL_FLAVORS",
    "build_model_config",
]

MINIMAX_M3_SPECIAL_TOKENS = {
    "image_token": "]<]image[>[",
    "video_token": "]<]video[>[",
    "vision_start_token": "]<]start of image[>[",
    "vision_end_token": "]<]end of image[>[",
    "pad_token": "]!p~[",
}

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_OFFSET_NORM_INIT = {"weight": nn.init.zeros_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_NORM_EPS = 1e-6
_VISION_NORM_EPS = 1e-5


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


def _offset_norm(dim: int, **kwargs) -> OffsetRMSNorm.Config:
    return OffsetRMSNorm.Config(
        dim=dim, eps=_NORM_EPS, param_init=_OFFSET_NORM_INIT, **kwargs
    )


def _indexer_config(
    *, dim: int, num_index_heads: int, index_head_dim: int
) -> MSAIndexer.Config:
    def projection(out_features: int, sharding_config) -> Linear.Config:
        return Linear.Config(
            in_features=dim,
            out_features=out_features,
            param_init=_LINEAR_INIT,
            sharding_config=sharding_config,
        )

    # Index query heads shard with their KV groups; the shared key head replicates.
    replicated = replicated_state_sharding_config("weight")
    return MSAIndexer.Config(
        head_dim=index_head_dim,
        wq=projection(
            num_index_heads * index_head_dim, replicated_input_colwise_sharding_config()
        ),
        wk=projection(index_head_dim, replicated),
        q_norm=_offset_norm(
            index_head_dim, sharding_config=head_norm_sharding_config()
        ),
        k_norm=_offset_norm(index_head_dim, sharding_config=replicated),
    )


def _attention_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope: CosSinRoPE.Config,
    sparse_layer: bool,
    attn_backend: str,
    index_head_dim: int,
    index_block_size: int,
    topk_blocks: int,
    sparse_attention: bool,
    indexer_loss_coeff: float,
    enable_sp: bool,
) -> GQAttention.Config:
    attention = make_gqa_config(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        wqkv_param_init=_LINEAR_INIT,
        wo_param_init=_depth_init(layer_id),
        inner_attention=get_attention_config(attn_backend),
        rope=rope,
        qk_norm=_offset_norm(head_dim),
        enable_sp=enable_sp,
    )
    if not sparse_layer:
        return attention
    inner_attention = MSAInnerAttention.Config(
        index_block_size=index_block_size,
        topk_blocks=topk_blocks,
        sparse=sparse_attention,
        aux_loss=MSAIndexerLoss.Config(
            coeff=indexer_loss_coeff,
            reduce_mesh="loss",
            num_groups=n_kv_heads,
            softmax_scale=head_dim**-0.5,
            index_softmax_scale=index_head_dim**-0.5,
        ),
        sharding_config=msa_inner_attention_sharding_config(),
    )
    # x_TD arrives gathered, so wqkv is a plain Linear (ColumnParallelLinear regathers).
    wqkv = attention.qkv_linear.wqkv
    qkv_linear = dataclasses.replace(
        attention.qkv_linear,
        wqkv=Linear.Config(
            **{
                f.name: getattr(wqkv, f.name)
                for f in dataclasses.fields(wqkv)
                if f.init and f.name != "sharding_config"
            },
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
    )
    replaced = {"inner_attention", "qkv_linear", "sharding_config"}
    return MiniMaxSparseAttention.Config(
        **{
            f.name: getattr(attention, f.name)
            for f in dataclasses.fields(attention)
            if f.init and f.name not in replaced
        },
        inner_attention=inner_attention,
        qkv_linear=qkv_linear,
        indexer=_indexer_config(
            dim=dim, num_index_heads=n_kv_heads, index_head_dim=index_head_dim
        ),
        sharding_config=msa_attention_sharding_config(enable_sp=enable_sp),
    )


def _feed_forward_config(
    *,
    layer_id: int,
    dim: int,
    moe_layer: bool,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    enable_ep: bool,
    enable_sp: bool,
):
    activation_fn = ClampedSwiGLU.Config()
    if not moe_layer:
        return dataclasses.replace(
            make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w13_param_init=_LINEAR_INIT,
                w2_param_init=_depth_init(layer_id),
                enable_sp=enable_sp,
            ),
            activation_fn=activation_fn,
        )
    return make_moe_config(
        num_experts=num_experts,
        router=make_router_config(
            dim=dim,
            num_experts=num_experts,
            gate_param_init=_depth_init(layer_id),
            top_k=top_k,
            score_func=Sigmoid.Config(),
            route_norm=True,
            route_scale=2.0,
        ),
        routed_experts=dataclasses.replace(
            make_routed_experts_config(
                dim=dim,
                hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                param_init=_depth_experts_init(layer_id),
                enable_ep=enable_ep,
            ),
            activation_fn=activation_fn,
        ),
        shared_experts=dataclasses.replace(
            make_shared_expert_ffn_config(
                dim=dim,
                hidden_dim=moe_hidden_dim,
                w13_param_init=_LINEAR_INIT,
                w2_param_init=_depth_init(layer_id),
                enable_sp=enable_sp,
            ),
            activation_fn=activation_fn,
        ),
        load_balance_coeff=1e-3,
        enable_sp=enable_sp,
    )


def _vision_encoder_config(
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    out_dim: int,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    spatial_merge_size: int = 2,
    in_channels: int = 3,
) -> MiniMaxM3VisionEncoder.Config:
    def linear(in_features: int, out_features: int, **kwargs) -> Linear.Config:
        return Linear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            param_init=_LINEAR_INIT,
            **kwargs,
        )

    def row_parallel(in_features: int, out_features: int):
        return InvariantRowParallelLinear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            param_init=_LINEAR_INIT,
            sharding_config=vision_rowwise_sharding_config(),
        )

    def norm() -> LayerNorm.Config:
        return LayerNorm.Config(
            normalized_shape=dim,
            eps=_VISION_NORM_EPS,
            sharding_config=invariant_norm_sharding_config(),
        )

    merged_dim = out_dim * spatial_merge_size**2
    return MiniMaxM3VisionEncoder.Config(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        spatial_merge_size=spatial_merge_size,
        patch_embed_proj=Linear.Config(
            in_features=in_channels * temporal_patch_size * patch_size**2,
            out_features=dim,
            param_init=_LINEAR_INIT,
            sharding_config=vision_invariant_linear_sharding_config(),
        ),
        pre_norm=norm(),
        block=VisionTransformerBlock.Config(
            norm1=norm(),
            norm2=norm(),
            attn=VisionAttention.Config(
                dim=dim,
                num_heads=num_heads,
                wq=linear(
                    dim,
                    dim,
                    sharding_config=vision_colwise_sharding_config(input_tp=spmd.R),
                ),
                wk=linear(
                    dim,
                    dim,
                    sharding_config=vision_colwise_sharding_config(input_tp=spmd.R),
                ),
                wv=linear(
                    dim,
                    dim,
                    sharding_config=vision_colwise_sharding_config(input_tp=spmd.R),
                ),
                proj=row_parallel(dim, dim),
                inner_attention=FlexInnerAttention.Config(
                    sharding_config=vision_inner_attention_sharding_config()
                ),
                sharding_config=vision_attention_sharding_config(rope_cache_dp=spmd.V),
            ),
            mlp=VisionMLP.Config(
                fc1=linear(
                    dim, ffn_dim, sharding_config=vision_colwise_sharding_config()
                ),
                fc2=row_parallel(ffn_dim, dim),
                act_fn=GELU.Config(),
            ),
        ),
        projector=MiniMaxM3VisionProjector.Config(
            spatial_merge_size=spatial_merge_size,
            linear_1=linear(
                dim, out_dim, sharding_config=vision_colwise_sharding_config()
            ),
            linear_2=row_parallel(out_dim, out_dim),
            merge_linear_1=linear(
                merged_dim, out_dim, sharding_config=vision_colwise_sharding_config()
            ),
            merge_linear_2=row_parallel(out_dim, out_dim),
            sharding_config=vision_projector_sharding_config(),
        ),
    )


def _minimax_m3_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope_theta: float,
    vocab_size: int,
    first_moe_layer: int,
    first_sparse_layer: int,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    index_head_dim: int,
    index_block_size: int,
    topk_blocks: int,
    vision_dim: int,
    vision_layers: int,
    vision_heads: int,
    vision_ffn_dim: int,
    attn_backend: str,
    seq_len: int,
    parallelism: ParallelismConfig,
    sparse_attention: bool,
    indexer_loss_coeff: float,
) -> MiniMaxM3Model.Config:
    enable_ep = parallelism.expert_parallel_degree > 1
    enable_sp = parallelism.enable_sequence_parallel
    rope = CosSinRoPE.Config(
        dim=rotary_dim, max_context_length=seq_len, theta=rope_theta
    )
    norm = norm_sharding_config(enable_sp=enable_sp)
    layers = [
        MiniMaxM3TransformerBlock.Config(
            attention=_attention_config(
                layer_id=layer_id,
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rope=rope,
                sparse_layer=layer_id >= first_sparse_layer,
                attn_backend=attn_backend,
                index_head_dim=index_head_dim,
                index_block_size=index_block_size,
                topk_blocks=topk_blocks,
                sparse_attention=sparse_attention,
                indexer_loss_coeff=indexer_loss_coeff,
                enable_sp=enable_sp,
            ),
            feed_forward=_feed_forward_config(
                layer_id=layer_id,
                dim=dim,
                moe_layer=layer_id >= first_moe_layer,
                dense_hidden_dim=dense_hidden_dim,
                moe_hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                enable_ep=enable_ep,
                enable_sp=enable_sp,
            ),
            attention_norm=_offset_norm(dim, sharding_config=norm),
            ffn_norm=_offset_norm(dim, sharding_config=norm),
            sharding_config=transformer_block_sharding_config(
                enable_sp=enable_sp, after_vision_scatter=layer_id == 0
            ),
        )
        for layer_id in range(n_layers)
    ]
    return MiniMaxM3Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        norm=_offset_norm(
            dim, sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp)
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_sharding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        vision_encoder=_vision_encoder_config(
            dim=vision_dim,
            num_layers=vision_layers,
            num_heads=vision_heads,
            ffn_dim=vision_ffn_dim,
            out_dim=dim,
        ),
    )


_DEBUG_SIZES = dict(
    dim=256,
    n_layers=8,
    n_heads=8,
    n_kv_heads=4,
    head_dim=32,
    rotary_dim=16,
    rope_theta=5_000_000.0,
    vocab_size=2048,
    first_moe_layer=1,
    dense_hidden_dim=512,
    moe_hidden_dim=128,
    num_experts=8,
    top_k=2,
    index_head_dim=32,
    index_block_size=16,
    topk_blocks=4,
    vision_dim=128,
    vision_layers=2,
    vision_heads=8,
    vision_ffn_dim=512,
)

_M3_SIZES = dict(
    dim=6144,
    n_layers=60,
    n_heads=64,
    n_kv_heads=4,
    head_dim=128,
    rotary_dim=64,
    vocab_size=200064,
    first_moe_layer=3,
    dense_hidden_dim=12288,
    moe_hidden_dim=3072,
    num_experts=128,
    top_k=4,
    index_head_dim=128,
    index_block_size=128,
    topk_blocks=16,
    vision_dim=1280,
    vision_layers=32,
    vision_heads=16,
    vision_ffn_dim=5120,
)


def _debugmodel(**kwargs) -> MiniMaxM3Model.Config:
    return _minimax_m3_config(**_DEBUG_SIZES, first_sparse_layer=1, **kwargs)


def _debugmodel_flash(**kwargs) -> MiniMaxM3Model.Config:
    return _minimax_m3_config(**_DEBUG_SIZES, first_sparse_layer=0, **kwargs)


def _m3(**kwargs) -> MiniMaxM3Model.Config:
    return _minimax_m3_config(
        **_M3_SIZES, rope_theta=5_000_000.0, first_sparse_layer=3, **kwargs
    )


def _m3_1_flash(**kwargs) -> MiniMaxM3Model.Config:
    return _minimax_m3_config(
        **_M3_SIZES, rope_theta=10_000_000.0, first_sparse_layer=0, **kwargs
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 8192),
    "debugmodel_flash": (_debugmodel_flash, 8192),
    "M3": (_m3, 1048576),
    "M3.1-Flash": (_m3_1_flash, 1048576),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
    sparse_attention: bool = True,
    indexer_loss_coeff: float = 0.01,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> MiniMaxM3Model.Config:
    """Build a MiniMax M3 config.

    ``attn_backend`` selects the full-attention layers' kernel; MSA layers always run
    on FlexAttention. ``sparse_attention=False`` runs every MSA layer as full causal
    attention while the indexer still trains on its KL loss (indexer warmup).
    """
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
        sparse_attention=sparse_attention,
        indexer_loss_coeff=indexer_loss_coeff,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)
    return config
