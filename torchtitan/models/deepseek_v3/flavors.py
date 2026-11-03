# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 model flavors."""

import copy
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
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    FeedForward,
    HiMidLoLinear,
    Linear,
    RMSNorm,
    RoPE,
    RowParallelLinear,
    Sigmoid,
    Softmax,
    TransformerBlock,
    UnaryActivationFn,
)
from torchtitan.models.common.attention import InnerAttention, MLAAttention
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    mla_attention_local_spmd_sharding_config,
    mla_kv_projection_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_input_colwise_sharding_config,
    replicated_state_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.moe.sharding import (
    router_gate_sharding_config,
    router_sharding_config,
)
from torchtitan.models.common.param_init import depth_scaled_std

from .attention import DeepSeekMLAAttention
from .model import DeepSeekV3Model, DeepSeekV3TransformerBlock
from .moe import DeepSeekV3Router
from .mtp import MTPCrossEntropyLoss, MTPDecoder, MTPTransformerBlock
from .sharding import (
    mla_attention_sharding_config,
    mtp_block_sharding_config,
    mtp_projection_sharding_config,
)

__all__ = [
    "DeepSeekV3Model",
    "DeepSeekV3Router",
    "MTPCrossEntropyLoss",
    "MTPDecoder",
    "MTPTransformerBlock",
    "MODEL_FLAVORS",
    "build_model_config",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


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


def make_deepseek_v3_router_config(
    *,
    dim: int,
    num_experts: int,
    gate_param_init: dict[str, Callable],
    top_k: int = 1,
    score_func: UnaryActivationFn.Config | None = None,
    route_norm: bool = False,
    route_scale: float = 1.0,
    num_expert_groups: int | None = None,
    num_limited_groups: int | None = None,
    bias: bool = False,
) -> DeepSeekV3Router.Config:
    return DeepSeekV3Router.Config(
        num_experts=num_experts,
        gate=HiMidLoLinear.Config(
            in_features=dim,
            out_features=num_experts,
            backward_mode="hi_mid_lo",
            bias=bias,
            param_init=gate_param_init,
            sharding_config=router_gate_sharding_config(),
        ),
        top_k=top_k,
        score_func=Sigmoid.Config() if score_func is None else score_func,
        route_norm=route_norm,
        route_scale=route_scale,
        num_expert_groups=num_expert_groups,
        num_limited_groups=num_limited_groups,
        sharding_config=router_sharding_config(),
    )


def make_mla_attention_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float = 1.0,
    attn_backend: str,
    linear_init: dict[str, Callable],
    norm_init: dict[str, Callable],
    depth_init: Callable[[int], dict[str, Callable]],
    rope: RoPE.Config,
    attention_config_factory: Callable[
        [str], InnerAttention.Config
    ] = get_attention_config,
    enable_sp: bool,
) -> DeepSeekMLAAttention.Config:
    """Build a fully-specified, sharded ``DeepSeekMLAAttention.Config``.

    All Linear and RMSNorm sub-configs have their dimensional fields set. When
    ``q_lora_rank == 0``, sets ``wq`` (not ``wq_a``/``wq_b``); when
    ``q_lora_rank > 0``, sets ``wq_a``/``wq_b`` (not ``wq``).

    Low-rank projections (``wkv_a``, ``wq_a``) and their norms keep replicated
    weights on TP; up-projections are column-parallel and ``wo`` is
    row-parallel back to the residual layout selected by ``enable_sp``.
    """
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    wkv_b = Linear.Config(
        in_features=kv_lora_rank,
        out_features=n_heads * (qk_nope_head_dim + v_head_dim),
        param_init=linear_init,
        sharding_config=mla_kv_projection_sharding_config(cp=spmd.S(0)),
    )
    inner_attention = MLAAttention.Config(
        wkv_b=wkv_b,
        packed_kv_head_dim=qk_nope_head_dim + v_head_dim,
        inner_attention=attention_config_factory(attn_backend),
        sharding_config=mla_attention_local_spmd_sharding_config(),
    )

    if q_lora_rank == 0:
        wq = Linear.Config(
            in_features=dim,
            out_features=n_heads * qk_head_dim,
            param_init=linear_init,
            sharding_config=replicated_input_colwise_sharding_config(),
        )
        wq_a = None
        wq_b = None
        # q_norm is unused when q_lora_rank == 0 (never built), but the field is
        # required on DeepSeekMLAAttention.Config so we supply a placeholder.
        q_norm = RMSNorm.Config(normalized_shape=1, eps=1e-6, param_init=norm_init)
    else:
        wq = None
        wq_a = Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=linear_init,
            sharding_config=replicated_state_sharding_config("weight"),
        )
        wq_b = Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * qk_head_dim,
            param_init=linear_init,
            sharding_config=replicated_input_colwise_sharding_config(),
        )
        q_norm = RMSNorm.Config(
            normalized_shape=q_lora_rank,
            eps=1e-6,
            param_init=norm_init,
            sharding_config=replicated_state_sharding_config("weight"),
        )

    return DeepSeekMLAAttention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=mscale,
        wq=wq,
        wq_a=wq_a,
        wq_b=wq_b,
        q_norm=q_norm,
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=linear_init,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=kv_lora_rank,
            eps=1e-6,
            param_init=norm_init,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        wo=RowParallelLinear.Config(
            in_features=n_heads * v_head_dim,
            out_features=dim,
            param_init=depth_init(layer_id),
            sharding_config=rowwise_sharding_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        inner_attention=inner_attention,
        rope=dataclasses.replace(rope, sharding_config=rope_cache_sharding_config()),
        sharding_config=mla_attention_sharding_config(enable_sp=enable_sp),
    )


def build_mla_moe_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: UnaryActivationFn.Config,
    router_route_scale: float = 1.0,
    router_route_norm: bool = False,
    aux_loss_coeff: float | None = None,
    attn_backend: str,
    linear_init: dict[str, Callable],
    norm_init: dict[str, Callable],
    depth_init: Callable[[int], dict[str, Callable]],
    depth_experts_init: Callable[[int], dict[str, Callable]],
    rope: RoPE.Config,
    attention_config_factory: Callable[
        [str], InnerAttention.Config
    ] = get_attention_config,
    router_config_factory: Callable[
        ..., TokenChoiceTopKRouter.Config
    ] = make_router_config,
    first_layer_sharding_config: ShardingConfig | None = None,
    norm_eps: float = 1e-6,
    enable_sp: bool,
    enable_ep: bool,
) -> list[TransformerBlock.Config]:
    """Build the sharded per-layer ``DeepSeekV3TransformerBlock`` configs.

    Layers with layer_id < n_dense_layers get a DenseFeedForward.
    Layers with layer_id >= n_dense_layers get an MoE.

    Router and expert inits are constructed per-layer so depth-scaled
    initializers are correct for each layer's position. The router factory lets
    architecture variants select their routing policy without changing this
    shared layer builder; it must return a sharded router config.
    ``first_layer_sharding_config`` sets layer 0's own boundary, for decoders
    whose embedding output is not in the residual layout. ``norm_eps`` is the
    epsilon of each block's ``attention_norm`` and ``ffn_norm`` (HF
    ``rms_norm_eps``); the MLA q/kv norms keep 1e-6.
    """
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = make_mla_attention_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            attn_backend=attn_backend,
            linear_init=linear_init,
            norm_init=norm_init,
            depth_init=depth_init,
            rope=rope,
            attention_config_factory=attention_config_factory,
            enable_sp=enable_sp,
        )

        ffn_cfg: FeedForward.Config
        if layer_id < n_dense_layers:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w13_param_init=linear_init,
                w2_param_init=depth_init(layer_id),
                enable_sp=enable_sp,
            )
        else:
            ffn_cfg = make_moe_config(
                num_experts=num_experts,
                router=router_config_factory(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=depth_init(layer_id),
                    top_k=router_top_k,
                    score_func=copy.deepcopy(router_score_func),
                    route_scale=router_route_scale,
                    route_norm=router_route_norm,
                ),
                routed_experts=make_routed_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    top_k=router_top_k,
                    param_init=depth_experts_init(layer_id),
                    enable_ep=enable_ep,
                ),
                shared_experts=make_shared_expert_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w13_param_init=linear_init,
                    w2_param_init=depth_init(layer_id),
                    enable_sp=enable_sp,
                ),
                load_balance_coeff=1e-3,
                aux_loss_coeff=aux_loss_coeff,
                enable_sp=enable_sp,
            )

        layers.append(
            DeepSeekV3TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=norm_init,
                    sharding_config=norm_sharding,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=norm_init,
                    sharding_config=norm_sharding,
                ),
                feed_forward=ffn_cfg,
                sharding_config=first_layer_sharding_config if layer_id == 0 else None,
            )
        )
    return layers


def _build_dsv3_layers(
    *,
    router_num_expert_groups: int | None = None,
    router_num_limited_groups: int | None = None,
    **kwargs,
) -> list[TransformerBlock.Config]:
    """Thin wrapper: ``build_mla_moe_layers`` with DeepSeek V3's own inits."""
    return build_mla_moe_layers(
        **kwargs,
        linear_init=_LINEAR_INIT,
        norm_init=_NORM_INIT,
        depth_init=_depth_init,
        depth_experts_init=_depth_experts_init,
        router_config_factory=partial(
            make_deepseek_v3_router_config,
            num_expert_groups=router_num_expert_groups,
            num_limited_groups=router_num_limited_groups,
        ),
    )


def _build_mtp_layers(
    inner_cfg: DeepSeekV3TransformerBlock.Config,
    *,
    dim: int,
    num_mtp_layers: int,
    enable_sp: bool,
) -> list[MTPTransformerBlock.Config]:
    """Build MTP blocks from copies of the (already sharded) ``inner_cfg``."""
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    mtp_layers = []
    for _ in range(num_mtp_layers):
        mtp_layers.append(
            MTPTransformerBlock.Config(
                attention=copy.deepcopy(inner_cfg.attention),
                feed_forward=copy.deepcopy(inner_cfg.feed_forward),
                attention_norm=copy.deepcopy(inner_cfg.attention_norm),
                ffn_norm=copy.deepcopy(inner_cfg.ffn_norm),
                enorm=RMSNorm.Config(
                    normalized_shape=dim, eps=1e-6, sharding_config=norm_sharding
                ),
                hnorm=RMSNorm.Config(
                    normalized_shape=dim, eps=1e-6, sharding_config=norm_sharding
                ),
                eh_proj=Linear.Config(
                    in_features=dim * 2,
                    out_features=dim,
                    bias=False,
                    sharding_config=mtp_projection_sharding_config(enable_sp=enable_sp),
                ),
                mtp_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=1e-6,
                    sharding_config=pre_lm_head_norm_sharding_config(
                        enable_sp=enable_sp
                    ),
                ),
                sharding_config=mtp_block_sharding_config(enable_sp=enable_sp),
            )
        )
    return mtp_layers


def _debugmodel(
    attn_backend: str,
    num_mtp_layers: int = 0,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> DeepSeekV3Model.Config:
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

    enable_sp = parallelism.enable_sequence_parallel
    layers = _build_dsv3_layers(
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
        router_score_func=Sigmoid.Config(),
        router_route_norm=True,
        aux_loss_coeff=1e-3,
        attn_backend=attn_backend,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=seq_len,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        enable_sp=enable_sp,
        enable_ep=parallelism.expert_parallel_degree > 1,
    )
    return DeepSeekV3Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=1e-6,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        mtp_layers=_build_mtp_layers(
            # pyrefly: ignore [bad-argument-type]
            layers[-1],
            dim=dim,
            num_mtp_layers=num_mtp_layers,
            enable_sp=enable_sp,
        ),
    )


def _16b(
    attn_backend: str,
    num_mtp_layers: int = 0,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> DeepSeekV3Model.Config:
    dim = 2048
    n_layers = 27
    vocab_size = 102400
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    num_experts = 64
    n_dense_layers = 1

    enable_sp = parallelism.enable_sequence_parallel
    layers = _build_dsv3_layers(
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
        router_top_k=6,
        router_score_func=Sigmoid.Config(),
        router_route_norm=True,
        aux_loss_coeff=1e-3,
        attn_backend=attn_backend,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=seq_len,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        enable_sp=enable_sp,
        enable_ep=parallelism.expert_parallel_degree > 1,
    )
    return DeepSeekV3Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=1e-6,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        mtp_layers=_build_mtp_layers(
            # pyrefly: ignore [bad-argument-type]
            layers[-1],
            dim=dim,
            num_mtp_layers=num_mtp_layers,
            enable_sp=enable_sp,
        ),
    )


def _236b(
    attn_backend: str,
    num_mtp_layers: int = 0,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> DeepSeekV3Model.Config:
    dim = 5120
    n_layers = 60
    vocab_size = 102400
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 1536
    num_shared_experts = 2
    dense_hidden_dim = 12288
    rope_dim = 64
    num_experts = 160
    n_dense_layers = 1

    enable_sp = parallelism.enable_sequence_parallel
    layers = _build_dsv3_layers(
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
        router_top_k=6,
        router_score_func=Softmax.Config(),
        router_num_expert_groups=8,
        router_num_limited_groups=3,
        router_route_scale=16.0,
        aux_loss_coeff=1e-3,
        attn_backend=attn_backend,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=seq_len,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        enable_sp=enable_sp,
        enable_ep=parallelism.expert_parallel_degree > 1,
    )
    return DeepSeekV3Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=1e-6,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        mtp_layers=_build_mtp_layers(
            # pyrefly: ignore [bad-argument-type]
            layers[-1],
            dim=dim,
            num_mtp_layers=num_mtp_layers,
            enable_sp=enable_sp,
        ),
    )


def _671b(
    attn_backend: str,
    num_mtp_layers: int = 0,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> DeepSeekV3Model.Config:
    dim = 7168
    n_layers = 61
    vocab_size = 129280
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 2048
    num_shared_experts = 1
    dense_hidden_dim = 18432
    rope_dim = 64
    num_experts = 256
    n_dense_layers = 3

    enable_sp = parallelism.enable_sequence_parallel
    layers = _build_dsv3_layers(
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
        router_score_func=Sigmoid.Config(),
        router_num_expert_groups=8,
        router_num_limited_groups=4,
        router_route_scale=2.5,
        router_route_norm=True,
        aux_loss_coeff=1e-3,
        attn_backend=attn_backend,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=seq_len,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        enable_sp=enable_sp,
        enable_ep=parallelism.expert_parallel_degree > 1,
    )
    return DeepSeekV3Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=1e-6,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        mtp_layers=_build_mtp_layers(
            # pyrefly: ignore [bad-argument-type]
            layers[-1],
            dim=dim,
            num_mtp_layers=num_mtp_layers,
            enable_sp=enable_sp,
        ),
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 16384),
    "16B": (_16b, 16384),
    "236B": (_236b, 16384),
    "671B": (_671b, 16384),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
    num_mtp_layers: int = 0,
) -> DeepSeekV3Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        num_mtp_layers=num_mtp_layers,
        seq_len=context_len,
        parallelism=parallelism,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)
    return config
