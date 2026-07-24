# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    CosSinRoPE,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.attention import (
    FusedQKVLinear,
    QKVLinear,
    VarlenAttention,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_token_dispatcher_config,
)
from torchtitan.models.common.linear import ScaledBiasRowwiseLinear
from torchtitan.models.common.moe import RoutedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec
from .model import Attention, GptOssModel, GptOssTransformerBlock
from .moe import GptOssGroupedExperts, GptOssMoE
from .parallelize import parallelize_gptoss
from .state_dict_adapter import GptOssStateDictAdapter

__all__ = [
    "parallelize_gptoss",
    "GptOssModel",
    "gptoss_configs",
]

_NORM_INIT = {"weight": nn.init.ones_}
# GPT-OSS uses std=0.02 for embeddings (model-specific)
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}


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


def _make_gptoss_attn_config(
    *,
    dim: int,
    layer_id: int,
    attn_backend: str = "varlen",
    n_heads: int = 64,
    n_kv_heads: int = 8,
    head_dim: int = 64,
    sliding_window_size: int | None = None,
    fuse_qkv: bool = True,
    rope: RoPE.Config,
) -> Attention.Config:
    """Build a fully-specified GPT-OSS Attention.Config for a single layer.

    All linear sub-configs have their in_features/out_features set.
    All linear params use depth-scaled init (including wq/wkv/wo).
    Sinks also use depth-scaled init.
    """

    inner_attention = get_attention_config(attn_backend)

    if sliding_window_size is not None and isinstance(
        inner_attention, VarlenAttention.Config
    ):
        inner_attention = dataclasses.replace(
            inner_attention, window_size=(sliding_window_size - 1, 0)
        )

    sinks_std = depth_scaled_std(0.02, layer_id)
    sinks_init = {
        "sinks": partial(
            nn.init.trunc_normal_, std=sinks_std, a=-3 * sinks_std, b=3 * sinks_std
        )
    }

    if fuse_qkv:
        qkv = FusedQKVLinear.Config(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            wqkv=Linear.Config(
                in_features=dim,
                out_features=(n_heads + 2 * n_kv_heads) * head_dim,
                bias=True,
                param_init=_depth_init(layer_id),
            ),
        )
    else:
        qkv = QKVLinear.Config(
            head_dim=head_dim,
            wq=Linear.Config(
                in_features=dim,
                out_features=n_heads * head_dim,
                bias=True,
                param_init=_depth_init(layer_id),
            ),
            wkv=Linear.Config(
                in_features=dim,
                out_features=n_kv_heads * head_dim,
                bias=True,
                param_init=_depth_init(layer_id),
            ),
        )

    return Attention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=qkv,
        wo=ScaledBiasRowwiseLinear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            bias=True,
            param_init=_depth_init(layer_id),
        ),
        sliding_window_size=sliding_window_size,
        inner_attention=inner_attention,
        param_init=sinks_init,
        rope=dataclasses.replace(rope),
    )


def _make_gptoss_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    layer_id: int,
    top_k: int,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> RoutedExperts.Config:
    """Build a fully-specified RoutedExperts.Config for a single GPT-OSS layer."""
    std = depth_scaled_std(0.02, layer_id)
    experts_init = {
        "mlp1_weight_EGD": partial(
            nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std
        ),
        "mlp1_bias_EG": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "mlp2_weight_EDF": partial(
            nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std
        ),
        "mlp2_bias_ED": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
    }
    return RoutedExperts.Config(
        inner_experts=GptOssGroupedExperts.Config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            param_init=experts_init,
        ),
        token_dispatcher=make_token_dispatcher_config(
            num_experts=num_experts,
            top_k=top_k,
            comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
            hidden_dim=dim,
        ),
    )


def _build_gptoss_layers(
    *,
    dim: int,
    n_layers: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    load_balance_coeff: float,
    attn_backend: str = "varlen",
    fuse_qkv: bool = True,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    rope: RoPE.Config,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for GPT-OSS.

    Even-indexed layers (0, 2, 4, ...) use sliding window attention.
    All dimensional fields are set directly.
    """
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = _make_gptoss_attn_config(
            dim=dim,
            layer_id=layer_id,
            attn_backend=attn_backend,
            sliding_window_size=128 if layer_id % 2 == 0 else None,
            fuse_qkv=fuse_qkv,
            rope=rope,
        )
        routed_experts_cfg = _make_gptoss_experts_config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            layer_id=layer_id,
            top_k=top_k,
            moe_comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        )
        moe_cfg = GptOssMoE.Config(
            num_experts=num_experts,
            load_balance_coeff=load_balance_coeff,
            routed_experts=routed_experts_cfg,
            router=TokenChoiceTopKRouter.Config(
                num_experts=num_experts,
                score_func="softmax",
                route_norm=True,
                gate=Linear.Config(
                    in_features=dim,
                    out_features=num_experts,
                    bias=True,
                    param_init=_depth_init(layer_id),
                ),
                top_k=top_k,
            ),
        )
        layer_cfg = GptOssTransformerBlock.Config(
            attention=attn_cfg,
            attention_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            moe=moe_cfg,
        )
        layers.append(layer_cfg)
    return layers


def _debugmodel(
    moe_comm_backend: str,
    attn_backend: str = "varlen",
) -> GptOssModel.Config:
    dim = 256
    hidden_dim = 2880
    n_layers = 4
    return GptOssModel.Config(
        vocab_size=2048,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=2048,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gptoss_layers(
            fuse_qkv=True,
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=8,
            top_k=4,
            load_balance_coeff=1e-3,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            rope=CosSinRoPE.Config(
                dim=64,
                max_seq_len=131072,
                theta=150000.0,
                scaling="yarn",
                rope_factor=32,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=False,
                original_seq_len=4096,
            ),
        ),
    )


def _20b(
    moe_comm_backend: str,
    attn_backend: str = "varlen",
) -> GptOssModel.Config:
    dim = 2880
    hidden_dim = 2880
    n_layers = 24
    return GptOssModel.Config(
        dim=dim,
        vocab_size=201088,
        tok_embeddings=Embedding.Config(
            num_embeddings=201088, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=201088,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gptoss_layers(
            fuse_qkv=True,
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=32,
            top_k=4,
            load_balance_coeff=1e-3,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            rope=CosSinRoPE.Config(
                dim=64,
                max_seq_len=131072,
                theta=150000.0,
                scaling="yarn",
                rope_factor=32,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=False,
                original_seq_len=4096,
            ),
        ),
    )


def _120b(
    moe_comm_backend: str,
    attn_backend: str = "varlen",
) -> GptOssModel.Config:
    dim = 2880
    hidden_dim = 2880
    n_layers = 36
    return GptOssModel.Config(
        dim=dim,
        vocab_size=201088,
        tok_embeddings=Embedding.Config(
            num_embeddings=201088, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=201088,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gptoss_layers(
            fuse_qkv=True,
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=128,
            top_k=4,
            load_balance_coeff=1e-3,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            rope=CosSinRoPE.Config(
                dim=64,
                max_seq_len=131072,
                theta=150000.0,
                scaling="yarn",
                rope_factor=32,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=False,
                original_seq_len=4096,
            ),
        ),
    )


gptoss_configs = {
    "debugmodel": _debugmodel,
    "20b": _20b,
    "120b": _120b,
}


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = gptoss_configs[flavor](
        moe_comm_backend=moe_comm_backend,
        attn_backend=attn_backend,
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)
    return ModelSpec(
        name="gpt_oss",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=GptOssStateDictAdapter,
    )
