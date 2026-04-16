# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE, TransformerBlock
from torchtitan.models.common.attention import FusedQKVLinear, QKVLinear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.param_init import depth_scaled_std
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
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _make_gptoss_attn_config(
    *,
    dim: int,
    layer_id: int,
    n_heads: int = 64,
    n_kv_heads: int = 8,
    head_dim: int = 64,
    sliding_window_size: int = 128,
    fuse_qkv: bool = False,
) -> Attention.Config:
    """Build a fully-specified GPT-OSS Attention.Config for a single layer.

    All linear sub-configs have their in_features/out_features set.
    All linear params use depth-scaled init (including wq/wkv/wo).
    Sinks also use depth-scaled init.
    """
    sinks_init = {
        "sinks": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id))
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
        wo=Linear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            bias=True,
            param_init=_depth_init(layer_id),
        ),
        sliding_window_size=sliding_window_size,
        param_init=sinks_init,
    )


def _make_gptoss_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    layer_id: int,
) -> GptOssGroupedExperts.Config:
    """Build a fully-specified GptOssGroupedExperts.Config for a single layer."""
    std = depth_scaled_std(0.02, layer_id)
    experts_init = {
        "mlp1_weight": partial(nn.init.trunc_normal_, std=std),
        "mlp1_bias": partial(nn.init.trunc_normal_, std=std),
        "mlp2_weight": partial(nn.init.trunc_normal_, std=std),
        "mlp2_bias": partial(nn.init.trunc_normal_, std=std),
    }
    return GptOssGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        param_init=experts_init,
    )


def _build_gptoss_layers(
    *,
    dim: int,
    n_layers: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    score_before_experts: bool,
    load_balance_coeff: float,
    fuse_qkv: bool = False,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for GPT-OSS.

    Even-indexed layers (0, 2, 4, ...) use sliding window attention.
    All dimensional fields are set directly.
    """
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = _make_gptoss_attn_config(
            dim=dim, layer_id=layer_id, fuse_qkv=fuse_qkv
        )
        experts_cfg = _make_gptoss_experts_config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            layer_id=layer_id,
        )
        moe_cfg = GptOssMoE.Config(
            num_experts=num_experts,
            score_before_experts=score_before_experts,
            load_balance_coeff=load_balance_coeff,
            experts=experts_cfg,
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
            use_sliding_attention=(layer_id % 2 == 0),
        )
        layers.append(layer_cfg)
    return layers


def _debugmodel() -> GptOssModel.Config:
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
        output=Linear.Config(
            in_features=dim,
            out_features=2048,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=8,
            top_k=4,
            score_before_experts=False,
            load_balance_coeff=1e-3,
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=131072,
            theta=150000.0,
            backend="cos_sin",
            scaling="yarn",
            rope_factor=32,
            beta_slow=32.0,
            beta_fast=1.0,
            original_seq_len=4096,
        ),
    )


def _20b() -> GptOssModel.Config:
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
        output=Linear.Config(
            in_features=dim,
            out_features=201088,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=32,
            top_k=4,
            score_before_experts=False,
            load_balance_coeff=1e-3,
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=131072,
            theta=150000.0,
            backend="cos_sin",
            scaling="yarn",
            rope_factor=32,
            beta_slow=32.0,
            beta_fast=1.0,
            original_seq_len=4096,
        ),
    )


def _120b() -> GptOssModel.Config:
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
        output=Linear.Config(
            in_features=dim,
            out_features=201088,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=128,
            top_k=4,
            score_before_experts=False,
            load_balance_coeff=1e-3,
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=131072,
            theta=150000.0,
            backend="cos_sin",
            scaling="yarn",
            rope_factor=32,
            beta_slow=32.0,
            beta_fast=1.0,
            original_seq_len=4096,
        ),
    )


gptoss_configs = {
    "debugmodel": _debugmodel,
    "20b": _20b,
    "120b": _120b,
}


def model_registry(flavor: str) -> ModelSpec:
    config = gptoss_configs[flavor]()
    return ModelSpec(
        name="gpt_oss",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=GptOssStateDictAdapter,
    )
