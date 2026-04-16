# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from copy import deepcopy
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RoPE, TransformerBlock
from torchtitan.models.common.attention import (
    FlexAttention,
    ScaledDotProductAttention,
    VarlenAttention,
)
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.model_spec import ModelSpec

from .model import Qwen3Model, Qwen3TransformerBlock
from .parallelize import parallelize_qwen3
from .state_dict_adapter import Qwen3StateDictAdapter

__all__ = [
    "parallelize_qwen3",
    "Qwen3Model",
    "qwen3_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}

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
        "w1": partial(nn.init.trunc_normal_, std=0.02),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _qwen3_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _build_qwen3_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    inner_attention=None,
    mask_type: str = "causal",
    fuse_qkv: bool = False,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for dense Qwen3 models with depth-scaled inits."""
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen3_norm(dim),
                ffn_norm=_qwen3_norm(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=(
                        inner_attention
                        if inner_attention is not None
                        else ScaledDotProductAttention.Config()
                    ),
                    fuse_qkv=fuse_qkv,
                    mask_type=mask_type,
                    rope_backend="cos_sin",
                    qk_norm=_qwen3_norm(head_dim),
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _build_qwen3_moe_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for MoE Qwen3 models with depth-scaled inits."""
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen3_norm(dim),
                ffn_norm=_qwen3_norm(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=ScaledDotProductAttention.Config(),
                    rope_backend="cos_sin",
                    qk_norm=_qwen3_norm(head_dim),
                ),
                moe=make_moe_config(
                    num_experts=num_experts,
                    score_before_experts=False,
                    router=make_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=_depth_init(layer_id),
                        top_k=top_k,
                        score_func="softmax",
                        route_norm=True,
                    ),
                    experts=make_experts_config(
                        dim=dim,
                        hidden_dim=moe_hidden_dim,
                        num_experts=num_experts,
                        param_init=_depth_experts_init(layer_id),
                    ),
                ),
            )
        )
    return layers


def _debugmodel() -> Qwen3Model.Config:
    dim = 256
    head_dim = 128
    n_layers = 8
    vocab_size = 2048
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=3072,
        ),
    )


def _debugmodel_fused_qkv() -> Qwen3Model.Config:
    dim = 256
    head_dim = 128
    n_layers = 8
    vocab_size = 2048
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=3072,
            fuse_qkv=True,
        ),
    )


def _debugmodel_flex() -> Qwen3Model.Config:
    config = _debugmodel()
    flex_cfg = FlexAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = flex_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _debugmodel_flex_flash() -> Qwen3Model.Config:
    from torchtitan.tools.utils import has_cuda_capability

    if has_cuda_capability(10, 0):
        # NOTE: On NVIDIA Blackwell, to use FLASH backend we need
        # block size at least (256, 128) due to how the kernel works.
        block_size = (256, 128)
    elif has_cuda_capability(9, 0):
        block_size = (128, 128)
    else:
        raise ValueError(
            "Flash backend of FlexAttention is only supported on Hopper or Blackwell"
        )
    config = _debugmodel()
    flex_cfg = FlexAttention.Config(
        block_size=block_size, kernel_options={"BACKEND": "FLASH"}
    )
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = flex_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _debugmodel_varlen() -> Qwen3Model.Config:
    config = _debugmodel()
    varlen_cfg = VarlenAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = varlen_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _0_6b() -> Qwen3Model.Config:
    dim = 1024
    head_dim = 128
    n_layers = 28
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=3072,
        ),
    )


def _0_6b_varlen() -> Qwen3Model.Config:
    config = _0_6b()
    varlen_cfg = VarlenAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = varlen_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _1_7b() -> Qwen3Model.Config:
    dim = 2048
    head_dim = 128
    n_layers = 28
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=6144,
        ),
    )


def _1_7b_varlen() -> Qwen3Model.Config:
    config = _1_7b()
    varlen_cfg = VarlenAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = varlen_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _4b() -> Qwen3Model.Config:
    dim = 2560
    head_dim = 128
    n_layers = 36
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=9728,
        ),
    )


def _8b() -> Qwen3Model.Config:
    dim = 4096
    head_dim = 128
    n_layers = 36
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=12288,
        ),
    )


def _8b_varlen() -> Qwen3Model.Config:
    config = _8b()
    varlen_cfg = VarlenAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = varlen_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _14b() -> Qwen3Model.Config:
    dim = 5120
    head_dim = 128
    n_layers = 40
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=40,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=17408,
        ),
    )


def _32b() -> Qwen3Model.Config:
    dim = 5120
    head_dim = 128
    n_layers = 64
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=25600,
        ),
    )


# Qwen3-MoE models


def _debugmodel_moe() -> Qwen3Model.Config:
    dim = 256
    head_dim = 128
    n_layers = 8
    vocab_size = 2048
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=64,
            top_k=8,
        ),
    )


def _30b_a3b() -> Qwen3Model.Config:
    dim = 2048
    head_dim = 128
    n_layers = 48
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=262144,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=128,
            top_k=8,
        ),
    )


def _235b_a22b() -> Qwen3Model.Config:
    dim = 4096
    head_dim = 128
    n_layers = 94
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=5000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=1536,
            num_experts=128,
            top_k=8,
        ),
    )


qwen3_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_fused_qkv": _debugmodel_fused_qkv,
    "debugmodel_flex": _debugmodel_flex,
    "debugmodel_flex_flash": _debugmodel_flex_flash,
    "debugmodel_varlen": _debugmodel_varlen,
    "0.6B": _0_6b,
    "0.6B_varlen": _0_6b_varlen,
    "1.7B": _1_7b,
    "1.7B_varlen": _1_7b_varlen,
    "4B": _4b,
    "8B": _8b,
    "8B_varlen": _8b_varlen,
    "14B": _14b,
    "32B": _32b,
    "debugmodel_moe": _debugmodel_moe,
    "30B-A3B": _30b_a3b,
    "235B-A22B": _235b_a22b,
}


def model_registry(flavor: str) -> ModelSpec:
    config = qwen3_configs[flavor]()
    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
