# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    CosSinRoPE,
    Embedding,
    Linear,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.models.utils import validate_converter_order

from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import Qwen3Model, Qwen3TransformerBlock
from .parallelize import parallelize_qwen3
from .state_dict_adapter import Qwen3StateDictAdapter

__all__ = [
    "parallelize_qwen3",
    "Qwen3Model",
    "qwen3_configs",
]


# Qwen3 uses normal initialization with std=0.02. For residual output
# projections, align with Megatron's scaled initializer. Megatron applies it
# to the attention output and MoE expert down projections:
# https://github.com/NVIDIA/Megatron-LM/blob/d12f6c8c9aff51e166d872fd70151687a8e3f375/megatron/core/transformer/transformer_config.py#L2289-L2303
_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
_EXPERTS_INIT: dict[str, Callable] = {
    "w1_EFD": _LINEAR_INIT["weight"],
    "w2_EDF": _LINEAR_INIT["weight"],
    "w3_EFD": _LINEAR_INIT["weight"],
}

_EPS = 1e-6


def _residual_output_weight_init(n_layers: int) -> Callable:
    std = 0.02 / (2 * n_layers) ** 0.5
    return partial(nn.init.normal_, std=std)


def _residual_output_init(n_layers: int) -> dict[str, Callable]:
    return {
        "weight": _residual_output_weight_init(n_layers),
        "bias": nn.init.zeros_,
    }


def _moe_experts_init(n_layers: int) -> dict[str, Callable]:
    return {
        **_EXPERTS_INIT,
        "w2_EDF": _residual_output_weight_init(n_layers),
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
    fuse_qkv: bool = True,
    attn_backend: str,
    rope: RoPE.Config,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for dense Qwen3 models."""
    inner_attention = get_attention_config(attn_backend)
    layers = []
    for _ in range(n_layers):
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
                    wo_param_init=_LINEAR_INIT,
                    inner_attention=inner_attention,
                    fuse_qkv=fuse_qkv,
                    rope=rope,
                    qk_norm=_qwen3_norm(head_dim),
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_LINEAR_INIT,
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
    fuse_qkv: bool = True,
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    rope: RoPE.Config,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for MoE Qwen3 models."""
    inner_attention = get_attention_config(attn_backend)
    output_init = _residual_output_init(n_layers)
    experts_init = _moe_experts_init(n_layers)
    layers = []
    for _ in range(n_layers):
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
                    wo_param_init=output_init,
                    inner_attention=inner_attention,
                    fuse_qkv=fuse_qkv,
                    rope=rope,
                    qk_norm=_qwen3_norm(head_dim),
                ),
                moe=make_moe_config(
                    num_experts=num_experts,
                    router=make_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=_LINEAR_INIT,
                        top_k=top_k,
                        score_func="softmax",
                        route_norm=True,
                    ),
                    routed_experts=make_routed_experts_config(
                        dim=dim,
                        hidden_dim=moe_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        param_init=experts_init,
                        comm_backend=moe_comm_backend,
                        non_blocking_capacity_factor=non_blocking_capacity_factor,
                    ),
                ),
            )
        )
    return layers


def _debugmodel(attn_backend: str, *, seq_len: int) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=3072,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1000000.0,
            ),
        ),
    )


def _debugmodel_non_fused_qkv(attn_backend: str, *, seq_len: int) -> Qwen3Model.Config:
    # Reverse of the default fused QKV: keeps coverage for the separate
    # wq/wk/wv path now that fuse_qkv defaults to True.
    config = _debugmodel(attn_backend, seq_len=seq_len)
    config.layers = _build_qwen3_layers(
        fuse_qkv=False,
        n_layers=8,
        dim=256,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=3072,
        attn_backend=attn_backend,
        rope=CosSinRoPE.Config(dim=128, max_context_length=seq_len, theta=1000000.0),
    )
    return config


def _0_6b(attn_backend: str, *, seq_len: int) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=3072,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1000000.0,
            ),
        ),
    )


def _1_7b(attn_backend: str, *, seq_len: int) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=6144,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1000000.0,
            ),
        ),
    )


def _4b(attn_backend: str, *, seq_len: int) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=9728,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1000000.0,
            ),
        ),
    )


def _8b(attn_backend: str, *, seq_len: int) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=12288,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1000000.0,
            ),
        ),
    )


def _14b(attn_backend: str, *, seq_len: int) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=40,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=17408,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1000000.0,
            ),
        ),
    )


def _32b(attn_backend: str, *, seq_len: int) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=25600,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1000000.0,
            ),
        ),
    )


# Qwen3-MoE models


def _debugmodel_moe(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    *,
    seq_len: int,
) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_moe_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=64,
            top_k=8,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1000000.0,
            ),
            moe_comm_backend=moe_comm_backend,
        ),
    )


def _30b_a3b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    *,
    seq_len: int,
) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_moe_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=128,
            top_k=8,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=1000000.0,
            ),
            moe_comm_backend=moe_comm_backend,
        ),
    )


def _235b_a22b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    *,
    seq_len: int,
) -> Qwen3Model.Config:
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
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
        ),
        layers=_build_qwen3_moe_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=1536,
            num_experts=128,
            top_k=8,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=seq_len,
                theta=5000000.0,
            ),
            moe_comm_backend=moe_comm_backend,
        ),
    )


qwen3_configs = {
    "debugmodel": (_debugmodel, 4096),
    "debugmodel_non_fused_qkv": (_debugmodel_non_fused_qkv, 4096),
    "0.6B": (_0_6b, 40960),
    "1.7B": (_1_7b, 40960),
    "4B": (_4b, 40960),
    "8B": (_8b, 40960),
    "14B": (_14b, 40960),
    "32B": (_32b, 40960),
    "debugmodel_moe": (_debugmodel_moe, 4096),
    "30B-A3B": (_30b_a3b, 40960),
    "235B-A22B": (_235b_a22b, 40960),
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    get_config, max_context_len = qwen3_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        seq_len=context_len,
        **{"moe_comm_backend": moe_comm_backend}
        if moe_comm_backend is not None
        else {},
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)
    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
