# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    ComplexRoPE,
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_moe_config,
    make_router_config,
    make_routed_experts_config,
    make_gqa_config,
    TpGemmBackend,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.utils import validate_converter_order

from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import Nemotron3NanoModel, NemotronTransformerBlock
from .parallelize import parallelize_nemotron
from .state_dict_adapter import NemotronStateDictAdapter

__all__ = [
    "parallelize_nemotron",
    "Nemotron3NanoModel",
    "nemotron_configs",
]

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


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


def _build_nemotron_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    rope: RoPE.Config,
    num_experts: int,
    top_k_experts: int,
    mamba_state_dim: int,
    mamba_conv_dim: int,
    n_kv_heads: int | None = None,
    fuse_qkv: bool = True,
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
) -> list[NemotronTransformerBlock.Config]:
    inner_attention = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        is_mamba = layer_id % 2 == 0
        layers.append(
            NemotronTransformerBlock.Config(
                is_mamba_block=is_mamba,
                mamba_state_dim=mamba_state_dim,
                mamba_conv_dim=mamba_conv_dim,
                mamba_input_projection=Linear.Config(
                    in_features=dim, out_features=mamba_conv_dim, param_init=_LINEAR_INIT
                ) if is_mamba else None,
                mamba_output_projection=Linear.Config(
                    in_features=mamba_conv_dim, out_features=dim, param_init=_LINEAR_INIT
                ) if is_mamba else None,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    fuse_qkv=fuse_qkv,
                    rope=rope,
                    tp_gemm_backend=tp_gemm_backend,
                ),
                moe=make_moe_config(
                    num_experts=num_experts,
                    router=make_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=_LINEAR_INIT,
                        top_k=top_k_experts,
                    ),
                    routed_experts=make_routed_experts_config(
                        dim=dim,
                        hidden_dim=hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k_experts,
                        param_init={
                            "w1_EFD": _LINEAR_INIT["weight"],
                            "w2_EDF": _depth_init(layer_id)["weight"],
                            "w3_EFD": _LINEAR_INIT["weight"],
                        },
                        comm_backend="standard",
                    ),
                ),
            )
        )
    return layers

def _debugmodel(
    attn_backend: str, tp_gemm_backend: TpGemmBackend = "default"
) -> Nemotron3NanoModel.Config:
    dim = 256
    n_heads = 16
    n_layers = 4
    num_experts = 4
    top_k_experts = 2
    return Nemotron3NanoModel.Config(
        dim=dim,
        vocab_size=262144,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        tok_embeddings=Embedding.Config(
            num_embeddings=262144, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim, out_features=262144, param_init=_output_linear_init(dim)
        ),
        layers=_build_nemotron_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            rope=ComplexRoPE.Config(
                dim=dim // n_heads,
                max_context_length=131072,
                theta=500000,
                scaling="llama",
            ),
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            mamba_state_dim=16,
            mamba_conv_dim=256,
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
        ),
    )


def _31b(
    attn_backend: str, tp_gemm_backend: TpGemmBackend = "default"
) -> Nemotron3NanoModel.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    vocab_size = 262144
    num_experts = 128
    top_k_experts = 6
    mamba_state_dim = 16
    mamba_conv_dim = 4096
    return Nemotron3NanoModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        mamba_state_dim=mamba_state_dim,
        mamba_conv_dim=mamba_conv_dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_nemotron_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=1152,
            rope=ComplexRoPE.Config(
                dim=dim // n_heads,
                max_context_length=1000000,
                theta=500000,
                scaling="llama",
            ),
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            mamba_state_dim=mamba_state_dim,
            mamba_conv_dim=mamba_conv_dim,
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
        ),
    )


nemotron_configs = {
    "debugmodel": _debugmodel,
    "31B": _31b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    tp_gemm_backend: TpGemmBackend = "default",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = nemotron_configs[flavor](
        attn_backend=attn_backend, tp_gemm_backend=tp_gemm_backend
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)
    return ModelSpec(
        name="nemotron_nano",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_nemotron,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=NemotronStateDictAdapter,
    )
