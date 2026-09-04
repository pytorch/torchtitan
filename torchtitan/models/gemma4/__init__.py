# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Gemma-4 Model Registry and Configs

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
    TransformerBlock,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
    TpGemmBackend,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.utils import validate_converter_order

from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import Gemma4Model, Gemma4TransformerBlock
from .parallelize import parallelize_gemma4
from .state_dict_adapter import Gemma4StateDictAdapter

__all__ = [
    "parallelize_gemma4",
    "Gemma4Model",
    "gemma4_configs",
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


def _build_gemma4_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    rope: RoPE.Config,
    n_kv_heads: int | None = None,
    fuse_qkv: bool = True,
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    sliding_window_size: int = 4096,
) -> list[TransformerBlock.Config]:
    """Build a list of per-layer Gemma4TransformerBlock configs with depth-scaled inits.
    
    Gemma-4 uses hybrid attention: sliding-window for most layers, global for final layer.
    """
    inner_attention = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        # Last layer uses global attention, others use sliding window
        use_global_attn = layer_id == n_layers - 1
        
        layers.append(
            Gemma4TransformerBlock.Config(
                use_global_attention=use_global_attn,
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
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                    tp_gemm_backend=tp_gemm_backend,
                ),
            )
        )
    return layers


def _debugmodel(
    attn_backend: str, tp_gemm_backend: TpGemmBackend = "default"
) -> Gemma4Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Gemma4Model.Config(
        dim=dim,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim, out_features=2048, param_init=_output_linear_init(dim)
        ),
        layers=_build_gemma4_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            rope=ComplexRoPE.Config(
                dim=dim // n_heads,
                max_context_length=256000,
                theta=500000,
                scaling="none",
            ),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
        ),
    )


def _12b(
    attn_backend: str, tp_gemm_backend: TpGemmBackend = "default"
) -> Gemma4Model.Config:
    """Gemma-4 12B configuration.
    
    Specifications:
    - Hidden dimension: 3584
    - Attention heads: 28
    - KV heads: 7 (4:1 grouped-query attention)
    - Layers: 42
    - Vocabulary size: 262144
    - Context length: 256K tokens
    """
    dim = 3584
    n_heads = 28
    n_kv_heads = 7
    n_layers = 42
    vocab_size = 262144
    
    return Gemma4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        sliding_window_size=4096,
        enable_sliding_window=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gemma4_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            rope=ComplexRoPE.Config(
                dim=dim // n_heads,
                max_context_length=256000,
                theta=500000,
                scaling="none",
            ),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
            sliding_window_size=4096,
        ),
    )


def _31b(
    attn_backend: str, tp_gemm_backend: TpGemmBackend = "default"
) -> Gemma4Model.Config:
    """Gemma-4 31B configuration.
    
    Specifications:
    - Hidden dimension: 5120
    - Attention heads: 40
    - KV heads: 10 (4:1 grouped-query attention)
    - Layers: 56
    - Vocabulary size: 262144
    - Context length: 256K tokens
    - Sliding window: 4096 tokens
    """
    dim = 5120
    n_heads = 40
    n_kv_heads = 10
    n_layers = 56
    vocab_size = 262144

    return Gemma4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        sliding_window_size=4096,
        enable_sliding_window=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gemma4_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            rope=ComplexRoPE.Config(
                dim=dim // n_heads,
                max_context_length=256000,
                theta=500000,
                scaling="none",
            ),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
            sliding_window_size=4096,
        ),
    )


gemma4_configs = {
    "debugmodel": _debugmodel,
    "12b": _12b,
    "12B": _12b,
    "31b": _31b,
    "31B": _31b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    tp_gemm_backend: TpGemmBackend = "default",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    """Register Gemma-4 model with TorchTitan.
    
    Args:
        flavor: Model size ("12b", "debugmodel")
        attn_backend: Attention backend ("flex", "sdpa")
        tp_gemm_backend: Tensor parallel GEMM backend
        converters: Optional config converters for custom experimentation
    
    Returns:
        ModelSpec for training with TorchTitan
    """
    config = gemma4_configs[flavor](
        attn_backend=attn_backend, tp_gemm_backend=tp_gemm_backend
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)
    return ModelSpec(
        name="gemma4",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_gemma4,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Gemma4StateDictAdapter,
    )
