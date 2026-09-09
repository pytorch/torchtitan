# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Gemma-4 Model Registry and Configs

import copy
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    VarlenAttention,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    TpGemmBackend,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.utils import validate_converter_order

from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    Gemma4Attention,
    Gemma4FeedForward,
    Gemma4GlobalSDPA,
    Gemma4Model,
    Gemma4QKVLinear,
    Gemma4RoPE,
    Gemma4TransformerBlock,
)
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


def _make_gemma4_ffn_config(
    *,
    dim: int,
    hidden_dim: int,
    w1_param_init: dict[str, Callable],
    w2w3_param_init: dict[str, Callable],
    tp_gemm_backend: TpGemmBackend = "default",
) -> Gemma4FeedForward.Config:
    return Gemma4FeedForward.Config(
        w1=Linear.Config(
            in_features=dim, out_features=hidden_dim, param_init=w1_param_init
        ),
        w2=Linear.Config(
            in_features=hidden_dim, out_features=dim, param_init=w2w3_param_init
        ),
        w3=Linear.Config(
            in_features=dim, out_features=hidden_dim, param_init=w2w3_param_init
        ),
    )


def _build_gemma4_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    rope: RoPE.Config,
    n_kv_heads: int | None = None,
    head_dim: int = 256,
    global_head_dim: int = 512,
    global_kv_heads: int | None = None,
    attention_k_eq_v: bool = False,
    fuse_qkv: bool = False,
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    sliding_window_size: int = 1024,
    global_attn_interval: int = 6,
) -> list[Gemma4TransformerBlock.Config]:
    """Build a list of per-layer Gemma4TransformerBlock configs with depth-scaled inits.

    Gemma-4 uses hybrid attention: interleaved sliding-window with periodic global layers
    according to global_attn_interval (default 6 for 5:1 ratio).
    """
    inner_attention = get_attention_config(attn_backend)

    layers: list[Gemma4TransformerBlock.Config] = []
    for layer_id in range(n_layers):
        use_global_attn = (layer_id + 1) % global_attn_interval == 0
        actual_head_dim = (
            global_head_dim
            if use_global_attn and global_head_dim is not None
            else head_dim
        )
        if use_global_attn and global_kv_heads is not None:
            actual_kv_heads = global_kv_heads
        else:
            actual_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads

        # Gemma-4 invariant: sliding layers (head_dim=256) use FlexAttention;
        # global layers (head_dim=512) exceed SRAM/LDS limits for multi-head GQA
        # across all FlashAttention/FlexAttention backends (see Dao-AILab/flash-attention#2427
        # and huggingface/transformers#45201) and dispatch to PyTorch's native C++ SDPA.
        if use_global_attn and actual_head_dim > 256:
            layer_inner_attn = Gemma4GlobalSDPA.Config()
        elif use_global_attn:
            layer_inner_attn = copy.deepcopy(inner_attention)
        else:
            layer_inner_attn = copy.deepcopy(inner_attention)
            if isinstance(layer_inner_attn, VarlenAttention.Config):
                layer_inner_attn.window_size = (sliding_window_size, 0)

        wq = Linear.Config(
            in_features=dim,
            out_features=n_heads * actual_head_dim,
            param_init=_LINEAR_INIT,
        )
        wk = Linear.Config(
            in_features=dim,
            out_features=actual_kv_heads * actual_head_dim,
            param_init=_LINEAR_INIT,
        )
        wv = (
            None
            if use_global_attn and attention_k_eq_v
            else Linear.Config(
                in_features=dim,
                out_features=actual_kv_heads * actual_head_dim,
                param_init=_LINEAR_INIT,
            )
        )
        qkv_linear = Gemma4QKVLinear.Config(
            head_dim=actual_head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
        )
        wo = Linear.Config(
            in_features=n_heads * actual_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        )
        if use_global_attn:
            layer_rope = Gemma4RoPE.Config(
                dim=actual_head_dim,
                max_context_length=rope.max_context_length,
                theta=1000000.0,
                partial_rotary_factor=0.25,
                scaling="none",
            )
        else:
            layer_rope = Gemma4RoPE.Config(
                dim=actual_head_dim,
                max_context_length=rope.max_context_length,
                theta=10000.0,
                partial_rotary_factor=1.0,
                scaling="none",
            )

        attention_cfg = Gemma4Attention.Config(
            n_heads=n_heads,
            dim=dim,
            qkv_linear=qkv_linear,
            wo=wo,
            qk_norm=RMSNorm.Config(
                normalized_shape=actual_head_dim, eps=1e-6, param_init=_NORM_INIT
            ),
            n_kv_heads=actual_kv_heads,
            head_dim=actual_head_dim,
            inner_attention=layer_inner_attn,
            rope=layer_rope,
            attn_scale=1.0,
            v_norm=RMSNorm.Config(
                normalized_shape=actual_head_dim,
                eps=1e-6,
                elementwise_affine=False,
            ),
        )

        layers.append(
            Gemma4TransformerBlock.Config(
                use_global_attention=use_global_attn,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT
                ),
                post_attention_norm=RMSNorm.Config(
                    normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT
                ),
                post_ffn_norm=RMSNorm.Config(
                    normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT
                ),
                attention=attention_cfg,
                feed_forward=_make_gemma4_ffn_config(
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
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    *,
    seq_len: int,
) -> Gemma4Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Gemma4Model.Config(
        dim=dim,
        vocab_size=2048,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim, out_features=2048, param_init=_output_linear_init(dim)
        ),
        layers=_build_gemma4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            rope=Gemma4RoPE.Config(
                dim=dim // n_heads,
                max_context_length=seq_len,
                theta=10000.0,
                scaling="none",
            ),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
        ),
    )


def _e2b(
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    *,
    seq_len: int,
) -> Gemma4Model.Config:
    """Gemma-4 E2B configuration (Edge 2B)."""
    dim = 1536
    intermediate_size = 6144
    n_heads = 8
    n_kv_heads = 1
    n_layers = 35
    vocab_size = 262144
    sliding_window = 512

    return Gemma4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        sliding_window_size=sliding_window,
        enable_sliding_window=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gemma4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=intermediate_size,
            head_dim=256,
            global_head_dim=512,
            global_kv_heads=1,
            attention_k_eq_v=False,
            rope=Gemma4RoPE.Config(
                dim=256,
                max_context_length=seq_len,
                theta=10000.0,
                scaling="none",
            ),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
            sliding_window_size=sliding_window,
            global_attn_interval=5,
        ),
    )


def _e4b(
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    *,
    seq_len: int,
) -> Gemma4Model.Config:
    """Gemma-4 E4B configuration (Edge 4B)."""
    dim = 2560
    intermediate_size = 10240
    n_heads = 8
    n_kv_heads = 2
    n_layers = 42
    vocab_size = 262144
    sliding_window = 512

    return Gemma4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        sliding_window_size=sliding_window,
        enable_sliding_window=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gemma4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=intermediate_size,
            head_dim=256,
            global_head_dim=512,
            global_kv_heads=2,
            attention_k_eq_v=False,
            rope=Gemma4RoPE.Config(
                dim=256,
                max_context_length=seq_len,
                theta=10000.0,
                scaling="none",
            ),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
            sliding_window_size=sliding_window,
            global_attn_interval=6,
        ),
    )


def _12b(
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    *,
    seq_len: int,
) -> Gemma4Model.Config:
    """Gemma-4 12B configuration."""
    dim = 3840
    intermediate_size = 15360
    n_heads = 16
    n_kv_heads = 8
    n_layers = 48
    vocab_size = 262144
    sliding_window = 1024

    return Gemma4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        sliding_window_size=sliding_window,
        enable_sliding_window=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gemma4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=intermediate_size,
            head_dim=256,
            global_head_dim=512,
            global_kv_heads=1,
            attention_k_eq_v=True,
            rope=Gemma4RoPE.Config(
                dim=256,
                max_context_length=seq_len,
                theta=10000.0,
                scaling="none",
            ),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
            sliding_window_size=sliding_window,
            global_attn_interval=6,
        ),
    )


def _26b_a4b(
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    *,
    seq_len: int,
) -> Gemma4Model.Config:
    """Gemma-4 26B A4B configuration."""
    dim = 2816
    intermediate_size = 2112
    n_heads = 16
    n_kv_heads = 8
    n_layers = 30
    vocab_size = 262144
    sliding_window = 1024

    return Gemma4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        sliding_window_size=sliding_window,
        enable_sliding_window=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gemma4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=intermediate_size,
            head_dim=256,
            global_head_dim=512,
            global_kv_heads=2,
            attention_k_eq_v=True,
            rope=Gemma4RoPE.Config(
                dim=256,
                max_context_length=seq_len,
                theta=10000.0,
                scaling="none",
            ),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
            sliding_window_size=sliding_window,
            global_attn_interval=6,
        ),
    )


def _31b(
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    *,
    seq_len: int,
) -> Gemma4Model.Config:
    """Gemma-4 31B (Dense) configuration."""
    dim = 5376
    intermediate_size = 21504
    n_heads = 32
    n_kv_heads = 16
    n_layers = 60
    vocab_size = 262144
    sliding_window = 1024

    return Gemma4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        sliding_window_size=sliding_window,
        enable_sliding_window=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_gemma4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=intermediate_size,
            head_dim=256,
            global_head_dim=512,
            global_kv_heads=4,
            attention_k_eq_v=True,
            rope=Gemma4RoPE.Config(
                dim=256,
                max_context_length=seq_len,
                theta=10000.0,
                scaling="none",
            ),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
            sliding_window_size=sliding_window,
            global_attn_interval=6,
        ),
    )


gemma4_configs = {
    "debugmodel": (_debugmodel, 262144),
    "e2b": (_e2b, 262144),
    "E2B": (_e2b, 262144),
    "e4b": (_e4b, 262144),
    "E4B": (_e4b, 262144),
    "12b": (_12b, 262144),
    "12B": (_12b, 262144),
    "26b_a4b": (_26b_a4b, 262144),
    "26B_A4B": (_26b_a4b, 262144),
    "31b": (_31b, 262144),
    "31B": (_31b, 262144),
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    tp_gemm_backend: TpGemmBackend = "default",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    """Register Gemma-4 model with TorchTitan.

    Args:
        flavor: Model size ("e2b", "e4b", "12b", "26b_a4b", "31b", "debugmodel")
        seq_len: Optional sequence length override
        attn_backend: Attention backend ("flex", "sdpa")
        tp_gemm_backend: Tensor parallel GEMM backend
        converters: Optional config converters for custom experimentation

    Returns:
        ModelSpec for training with TorchTitan
    """
    get_config, max_context_len = gemma4_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        tp_gemm_backend=tp_gemm_backend,
        seq_len=context_len,
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)
    return ModelSpec(
        name="gemma4",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_gemma4,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Gemma4StateDictAdapter,
    )
