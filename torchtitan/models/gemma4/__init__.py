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
from typing import Any

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    VarlenAttention,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_token_dispatcher_config,
    TpGemmBackend,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan.models.common.moe import RoutedExperts, TokenChoiceTopKRouter
from .moe import Gemma4GroupedExperts, Gemma4MoE
from .ple import Gemma4LayerPLE, Gemma4PerLayerEmbedding
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
    "Gemma4MoE",
    "Gemma4GroupedExperts",
    "Gemma4LayerPLE",
    "Gemma4PerLayerEmbedding",
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
        w1=Linear.Config(in_features=dim, out_features=hidden_dim, param_init=w1_param_init),
        w2=Linear.Config(in_features=hidden_dim, out_features=dim, param_init=w2w3_param_init),
        w3=Linear.Config(in_features=dim, out_features=hidden_dim, param_init=w2w3_param_init),
    )


def _make_wv_config(
    *,
    use_global_attn: bool,
    attention_k_eq_v: bool,
    is_kv_shared: bool,
    dim: int,
    actual_kv_heads: int,
    actual_head_dim: int,
) -> "Linear.Config | None":
    """Compute the v_proj config for a layer.

    Rules:
    - Shared KV layers: no wv (they borrow K/V from the donor layer).
    - Non-shared global layers: always have wv (edge: attention_k_eq_v=False,
      dense 12b/31b: attention_k_eq_v=True so wv=None and K is reused as V).
    - Non-shared sliding layers: always have wv.
    """
    if is_kv_shared:
        return None
    if use_global_attn and attention_k_eq_v:
        # Dense models (12b, 31b): global layers use K as V, no separate v_proj.
        return None
    return Linear.Config(
        in_features=dim,
        out_features=actual_kv_heads * actual_head_dim,
        param_init=_LINEAR_INIT,
    )


def _build_gemma4_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int | list[int],
    rope: RoPE.Config,
    n_kv_heads: int | None = None,
    head_dim: int = 256,
    global_head_dim: int = 512,
    global_kv_heads: int | None = None,
    attention_k_eq_v: bool = False,
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    sliding_window_size: int = 1024,
    global_attn_interval: int = 6,
    num_experts: int | None = None,
    top_k: int | None = None,
    shared_experts_dim: int | None = None,
    ple_dim: int | None = None,
    num_kv_shared_layers: int = 0,
) -> list[Gemma4TransformerBlock.Config]:
    """Build per-layer Gemma4TransformerBlock configs with hybrid attention routing."""
    inner_attention = get_attention_config(attn_backend)
    norm_cfg = RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT)

    first_kv_shared_layer_idx = n_layers - num_kv_shared_layers
    layer_types = [
        "global" if (i + 1) % global_attn_interval == 0 else "sliding"
        for i in range(n_layers)
    ]
    prev_unshared_layers = layer_types[:first_kv_shared_layer_idx]

    layers: list[Gemma4TransformerBlock.Config] = []
    for layer_id in range(n_layers):
        layer_hidden_dim = (
            hidden_dim[layer_id] if isinstance(hidden_dim, list) else hidden_dim
        )
        layer_type = layer_types[layer_id]
        use_global_attn = (layer_type == "global")
        is_kv_shared = (layer_id >= first_kv_shared_layer_idx)
        store_full_length_kv = (
            num_kv_shared_layers > 0
            and not is_kv_shared
            and layer_id == len(prev_unshared_layers) - 1 - prev_unshared_layers[::-1].index(layer_type)
        )

        actual_head_dim = (
            global_head_dim if use_global_attn and global_head_dim is not None else head_dim
        )
        actual_kv_heads = (
            global_kv_heads if use_global_attn and global_kv_heads is not None else (n_kv_heads or n_heads)
        )

        # Sliding layers use FlexAttention; global layers (head_dim=512) dispatch to SDPA
        if use_global_attn and actual_head_dim > 256:
            layer_inner_attn = Gemma4GlobalSDPA.Config()
        else:
            layer_inner_attn = copy.deepcopy(inner_attention)
            if not use_global_attn and isinstance(layer_inner_attn, VarlenAttention.Config):
                layer_inner_attn.window_size = (sliding_window_size, 0)

        wv_cfg = _make_wv_config(
            use_global_attn=use_global_attn,
            attention_k_eq_v=attention_k_eq_v,
            is_kv_shared=is_kv_shared,
            dim=dim,
            actual_kv_heads=actual_kv_heads,
            actual_head_dim=actual_head_dim,
        )
        if is_kv_shared:
            qkv_linear = Gemma4QKVLinear.Config(
                head_dim=actual_head_dim,
                wq=Linear.Config(in_features=dim, out_features=n_heads * actual_head_dim, param_init=_LINEAR_INIT),
                wk=None,
                wv=None,
            )
            layer_v_norm = None
        else:
            qkv_linear = Gemma4QKVLinear.Config(
                head_dim=actual_head_dim,
                wq=Linear.Config(in_features=dim, out_features=n_heads * actual_head_dim, param_init=_LINEAR_INIT),
                wk=Linear.Config(in_features=dim, out_features=actual_kv_heads * actual_head_dim, param_init=_LINEAR_INIT),
                wv=wv_cfg,
            )
            layer_v_norm = RMSNorm.Config(normalized_shape=actual_head_dim, eps=1e-6, elementwise_affine=False)

        layer_rope = Gemma4RoPE.Config(
            dim=actual_head_dim,
            max_context_length=rope.max_context_length,
            theta=1000000.0 if use_global_attn else 10000.0,
            partial_rotary_factor=0.25 if use_global_attn else 1.0,
            scaling="none",
        )
        attention_cfg = Gemma4Attention.Config(
            n_heads=n_heads,
            dim=dim,
            qkv_linear=qkv_linear,
            wo=Linear.Config(in_features=n_heads * actual_head_dim, out_features=dim, param_init=_depth_init(layer_id)),
            qk_norm=RMSNorm.Config(normalized_shape=actual_head_dim, eps=1e-6, param_init=_NORM_INIT),
            n_kv_heads=actual_kv_heads,
            head_dim=actual_head_dim,
            inner_attention=layer_inner_attn,
            rope=layer_rope,
            attn_scale=1.0,
            v_norm=layer_v_norm,
            is_kv_shared_layer=is_kv_shared,
            store_full_length_kv=store_full_length_kv,
            layer_type=layer_type,
        )

        if ple_dim is not None:
            ple_cfg = Gemma4LayerPLE.Config(
                dim=dim,
                ple_dim=ple_dim,
                per_layer_input_gate=Linear.Config(
                    in_features=dim,
                    out_features=ple_dim,
                    bias=False,
                    param_init=_LINEAR_INIT,
                ),
                per_layer_projection=Linear.Config(
                    in_features=ple_dim,
                    out_features=dim,
                    bias=False,
                    param_init=_LINEAR_INIT,
                ),
                post_per_layer_input_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=1e-6,
                    param_init=_NORM_INIT,
                ),
            )
        else:
            ple_cfg = None

        if num_experts is not None and num_experts > 1:
            token_dispatcher = make_token_dispatcher_config(
                num_experts=num_experts,
                top_k=top_k or 4,
                comm_backend="standard",
                hidden_dim=dim,
            )
            if shared_experts_dim is not None:
                shared_experts_cfg = _make_gemma4_ffn_config(
                    dim=dim,
                    hidden_dim=shared_experts_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                    tp_gemm_backend=tp_gemm_backend,
                )
            else:
                shared_experts_cfg = None

            moe_cfg = Gemma4MoE.Config(
                num_experts=num_experts,
                routed_experts=RoutedExperts.Config(
                    inner_experts=Gemma4GroupedExperts.Config(
                        dim=dim,
                        hidden_dim=layer_hidden_dim,
                        num_experts=num_experts,
                    ),
                    token_dispatcher=token_dispatcher,
                ),
                router=TokenChoiceTopKRouter.Config(
                    num_experts=num_experts,
                    top_k=top_k or 4,
                    score_func="softmax",
                    route_norm=True,
                    gate=Linear.Config(
                        in_features=dim,
                        out_features=num_experts,
                        bias=False,
                        param_init=_LINEAR_INIT,
                    ),
                ),
                shared_experts=shared_experts_cfg,
            )
            ffn_cfg = None
        else:
            moe_cfg = None
            ffn_cfg = _make_gemma4_ffn_config(
                dim=dim,
                hidden_dim=layer_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
                tp_gemm_backend=tp_gemm_backend,
            )

        layers.append(
            Gemma4TransformerBlock.Config(
                use_global_attention=use_global_attn,
                attention_norm=norm_cfg,
                post_attention_norm=norm_cfg,
                ffn_norm=norm_cfg,
                post_ffn_norm=norm_cfg,
                attention=attention_cfg,
                feed_forward=ffn_cfg,
                moe=moe_cfg,
                ple=ple_cfg,
            )
        )
    return layers


def _create_gemma4_config(
    *,
    dim: int,
    n_heads: int,
    n_layers: int,
    hidden_dim: int | list[int],
    n_kv_heads: int | None = None,
    head_dim: int = 256,
    global_head_dim: int | None = 512,
    global_kv_heads: int | None = None,
    attention_k_eq_v: bool = False,
    vocab_size: int = 262144,
    sliding_window_size: int = 1024,
    enable_sliding_window: bool = True,
    global_attn_interval: int = 6,
    rope_dim: int = 256,
    num_experts: int | None = None,
    top_k: int | None = None,
    shared_experts_dim: int | None = None,
    ple_dim: int | None = None,
    num_kv_shared_layers: int = 0,
    final_logit_softcapping: float | None = None,
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    seq_len: int,
) -> Gemma4Model.Config:
    ple_model_cfg = (
        Gemma4PerLayerEmbedding.Config(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=n_layers,
            ple_dim=ple_dim,
            tok_embeddings_per_layer=Embedding.Config(
                num_embeddings=vocab_size,
                embedding_dim=n_layers * ple_dim,
                param_init=_EMBEDDING_INIT,
            ),
            per_layer_model_projection=Linear.Config(
                in_features=dim,
                out_features=n_layers * ple_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            ),
            per_layer_projection_norm=RMSNorm.Config(
                normalized_shape=ple_dim,
                eps=1e-6,
                param_init=_NORM_INIT,
            ),
        )
        if ple_dim is not None
        else None
    )

    return Gemma4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        sliding_window_size=sliding_window_size,
        enable_sliding_window=enable_sliding_window,
        final_logit_softcapping=final_logit_softcapping,
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(normalized_shape=dim, eps=1e-6, param_init=_NORM_INIT),
        lm_head=Linear.Config(in_features=dim, out_features=vocab_size, param_init=_output_linear_init(dim)),
        ple=ple_model_cfg,
        layers=_build_gemma4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            global_head_dim=global_head_dim,
            global_kv_heads=global_kv_heads,
            attention_k_eq_v=attention_k_eq_v,
            rope=Gemma4RoPE.Config(dim=rope_dim, max_context_length=seq_len, theta=10000.0, scaling="none"),
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
            sliding_window_size=sliding_window_size,
            global_attn_interval=global_attn_interval,
            num_experts=num_experts,
            top_k=top_k,
            shared_experts_dim=shared_experts_dim,
            ple_dim=ple_dim,
            num_kv_shared_layers=num_kv_shared_layers,
        ),
    )


_FLAVOR_SPECS: dict[str, dict[str, Any]] = {
    "debugmodel": dict(
        dim=256, n_heads=16, n_layers=6, hidden_dim=1024,
        head_dim=16, global_head_dim=None, rope_dim=16,
        vocab_size=2048, enable_sliding_window=False,
    ),
    # Edge models: attention_k_eq_v=False because global layers have distinct v_proj weights.
    # KV-shared layers (tail of the stack) borrow K/V from the last donor layer of
    # each attention type, so they have no wk/wv of their own.
    "e2b": dict(
        dim=1536, n_heads=8, n_kv_heads=1, n_layers=35,
        hidden_dim=[6144] * 15 + [12288] * 20,
        global_kv_heads=1,
        attention_k_eq_v=False,  # edge: global layers have real v_proj
        sliding_window_size=512, global_attn_interval=5,
        ple_dim=256, num_kv_shared_layers=20,
        final_logit_softcapping=30.0,
    ),
    "e4b": dict(
        dim=2560, n_heads=8, n_kv_heads=2, n_layers=42, hidden_dim=10240,
        global_kv_heads=2,
        attention_k_eq_v=False,  # edge: global layers have real v_proj
        sliding_window_size=512, global_attn_interval=6,
        ple_dim=256, num_kv_shared_layers=18,
        final_logit_softcapping=30.0,
    ),
    # Dense models: attention_k_eq_v=True — global layers reuse K as V (no v_proj).
    "12b": dict(
        dim=3840, n_heads=16, n_kv_heads=8, n_layers=48, hidden_dim=15360,
        global_kv_heads=1, attention_k_eq_v=True, sliding_window_size=1024, global_attn_interval=6,
    ),
    "26b_a4b": dict(
        dim=2816, n_heads=16, n_kv_heads=8, n_layers=30, hidden_dim=704,
        num_experts=128, top_k=8, shared_experts_dim=2112,
        global_kv_heads=2, attention_k_eq_v=True, sliding_window_size=1024, global_attn_interval=6,
    ),
    "31b": dict(
        dim=5376, n_heads=32, n_kv_heads=16, n_layers=60, hidden_dim=21504,
        global_kv_heads=4, attention_k_eq_v=True, sliding_window_size=1024, global_attn_interval=6,
    ),
}


def _get_flavor_builder(flavor: str) -> Callable:
    params = _FLAVOR_SPECS[flavor.lower()]

    def builder(
        attn_backend: str,
        tp_gemm_backend: TpGemmBackend = "default",
        *,
        seq_len: int,
    ) -> Gemma4Model.Config:
        return _create_gemma4_config(
            **params,
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
            seq_len=seq_len,
        )

    return builder


gemma4_configs = {
    key: (_get_flavor_builder(key), 262144)
    for key in [
        "debugmodel", "e2b", "E2B", "e4b", "E4B",
        "12b", "12B", "26b_a4b", "26B_A4B", "31b", "31B",
    ]
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    tp_gemm_backend: TpGemmBackend = "default",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    """Register Gemma-4 model with TorchTitan."""
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
