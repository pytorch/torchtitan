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
    make_gqa_config,
    make_ffn_config,
    make_moe_config,
    make_router_config,
    make_token_dispatcher_config,
    TpGemmBackend,
)
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.utils import validate_converter_order

from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import Nemotron3Model, NemotronMLP, NemotronTransformerBlock, NoRoPE
from .moe import NemotronGroupedExperts
from .parallelize import parallelize_nemotron
from .state_dict_adapter import NemotronStateDictAdapter

__all__ = [
    "parallelize_nemotron",
    "Nemotron3Model",
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


# Nemotron-H hybrid pattern characters -> block type.
# M = Mamba-2 mixer, * = self-attention, - = (ungated) MLP, E = MoE.
_PATTERN_TO_BLOCK_TYPE = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}


def parse_hybrid_pattern(pattern: str) -> list[str]:
    """Expand a ``hybrid_override_pattern`` string into per-layer block types."""
    unknown = sorted(set(pattern) - set(_PATTERN_TO_BLOCK_TYPE))
    if unknown:
        raise ValueError(
            f"Unknown hybrid_override_pattern character(s): {unknown}. "
            f"Valid characters are {sorted(_PATTERN_TO_BLOCK_TYPE)}."
        )
    return [_PATTERN_TO_BLOCK_TYPE[c] for c in pattern]


def _build_nemotron_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    rope: RoPE.Config,
    num_experts: int,
    top_k_experts: int,
    mamba_num_heads: int,
    mamba_head_dim: int,
    mamba_conv_dim: int,
    n_kv_heads: int | None = None,
    fuse_qkv: bool = True,
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    hybrid_pattern: str | None = None,
    head_dim: int | None = None,
    mamba_n_groups: int = 8,
    mamba_state_dim: int = 128,
    mamba_conv_kernel: int = 4,
    mamba_chunk_size: int = 128,
    moe_intermediate_dim: int | None = None,
    shared_expert_dim: int | None = None,
    route_norm: bool = True,
    route_scale: float = 1.0,
) -> list[NemotronTransformerBlock.Config]:
    inner_attention = get_attention_config(attn_backend)

    # Routed experts have their own width (moe_intermediate_size), which differs
    # from the dense MLP width; fall back to the dense width when unspecified.
    if moe_intermediate_dim is None:
        moe_intermediate_dim = hidden_dim

    # Mamba-2 shares each B/C pair across the heads of a group and expands them
    # with repeat_interleave(num_heads // n_groups). A non-divisible pair makes
    # that expansion silently produce the wrong width (or a zero-size repeat),
    # so reject it here where the flavor is readable rather than mid-scan.
    if mamba_num_heads % mamba_n_groups != 0:
        raise ValueError(
            f"mamba_num_heads ({mamba_num_heads}) must be divisible by "
            f"mamba_n_groups ({mamba_n_groups})"
        )

    if hybrid_pattern is not None:
        block_types = parse_hybrid_pattern(hybrid_pattern)
        if len(block_types) != n_layers:
            raise ValueError(
                f"hybrid_override_pattern length {len(block_types)} != n_layers {n_layers}"
            )
    else:
        # No pattern supplied (debug/synthetic flavors): alternate Mamba and
        # attention. Real flavors always pass the published pattern.
        block_types = [
            "mamba" if layer_id % 2 == 0 else "attention" for layer_id in range(n_layers)
        ]

    layers = []
    for layer_id in range(n_layers):
        block_type = block_types[layer_id]
        is_mamba = block_type == "mamba"
        layers.append(
            NemotronTransformerBlock.Config(
                block_type=block_type,
                mamba_n_groups=mamba_n_groups,
                mamba_state_dim=mamba_state_dim,
                mamba_conv_kernel=mamba_conv_kernel,
                mamba_chunk_size=mamba_chunk_size,
                is_mamba_block=is_mamba,
                mamba_num_heads=mamba_num_heads,
                mamba_head_dim=mamba_head_dim,
                mamba_conv_dim=mamba_conv_dim,
                mamba_input_projection=Linear.Config(
                    in_features=dim,
                    out_features=mamba_conv_dim * 2,
                    param_init=_LINEAR_INIT,
                )
                if is_mamba
                else None,
                mamba_output_projection=Linear.Config(
                    in_features=mamba_conv_dim,
                    out_features=dim,
                    param_init=_LINEAR_INIT,
                )
                if is_mamba
                else None,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                # pyre-ignore[6]
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT)
                if block_type in ("mlp", "moe")
                else None,
                # pyre-ignore[6]
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    fuse_qkv=fuse_qkv,
                    rope=rope,
                    tp_gemm_backend=tp_gemm_backend,
                )
                if block_type == "attention"
                else None,
                moe=make_moe_config(
                    num_experts=num_experts,
                    router=make_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=_LINEAR_INIT,
                        top_k=top_k_experts,
                        # Nemotron-H routes on sigmoid scores, renormalizes the
                        # top-k probabilities and rescales by
                        # routed_scaling_factor. n_group/topk_group are both 1
                        # in the released config, i.e. no group-limited
                        # routing, so the group args stay None.
                        score_func="sigmoid",
                        route_norm=route_norm,
                        route_scale=route_scale,
                    ),
                    routed_experts=RoutedExperts.Config(
                        # UNGATED experts: up/down only, no w3 gate branch.
                        inner_experts=NemotronGroupedExperts.Config(
                            dim=dim,
                            hidden_dim=moe_intermediate_dim,
                            num_experts=num_experts,
                            param_init={
                                "w1_EFD": _LINEAR_INIT["weight"],
                                "w2_EDF": _depth_init(layer_id)["weight"],
                            },
                        ),
                        token_dispatcher=make_token_dispatcher_config(
                            num_experts=num_experts,
                            top_k=top_k_experts,
                            comm_backend="standard",
                            hidden_dim=dim,
                        ),
                    ),
                    # One shared expert, also ungated, with its own width.
                    shared_experts=NemotronMLP.Config(
                        dim=dim,
                        hidden_dim=shared_expert_dim,
                        up_proj=Linear.Config(
                            in_features=dim,
                            out_features=shared_expert_dim,
                            param_init=_LINEAR_INIT,
                        ),
                        down_proj=Linear.Config(
                            in_features=shared_expert_dim,
                            out_features=dim,
                            param_init=_depth_init(layer_id),
                        ),
                    )
                    if shared_expert_dim
                    else None,
                )
                if block_type == "moe"
                else None,
                feed_forward=NemotronMLP.Config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    up_proj=Linear.Config(
                        in_features=dim,
                        out_features=hidden_dim,
                        param_init=_LINEAR_INIT,
                    ),
                    down_proj=Linear.Config(
                        in_features=hidden_dim,
                        out_features=dim,
                        param_init=_depth_init(layer_id),
                    ),
                )
                if block_type == "mlp"
                else None,
            )
        )
    return layers


def _debugmodel(
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    *,
    seq_len: int,
) -> Nemotron3Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 4
    num_experts = 4
    top_k_experts = 2
    return Nemotron3Model.Config(
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
                max_context_length=seq_len,
                theta=500000,
                scaling="llama",
            ),
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            mamba_num_heads=4,
            mamba_head_dim=64,
            # Keep the debug model internally consistent: B/C are shared across
            # heads within a group, so num_heads must be divisible by n_groups.
            # The real flavors use 8 groups, which 4 heads cannot support.
            mamba_n_groups=2,
            mamba_state_dim=16,
            mamba_conv_dim=4 * 64 + 2 * 2 * 16,
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
        ),
    )


def _4b(
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    *,
    seq_len: int,
) -> Nemotron3Model.Config:
    # NVIDIA-Nemotron-3-Nano-4B-BF16 config.json (dense: no MoE layers).
    # 42 layers = 21 Mamba / 17 MLP / 4 attention.
    dim = 3136
    n_heads = 40
    n_kv_heads = 8
    n_layers = 42
    vocab_size = 131072
    hidden_dim = 12544
    hybrid_pattern = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
    num_experts = 0
    top_k_experts = 0
    mamba_num_heads = 96
    mamba_head_dim = 80
    mamba_conv_dim = mamba_num_heads * mamba_head_dim
    return Nemotron3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        mamba_num_heads=mamba_num_heads,
        mamba_head_dim=mamba_head_dim,
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
            hidden_dim=hidden_dim,
            hybrid_pattern=hybrid_pattern,
            head_dim=128,
            # Nemotron-H attention layers use NO positional embedding.
            rope=NoRoPE.Config(dim=128, max_context_length=seq_len),
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            mamba_num_heads=mamba_num_heads,
            mamba_head_dim=mamba_head_dim,
            mamba_conv_dim=mamba_conv_dim,
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
        ),
    )


def _31b(
    attn_backend: str,
    tp_gemm_backend: TpGemmBackend = "default",
    *,
    seq_len: int,
) -> Nemotron3Model.Config:
    # NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 config.json.
    # 52 layers = 23 Mamba / 23 MoE / 6 attention.
    dim = 2688
    n_heads = 32
    n_kv_heads = 2
    n_layers = 52
    vocab_size = 131072
    hidden_dim = 1856
    hybrid_pattern = (
        "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    )
    num_experts = 128
    top_k_experts = 6
    moe_intermediate_dim = 1856
    shared_expert_dim = 3712  # moe_shared_expert_intermediate_size
    route_scale = 2.5  # routed_scaling_factor
    mamba_num_heads = 64
    mamba_head_dim = 64
    mamba_conv_dim = mamba_num_heads * mamba_head_dim
    return Nemotron3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        mamba_num_heads=mamba_num_heads,
        mamba_head_dim=mamba_head_dim,
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
            hidden_dim=hidden_dim,
            hybrid_pattern=hybrid_pattern,
            head_dim=128,
            # Nemotron-H attention layers use no positional embedding.
            rope=NoRoPE.Config(dim=128, max_context_length=seq_len),
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            moe_intermediate_dim=moe_intermediate_dim,
            shared_expert_dim=shared_expert_dim,
            route_norm=True,  # norm_topk_prob
            route_scale=route_scale,
            mamba_num_heads=mamba_num_heads,
            mamba_head_dim=mamba_head_dim,
            mamba_conv_dim=mamba_conv_dim,
            attn_backend=attn_backend,
            tp_gemm_backend=tp_gemm_backend,
        ),
    )


nemotron_configs = {
    "debugmodel": (_debugmodel, 131072),
    "4B": (_4b, 1000000),
    "4b": (_4b, 1000000),
    "31B": (_31b, 1000000),
    "31b": (_31b, 1000000),
}
# NOTE: 120B and 550B flavors were removed. They were never ports of real
# NVIDIA configs -- they omitted `hybrid_pattern`, so they silently built as
# dense mamba/attention stacks with zero MoE layers (8.4B and 16.1B params
# rather than 120B/550B), and they used ComplexRoPE and vocab_size=262144
# where the Nemotron-3 family uses NoRoPE and 131072. Re-add them only from a
# published config.json, the way 4b/31b were done.


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    tp_gemm_backend: TpGemmBackend = "default",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    get_config, max_context_len = nemotron_configs[flavor]
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
        name="nemotron3",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_nemotron,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=NemotronStateDictAdapter,
    )
