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
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.model_spec import ModelSpec

from .model import compute_moe_hidden_dim, Llama4Model, Llama4TransformerBlock
from .parallelize import parallelize_llama
from .state_dict_adapter import Llama4StateDictAdapter

__all__ = [
    "Llama4Model",
    "llama4_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int):
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


def _build_llama4_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    n_kv_heads: int | None = None,
    every_n_layers_nope: int = 4,
    interleave_moe_layer_step: int = 1,
    fixed_attn_block_size: int = 8192,
    mask_type: str = "block_causal",
    shared_experts_hidden_dim: int | None = None,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for a Llama4 model.

    Handles iRoPE (NoPE on every N layers) and MoE interleaving. For each
    layer, depth-scaled inits are computed using the layer index.
    """
    if every_n_layers_nope <= 1:
        raise ValueError("every_n_layers_nope must be greater than 1")

    _shared_experts_hidden_dim = (
        shared_experts_hidden_dim
        if shared_experts_hidden_dim is not None
        else moe_hidden_dim
    )

    layers = []
    for layer_id in range(n_layers):
        use_rope = (layer_id % every_n_layers_nope) != 0
        moe_enabled = (layer_id + 1) % interleave_moe_layer_step == 0

        attn = make_gqa_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            wqkv_param_init=_LINEAR_INIT,
            wo_param_init=_depth_init(layer_id),
            use_rope=use_rope,
            inner_attention=FlexAttention.Config(),
            mask_type=mask_type,
            rope_backend="complex",
        )

        if moe_enabled:
            router = make_router_config(
                dim=dim,
                num_experts=num_experts,
                gate_param_init=_depth_init(layer_id),
            )
            experts = make_experts_config(
                dim=dim,
                hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                param_init=_depth_experts_init(layer_id),
            )
            shared_experts = make_ffn_config(
                dim=dim,
                hidden_dim=_shared_experts_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = make_moe_config(
                num_experts=num_experts,
                router=router,
                experts=experts,
                shared_experts=shared_experts,
            )
            ffn_cfg = None
        else:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = None

        layer = Llama4TransformerBlock.Config(
            attention=attn,
            attention_norm=RMSNorm.Config(
                normalized_shape=dim,
                param_init=_NORM_INIT,
            ),
            ffn_norm=RMSNorm.Config(
                normalized_shape=dim,
                param_init=_NORM_INIT,
            ),
            fixed_attn_block_size=fixed_attn_block_size,
            moe=moe_cfg,
            feed_forward=ffn_cfg,
        )
        layers.append(layer)

    return layers


def _debugmodel() -> Llama4Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Llama4Model.Config(
        dim=dim,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=2048,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_llama4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            moe_hidden_dim=compute_moe_hidden_dim(dim),
            num_experts=8,
            every_n_layers_nope=4,
            interleave_moe_layer_step=2,
            fixed_attn_block_size=256,
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=1048576,
            theta=500000,
            backend="complex",
            scaling="llama",
            scaling_factor=16.0,
            high_freq_factor=1.0,
        ),
    )


def _17bx16e() -> Llama4Model.Config:
    dim = 5120
    n_heads = 40
    n_kv_heads = 8
    n_layers = 48
    vocab_size = 202048
    moe_hidden_dim = compute_moe_hidden_dim(
        dim,
        multiple_of=2048,
        ffn_dim_multiplier=1.2,
        top_k=1,
        num_shared_experts=1,
    )
    return Llama4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_llama4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=2048, ffn_dim_multiplier=1.2
            ),
            moe_hidden_dim=moe_hidden_dim,
            num_experts=16,
            every_n_layers_nope=4,
            interleave_moe_layer_step=1,
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=10485760,
            theta=500000,
            backend="complex",
            scaling="llama",
            scaling_factor=16.0,
            high_freq_factor=1.0,
        ),
    )


def _17bx128e() -> Llama4Model.Config:
    dim = 5120
    n_heads = 40
    n_kv_heads = 8
    n_layers = 48
    vocab_size = 202048
    moe_hidden_dim = compute_moe_hidden_dim(
        dim,
        multiple_of=2048,
        ffn_dim_multiplier=1.2,
        top_k=1,
        num_shared_experts=1,
    )
    return Llama4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_llama4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=2048, ffn_dim_multiplier=1.2
            ),
            moe_hidden_dim=moe_hidden_dim,
            num_experts=128,
            every_n_layers_nope=4,
            interleave_moe_layer_step=1,
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=1048576,
            theta=500000,
            backend="complex",
            scaling="none",
        ),
    )


llama4_configs = {
    "debugmodel": _debugmodel,
    "17bx16e": _17bx16e,
    "17bx128e": _17bx128e,
}


def model_registry(flavor: str) -> ModelSpec:
    config = llama4_configs[flavor]()
    return ModelSpec(
        name="llama4",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Llama4StateDictAdapter,
    )
