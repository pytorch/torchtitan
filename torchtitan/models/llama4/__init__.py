# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import replace
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.config import DeferredCallable
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    FeedForward,
    GQAttention,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.config_expand import (
    fill_decoder_fields,
    fill_ffn_fields,
    fill_gqa_fields,
    fill_moe_fields,
)
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
from torchtitan.models.common.moe.moe import GroupedExperts
from torchtitan.models.common.param_init import depth_scaled_std, resolve_deferred
from torchtitan.protocols.model_spec import ModelSpec

from .model import compute_moe_hidden_dim, Llama4Model, Llama4TransformerBlock

from .parallelize import parallelize_llama
from .state_dict_adapter import Llama4StateDictAdapter

__all__ = [
    "Llama4Model",
    "llama4_configs",
]


def expand_layer_configs(config) -> None:
    """Expand the layer template into per-layer configs for a single model config.

    Handles iRoPE (NoPE on every N layers) and MoE interleaving.
    Mutates config in place.
    """
    dim = config.dim
    fill_decoder_fields(config)
    assert isinstance(config.layer, Llama4TransformerBlock.Config)
    if (
        config.layer.every_n_layers_nope is not None
        and config.layer.every_n_layers_nope <= 1
    ):
        raise ValueError("every_n_layers_nope must be greater than 1")
    config.layer.attention_norm.normalized_shape = dim
    config.layer.ffn_norm.normalized_shape = dim
    layers = []
    for layer_id in range(config.n_layers):
        cfg = deepcopy(config.layer)
        # iRoPE: override use_rope=False on certain layers
        if cfg.every_n_layers_nope is not None:
            if layer_id % cfg.every_n_layers_nope == 0:
                cfg = replace(cfg, attention=replace(cfg.attention, use_rope=False))
        # MoE interleaving: keep only the appropriate FFN type per layer
        moe_enabled = (layer_id + 1) % cfg.interleave_moe_layer_step == 0
        if moe_enabled:
            cfg = replace(cfg, feed_forward=None)
        else:
            cfg = replace(cfg, moe=None)
        resolve_deferred(cfg, layer_id)
        fill_gqa_fields(cfg.attention, dim)  # pyrefly: ignore [bad-argument-type]
        if cfg.feed_forward is not None:
            fill_ffn_fields(cfg.feed_forward, dim)
        if cfg.moe is not None:
            fill_moe_fields(cfg.moe, dim)
        layers.append(cfg)
    config.layers = layers


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_LINEAR_DEPTH_INIT = DeferredCallable.Config(
    fn=lambda layer_id: {  # pyrefly: ignore [bad-argument-type]
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }
)
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EXPERTS_DEPTH_INIT = DeferredCallable.Config(
    fn=lambda layer_id: {  # pyrefly: ignore [bad-argument-type]
        "w1": partial(nn.init.trunc_normal_, std=0.02),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }
)


def _output_linear_init(dim: int):
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _debugmodel() -> Llama4Model.Config:
    dim = 256
    n_heads = 16
    return Llama4Model.Config(
        dim=dim,
        n_layers=6,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama4TransformerBlock.Config(
            every_n_layers_nope=4,
            fixed_attn_block_size=256,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="complex",
            ),
            moe=MoE.Config(
                hidden_dim=compute_moe_hidden_dim(dim),
                router=TokenChoiceTopKRouter.Config(
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
                shared_experts=FeedForward.Config(
                    hidden_dim=compute_moe_hidden_dim(dim),
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
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
    moe_hidden_dim = compute_moe_hidden_dim(
        dim,
        multiple_of=2048,
        ffn_dim_multiplier=1.2,
        top_k=1,
        num_shared_experts=1,
    )
    return Llama4Model.Config(
        dim=dim,
        n_layers=48,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama4TransformerBlock.Config(
            every_n_layers_nope=4,
            interleave_moe_layer_step=1,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                num_experts=16,
                hidden_dim=moe_hidden_dim,
                router=TokenChoiceTopKRouter.Config(
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
                shared_experts=FeedForward.Config(
                    hidden_dim=moe_hidden_dim,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    dim, multiple_of=2048, ffn_dim_multiplier=1.2
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="complex",
            ),
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
    moe_hidden_dim = compute_moe_hidden_dim(
        dim,
        multiple_of=2048,
        ffn_dim_multiplier=1.2,
        top_k=1,
        num_shared_experts=1,
    )
    return Llama4Model.Config(
        dim=dim,
        n_layers=48,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(dim)),
        layer=Llama4TransformerBlock.Config(
            every_n_layers_nope=4,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                num_experts=128,
                hidden_dim=moe_hidden_dim,
                router=TokenChoiceTopKRouter.Config(
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(param_init=_EXPERTS_DEPTH_INIT),
                shared_experts=FeedForward.Config(
                    hidden_dim=moe_hidden_dim,
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    dim, multiple_of=2048, ffn_dim_multiplier=1.2
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                inner_attention=FlexAttention.Config(),
                mask_type="block_causal",
                rope_backend="complex",
            ),
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
    expand_layer_configs(config)
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
