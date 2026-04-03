# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from typing import Literal

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE, TransformerBlock
from torchtitan.models.common.attention import FlexAttention, ScaledDotProductAttention
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.model_spec import ModelSpec

from .model import Attention, DeepSeekV3Model, DeepSeekV3TransformerBlock

from .parallelize import parallelize_deepseekv3
from .state_dict_adapter import DeepSeekV3StateDictAdapter

__all__ = [
    "parallelize_deepseekv3",
    "DeepSeekV3Model",
    "deepseekv3_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


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


def _make_dsv3_attn_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float = 1.0,
    inner_attention=None,
    mask_type: str = "causal",
) -> Attention.Config:
    """Build a fully-specified DeepSeek V3 MLA Attention.Config.

    All Linear and RMSNorm sub-configs have their dimensional fields set.
    When q_lora_rank == 0, sets wq (not wq_a/wq_b).
    When q_lora_rank > 0, sets wq_a/wq_b (not wq).
    """
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    if q_lora_rank == 0:
        wq = Linear.Config(
            in_features=dim,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        wq_a = None
        wq_b = None
        # q_norm is unused when q_lora_rank == 0 (never built), but the field is
        # required on Attention.Config so we supply a placeholder.
        q_norm = RMSNorm.Config(normalized_shape=1, param_init=_NORM_INIT)
    else:
        wq = None
        wq_a = Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=_LINEAR_INIT,
        )
        wq_b = Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        q_norm = RMSNorm.Config(normalized_shape=q_lora_rank, param_init=_NORM_INIT)

    return Attention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=mscale,
        wq=wq,
        wq_a=wq_a,
        wq_b=wq_b,
        q_norm=q_norm,
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(normalized_shape=kv_lora_rank, param_init=_NORM_INIT),
        wkv_b=Linear.Config(
            in_features=kv_lora_rank,
            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
            param_init=_LINEAR_INIT,
        ),
        wo=Linear.Config(
            in_features=n_heads * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        inner_attention=(
            inner_attention
            if inner_attention is not None
            else ScaledDotProductAttention.Config()
        ),
        mask_type=mask_type,
    )


def _build_dsv3_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: Literal["sigmoid", "softmax"],
    router_num_expert_groups: int | None = None,
    router_num_limited_groups: int | None = None,
    router_route_scale: float = 1.0,
    router_route_norm: bool = False,
    score_before_experts: bool = False,
    inner_attention=None,
    mask_type: str = "causal",
) -> list[TransformerBlock.Config]:
    """Build the list of per-layer TransformerBlock configs.

    Layers with layer_id < n_dense_layers get a dense FeedForward and no MoE.
    Layers with layer_id >= n_dense_layers get a MoE and no FeedForward.

    Router and expert inits are constructed per-layer so depth-scaled
    initializers are correct for each layer's position.
    """
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = _make_dsv3_attn_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            inner_attention=inner_attention,
            mask_type=mask_type,
        )

        if layer_id < n_dense_layers:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = make_moe_config(
                num_experts=num_experts,
                score_before_experts=score_before_experts,
                router=make_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=_depth_init(layer_id),
                    top_k=router_top_k,
                    score_func=router_score_func,
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    route_norm=router_route_norm,
                ),
                experts=make_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    param_init=_depth_experts_init(layer_id),
                ),
                shared_experts=make_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )

        layers.append(
            DeepSeekV3TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                feed_forward=ffn_cfg,
                moe=moe_cfg,
            )
        )
    return layers


def _debugmodel() -> DeepSeekV3Model.Config:
    dim = 256
    n_layers = 6
    vocab_size = 2048
    n_heads = 16
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    rope_dim = 64
    num_experts = 8
    n_dense_layers = 1

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
    )
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _debugmodel_flex_attn() -> DeepSeekV3Model.Config:
    dim = 256
    n_layers = 6
    vocab_size = 2048
    n_heads = 16
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    rope_dim = 64
    num_experts = 8
    n_dense_layers = 1

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
    )
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _16b() -> DeepSeekV3Model.Config:
    dim = 2048
    n_layers = 27
    vocab_size = 102400
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    num_experts = 64
    n_dense_layers = 1

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
    )
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _236b() -> DeepSeekV3Model.Config:
    dim = 5120
    n_layers = 60
    vocab_size = 102400
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 1536
    num_shared_experts = 2
    dense_hidden_dim = 12288
    rope_dim = 64
    num_experts = 160
    n_dense_layers = 1

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        router_num_expert_groups=8,
        router_num_limited_groups=3,
        router_route_scale=16.0,
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
    )
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _671b() -> DeepSeekV3Model.Config:
    dim = 7168
    n_layers = 61
    vocab_size = 129280
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 2048
    num_shared_experts = 1
    dense_hidden_dim = 18432
    rope_dim = 64
    num_experts = 256
    n_dense_layers = 3

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=8,
        router_score_func="sigmoid",
        router_num_expert_groups=8,
        router_num_limited_groups=4,
        router_route_scale=2.5,
        router_route_norm=True,
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
    )
    return DeepSeekV3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


deepseekv3_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_flex_attn": _debugmodel_flex_attn,
    "16B": _16b,
    "236B": _236b,
    "671B": _671b,
}


def model_registry(flavor: str) -> ModelSpec:
    config = deepseekv3_configs[flavor]()
    return ModelSpec(
        name="deepseek_v3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
