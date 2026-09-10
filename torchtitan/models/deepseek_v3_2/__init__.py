# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.config import derive
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    BatchedLinear,
    ComplexRoPE,
    CosSinRoPE,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
)
from torchtitan.models.deepseek_v3 import (
    _LINEAR_INIT,
    _NORM_INIT,
    _EMBEDDING_INIT,
    _output_linear_init,
    _build_dsv3_layers,
)
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    Attention,
    DeepSeekV32Model,
    Indexer,
    SparseInnerAttention,
)
from .parallelize import parallelize_deepseek_v3_2
from .state_dict_adapter import DeepSeekV32StateDictAdapter

__all__ = [
    "parallelize_deepseek_v3_2",
    "deepseekv3_2_configs",
]


def _build_dsv3_2_layers(
    *,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    rope,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    **kwargs,
):
    layers = _build_dsv3_layers(
        dim=dim, n_heads=n_heads,
        q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim, qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim, rope=rope,
        **kwargs,
    )

    for layer_cfg in layers:
        layer_cfg.attention = derive(layer_cfg.attention, Attention.Config,
            inner_attention=SparseInnerAttention.Config(index_topk=index_topk),
            indexer=Indexer.Config(
                dim=dim, q_lora_rank=q_lora_rank,
                index_n_heads=index_n_heads, index_head_dim=index_head_dim,
                rope_head_dim=qk_rope_head_dim, index_topk=index_topk,
                wq_b=Linear.Config(
                    in_features=q_lora_rank,
                    out_features=index_n_heads * index_head_dim,
                    param_init=_LINEAR_INIT,
                ),
                wk=Linear.Config(
                    in_features=dim, out_features=index_head_dim,
                    param_init=_LINEAR_INIT,
                ),
                k_norm=LayerNorm.Config(normalized_shape=index_head_dim),
                weights_proj=Linear.Config(
                    in_features=dim, out_features=index_n_heads,
                    param_init={"weight": partial(nn.init.normal_, std=1.0)},
                ),
                rope=CosSinRoPE.Config(
                    dim=qk_rope_head_dim, max_seq_len=rope.max_seq_len,
                    theta=rope.theta, scaling=rope.scaling,
                    rope_factor=rope.rope_factor,
                    beta_fast=rope.beta_fast, beta_slow=rope.beta_slow,
                    original_seq_len=rope.original_seq_len,
                ),
            ),
            w_uk=BatchedLinear.Config(
                n_heads=n_heads,
                in_features=qk_nope_head_dim,
                out_features=kv_lora_rank,
                param_init=_LINEAR_INIT,
            ),
            w_uv=BatchedLinear.Config(
                n_heads=n_heads,
                in_features=kv_lora_rank,
                out_features=v_head_dim,
                param_init=_LINEAR_INIT,
            ),
        )
    return layers


def _debugmodel_v3_2(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> DeepSeekV32Model.Config:
    dim = 256
    n_layers = 6
    vocab_size = 2048
    n_heads = 16
    q_lora_rank = 64
    kv_lora_rank = 128
    qk_nope_head_dim = 64
    qk_rope_head_dim = 64
    v_head_dim = 64
    index_n_heads = 4
    index_head_dim = 128
    index_topk = 32
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    num_experts = 8
    n_dense_layers = 1

    layers = _build_dsv3_2_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        aux_loss_coeff=1e-4,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=ComplexRoPE.Config(
            dim=qk_rope_head_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
    )
    return DeepSeekV32Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


deepseekv3_2_configs = {
    "debugmodel": _debugmodel_v3_2,
}


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = deepseekv3_2_configs[flavor](
        attn_backend="flex",
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)
    return ModelSpec(
        name="deepseek_v3_2",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseek_v3_2,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV32StateDictAdapter,
    )
