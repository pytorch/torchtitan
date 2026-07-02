# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    ComplexRoPE,
    CosSinRoPE,
    LayerNorm,
    Linear,
    RMSNorm,
)
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan.models.deepseek_v3.__init__ import (
    _build_dsv3_layers_impl,
    _EMBEDDING_INIT,
    _LINEAR_INIT,
    _NORM_INIT,
    _output_linear_init,
    _make_dsv3_attn_config,
)

from .model import (
    Attention,
    DeepSeekV32Model,
    DSAFlexAttention,
    Indexer,
)
from .state_dict_adapter import DeepSeekV32StateDictAdapter

# V3.2 parallelization delegates entirely to V3.
from torchtitan.models.deepseek_v3.parallelize import (
    parallelize_deepseekv3 as parallelize_deepseekv32,
)

__all__ = ["parallelize_deepseekv32", "DeepSeekV32Model", "deepseekv32_configs"]


def _make_dsv32_attn_config(
    layer_id: int, *, index_n_heads: int, index_head_dim: int, index_topk: int,
    **kwargs,
) -> Attention.Config:
    """V3 MLA + Indexer + DSAFlexAttention inner."""
    c = _make_dsv3_attn_config(layer_id=layer_id, **kwargs)
    rhd = c.qk_rope_head_dim
    indexer_cfg = Indexer.Config(
        dim=c.dim, q_lora_rank=c.q_lora_rank,
        index_n_heads=index_n_heads, index_head_dim=index_head_dim,
        rope_head_dim=rhd, index_topk=index_topk,
        wq_b=Linear.Config(
            in_features=c.q_lora_rank,
            out_features=index_n_heads * index_head_dim,
            param_init=_LINEAR_INIT,
        ),
        wk=Linear.Config(
            in_features=c.dim, out_features=index_head_dim,
            param_init=_LINEAR_INIT,
        ),
        k_norm=LayerNorm.Config(normalized_shape=index_head_dim),
        weights_proj=Linear.Config(
            in_features=c.dim, out_features=index_n_heads,
            param_init={"weight": partial(nn.init.normal_, std=1.0)},
        ),
        # Reference applies non-interleaved (rotate-half) rope with the
        # same yarn-scaled freqs as MLA.
        rope=CosSinRoPE.Config(
            dim=rhd, max_seq_len=c.rope.max_seq_len, theta=c.rope.theta,
            scaling="yarn", rope_factor=c.rope.rope_factor,
            beta_fast=c.rope.beta_fast, beta_slow=c.rope.beta_slow,
            original_seq_len=c.rope.original_seq_len,
        ),
    )
    return Attention.Config(
        dim=c.dim, n_heads=c.n_heads,
        q_lora_rank=c.q_lora_rank, kv_lora_rank=c.kv_lora_rank,
        qk_nope_head_dim=c.qk_nope_head_dim, qk_rope_head_dim=c.qk_rope_head_dim,
        v_head_dim=c.v_head_dim, mscale=c.mscale,
        wq=c.wq, wq_a=c.wq_a, wq_b=c.wq_b, q_norm=c.q_norm,
        wkv_a=c.wkv_a, kv_norm=c.kv_norm, wkv_b=c.wkv_b, wo=c.wo,
        inner_attention=DSAFlexAttention.Config(index_topk=index_topk),
        rope=dataclasses.replace(c.rope),
        indexer=indexer_cfg,
    )


def _build_dsv32_layers(**kwargs):
    """V32-specific thin wrapper around ``_build_dsv3_layers_impl``."""
    return _build_dsv3_layers_impl(
        attn_fn=_make_dsv32_attn_config,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Model flavors
# ---------------------------------------------------------------------------

def _debugmodel(
    attn_backend: str, moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
) -> DeepSeekV32Model.Config:
    dim = 256
    layers = _build_dsv32_layers(
        n_layers=6, n_dense_layers=1, dim=dim, n_heads=16,
        q_lora_rank=64, kv_lora_rank=128, qk_nope_head_dim=32,
        qk_rope_head_dim=16, v_head_dim=32, mscale=0.70,
        dense_hidden_dim=512, moe_hidden_dim=128, num_experts=8,
        num_shared_experts=2, router_top_k=3, router_score_func="softmax",
        attn_backend=attn_backend, moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=ComplexRoPE.Config(dim=16, max_seq_len=16384, theta=10000.0,
                                scaling="yarn", rope_factor=40.0,
                                beta_fast=32.0, beta_slow=1.0, original_seq_len=4096),
        index_n_heads=4, index_head_dim=32, index_topk=16,
    )
    from torchtitan.models.common import Embedding
    return DeepSeekV32Model.Config(
        vocab_size=2048, dim=dim,
        tok_embeddings=Embedding.Config(num_embeddings=2048, embedding_dim=dim,
                                        param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(in_features=dim, out_features=2048,
                              param_init=_output_linear_init(dim)),
        layers=layers,
    )


deepseekv32_configs = {
    "debugmodel": _debugmodel,
}


def model_registry(
    flavor: str, attn_backend: str = "flex",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
) -> ModelSpec:
    config = deepseekv32_configs[flavor](
        attn_backend=attn_backend, moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    return ModelSpec(
        name="deepseek_v3_2", flavor=flavor, model=config,
        parallelize_fn=parallelize_deepseekv32, pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV32StateDictAdapter,
    )
