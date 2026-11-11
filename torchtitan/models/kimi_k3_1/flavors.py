# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model flavors."""

from functools import partial

import spmd_types as spmd
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    mla_attention_local_spmd_sharding_config,
    replicated_state_sharding_config,
)
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.kimi_k3.attention import KimiMLAAttention
from torchtitan.models.kimi_k3.flavors import (
    kimi_k3_config,
    linear,
    mla_attention_config,
    vision_encoder_config,
)
from torchtitan.models.kimi_k3.model import KimiK3Model

from .dsa import (
    AbsorbedKVProjection,
    KimiDSAAttention,
    KimiDSAIndexer,
    KimiSparseAttention,
)
from .model import KimiK31Model

__all__ = ["build_model_config", "KimiK31Model", "MODEL_FLAVORS"]


def _dsa_config(
    *,
    dim: int,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    enable_sp: bool,
    index_topk: int,
    num_index_heads: int,
    index_head_dim: int,
    indexer_loss_coeff: float,
) -> KimiMLAAttention.Config:
    return mla_attention_config(
        config_cls=partial(
            KimiDSAAttention.Config,
            indexer=KimiDSAIndexer.Config(
                num_index_heads=num_index_heads,
                index_head_dim=index_head_dim,
                wq_b=linear(
                    q_lora_rank,
                    num_index_heads * index_head_dim,
                    sharding_config=replicated_state_sharding_config("weight"),
                ),
                wk=linear(
                    dim,
                    index_head_dim,
                    sharding_config=replicated_state_sharding_config("weight"),
                ),
                k_norm=LayerNorm.Config(
                    normalized_shape=index_head_dim,
                    sharding_config=replicated_state_sharding_config("weight", "bias"),
                ),
                weights_proj=linear(
                    dim,
                    num_index_heads,
                    sharding_config=replicated_state_sharding_config("weight"),
                ),
            ),
        ),
        inner_attention=KimiSparseAttention.Config(
            wkv_b=AbsorbedKVProjection.Config(
                kv_lora_rank=kv_lora_rank,
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                v_head_dim=v_head_dim,
                param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
                sharding_config=ShardingConfig(
                    state_shardings={"weight": dense_param_placement(tp=spmd.S(0))}
                ),
            ),
            index_topk=index_topk,
            aux_loss=(
                SparseIndexerLoss.Config(
                    coeff=indexer_loss_coeff,
                    reduce_mesh="loss",
                    softmax_scale=(qk_nope_head_dim + qk_rope_head_dim) ** -0.5,
                    num_heads=num_heads,
                    compress_ratio=1,
                )
                if indexer_loss_coeff > 0
                else None
            ),
            sharding_config=mla_attention_local_spmd_sharding_config(),
        ),
        dim=dim,
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        enable_sp=enable_sp,
    )


def _debugmodel(
    *, seq_len: int, indexer_loss_coeff: float, parallelism: ParallelismConfig
) -> KimiK3Model.Config:
    dim = 256
    return kimi_k3_config(
        config_cls=KimiK31Model.Config,
        max_context_length=seq_len,
        dim=dim,
        vocab_size=2048,
        num_layers=17,
        full_attention_layers={3, 7, 11, 15, 16},
        full_attention=partial(
            _dsa_config,
            index_topk=128,
            num_index_heads=4,
            index_head_dim=64,
            indexer_loss_coeff=indexer_loss_coeff,
        ),
        attn_res_block_size=4,
        num_heads=16,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=512,
        latent_dim=128,
        expert_hidden_dim=128,
        num_experts=8,
        top_k=2,
        num_shared_experts=2,
        vision_encoder=vision_encoder_config(
            text_dim=dim,
            dim=256,
            qkv_dim=512,
            hidden_dim=512,
            num_layers=2,
            num_heads=4,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        parallelism=parallelism,
    )


def _kimi_k3_1_pro(
    *, seq_len: int, indexer_loss_coeff: float, parallelism: ParallelismConfig
) -> KimiK3Model.Config:
    dim = 7168
    return kimi_k3_config(
        config_cls=KimiK31Model.Config,
        max_context_length=seq_len,
        dim=dim,
        vocab_size=163840,
        num_layers=93,
        full_attention_layers=set(range(3, 92, 4)) | {92},
        full_attention=partial(
            _dsa_config,
            index_topk=2048,
            num_index_heads=32,
            index_head_dim=128,
            indexer_loss_coeff=indexer_loss_coeff,
        ),
        attn_res_block_size=12,
        num_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=33792,
        latent_dim=3584,
        expert_hidden_dim=3072,
        num_experts=896,
        top_k=16,
        num_shared_experts=2,
        vision_encoder=vision_encoder_config(
            text_dim=dim,
            dim=1024,
            qkv_dim=1536,
            hidden_dim=4096,
            num_layers=27,
            num_heads=12,
            init_pos_emb_height=64,
            init_pos_emb_width=64,
        ),
        parallelism=parallelism,
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 16384),
    "Kimi-K3.1-Pro": (_kimi_k3_1_pro, 1_048_576),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    indexer_loss_coeff: float = 0.01,
) -> KimiK3Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    if indexer_loss_coeff < 0:
        raise ValueError(
            f"indexer_loss_coeff must be nonnegative, got {indexer_loss_coeff}."
        )
    return get_config(
        seq_len=context_len,
        indexer_loss_coeff=indexer_loss_coeff,
        parallelism=parallelism,
    )
