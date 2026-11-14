# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment-owned GLM sizes; their historical names remain test contracts."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glm5 import build_model_config
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.glm5.flavors import build_debug_model_config, configure_glm5_training


def glm5_reference_model_config(*, seq_len=DEFAULT_DEBUG_MODEL_SEQ_LEN, parallelism=None):
    return build_debug_model_config(seq_len=seq_len, parallelism=parallelism)


def glm5_scaled_model_config(*, seq_len=4096, parallelism=None):
    sources = tuple(layer if layer < 3 or (layer - 2) % 4 == 0
                    else 2 + 4 * ((layer - 2) // 4) for layer in range(16))
    return build_debug_model_config(
        seq_len=seq_len, parallelism=parallelism, dim=768, vocab_size=32768,
        num_layers=16, num_dense_layers=3, index_sources=sources,
        n_heads=8, q_lora_rank=384, kv_lora_rank=192, qk_nope_head_dim=192,
        qk_rope_head_dim=64, v_head_dim=256, dense_hidden_dim=1536,
        moe_hidden_dim=384, num_experts=24, router_top_k=8,
        index_n_heads=8, index_head_dim=128, index_topk=512,
    )


TEST_MODEL_CONFIGS = {
    "debugmodel": (glm5_reference_model_config, DEFAULT_DEBUG_MODEL_SEQ_LEN),
    "glm5_2_scaled_debugmodel": (glm5_scaled_model_config, 4096),
}


def resolve_model_config(flavor, *, parallelism=None, seq_len=None,
                         indexer_stage="frozen", indexer_loss_coeff=1.0):
    parallelism = parallelism or ParallelismConfig()
    if flavor not in TEST_MODEL_CONFIGS:
        return build_model_config(flavor, parallelism=parallelism, seq_len=seq_len,
                                  indexer_stage=indexer_stage, indexer_loss_coeff=indexer_loss_coeff)
    factory, maximum = TEST_MODEL_CONFIGS[flavor]
    length = maximum if seq_len is None else seq_len
    if not 0 < length <= maximum:
        raise ValueError(f"Requested seq_len {length} exceeds test config limit {maximum}")
    return configure_glm5_training(factory(seq_len=length, parallelism=parallelism),
                                   parallelism=parallelism, indexer_stage=indexer_stage,
                                   indexer_loss_coeff=indexer_loss_coeff)
