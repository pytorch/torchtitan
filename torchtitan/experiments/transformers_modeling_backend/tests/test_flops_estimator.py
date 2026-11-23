# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import torch
from transformers import LlamaConfig

from torchtitan.experiments.transformers_modeling_backend import TitanModelConfig
from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)


def _build_estimator(attn_mask_type):
    hf_config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
        architectures=["LlamaForCausalLM"],
    )
    with tempfile.TemporaryDirectory() as hf_model_dir:
        hf_config.save_pretrained(hf_model_dir)
        config = HFTransformerModel.Config(
            model_config=TitanModelConfig(
                dim=64,
                n_layers=2,
                n_heads=4,
                n_kv_heads=4,
                attn_mask_type=attn_mask_type,
            )
        )
        config.load_hf_config(
            hf_model_id=hf_model_dir, max_context_length=128, deterministic=False
        )
    with torch.device("meta"):
        model = config.build()
    return config.build_flops_estimator(model, seq_len=128)


def test_attention_counts_keys_per_query():
    # Two documents of lengths 3 and 2, then one padding token.
    inputs = {
        "input": torch.empty(6),
        "positions": torch.tensor([0, 1, 2, 0, 1, 2]),
        "padding_mask": torch.tensor([False] * 5 + [True]),
    }
    block_causal = _build_estimator("block_causal")
    causal = _build_estimator("causal")
    # 2 layers, 4 heads, head_dim 16 for Q/K and V.
    flops_per_key = 2 * 6 * 4 * (16 + 16)
    # "block_causal" restarts at each document (1+2+3 + 1+2 = 9 keys); "causal"
    # attends across documents (1+2+3+4+5 = 15 keys).
    assert causal(inputs) - block_causal(inputs) == (15 - 9) * flops_per_key
    # The padding token adds no FLOPs.
    unpadded = {key: value[:5] for key, value in inputs.items()}
    assert block_causal(inputs) == block_causal(unpadded)
