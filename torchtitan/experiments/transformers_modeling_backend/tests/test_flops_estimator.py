# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import pytest
import torch
from transformers import LlamaConfig

from torchtitan.experiments.transformers_modeling_backend import TitanModelConfig
from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)


# "block_causal" masks follow document boundaries; "causal" masks attend across
# them. Two documents of lengths 3 and 2, then one padding token.
@pytest.mark.parametrize(
    "attn_mask_type,num_attended_keys",
    [("block_causal", 1 + 2 + 3 + 1 + 2), ("causal", 1 + 2 + 3 + 4 + 5)],
)
def test_attention_counts_keys_per_query(attn_mask_type, num_attended_keys):
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
    estimate = config.build_flops_estimator(model, seq_len=128)
    _, active_nparams = config._parameter_counts(model, include_embeddings=False)
    # 2 layers, 4 heads, head_dim 16 for Q/K and V.
    flops_per_key = 2 * 6 * 4 * (16 + 16)
    inputs = {
        "input": torch.empty(6),
        "positions": torch.tensor([0, 1, 2, 0, 1, 2]),
        "padding_mask": torch.tensor([False] * 5 + [True]),
    }
    assert estimate(inputs) == (
        5 * 6 * active_nparams + num_attended_keys * flops_per_key
    )
