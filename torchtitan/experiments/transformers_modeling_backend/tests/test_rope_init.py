# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that init_states() populates HF rotary buffers after meta init.

HF rotary modules compute ``inv_freq`` in ``__init__`` and register it as a
non-persistent buffer, so meta init + ``to_empty()`` leaves it uninitialized and
a checkpoint load cannot restore it. The failure is silent: RoPE just rotates by
garbage. Expected values come from the rotary module constructed directly on
CPU, independent of the init path under test.

Run with:
    python -m pytest torchtitan/experiments/transformers_modeling_backend/tests/test_rope_init.py -x -v
"""

import tempfile
import unittest

import torch
from transformers import LlamaConfig, Qwen3Config

from torchtitan.experiments.transformers_modeling_backend import TitanModelConfig
from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)

_SIZES = dict(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_hidden_layers=2,
    vocab_size=128,
    max_position_embeddings=256,
)


def _build_and_init(hf_config) -> HFTransformerModel:
    with tempfile.TemporaryDirectory() as hf_model_dir:
        hf_config.save_pretrained(hf_model_dir)
        config = HFTransformerModel.Config(
            model_config=TitanModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=4)
        )
        config.load_hf_config(
            hf_model_id=hf_model_dir, max_context_length=128, deterministic=False
        )
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    # Make "not reinitialized" deterministic instead of depending on whatever
    # memory to_empty() handed back.
    model.rotary_emb.inv_freq.fill_(float("nan"))
    model.rotary_emb.original_inv_freq.fill_(float("nan"))
    model.init_states(buffer_device=torch.device("cpu"))
    return model


class TestRotaryInit(unittest.TestCase):
    def test_init_states_populates_rotary_buffers(self):
        cases = {
            "llama_default": LlamaConfig(**_SIZES, architectures=["LlamaForCausalLM"]),
            "llama_llama3": LlamaConfig(
                **_SIZES,
                architectures=["LlamaForCausalLM"],
                rope_parameters={
                    "rope_type": "llama3",
                    "rope_theta": 10000.0,
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 64,
                },
            ),
            # dynamic re-registers inv_freq from original_inv_freq during forward,
            # so both buffers must be populated.
            "llama_dynamic": LlamaConfig(
                **_SIZES,
                architectures=["LlamaForCausalLM"],
                rope_parameters={
                    "rope_type": "dynamic",
                    "rope_theta": 10000.0,
                    "factor": 2.0,
                },
            ),
            "qwen3_yarn": Qwen3Config(
                **_SIZES,
                head_dim=16,
                architectures=["Qwen3ForCausalLM"],
                rope_parameters={
                    "rope_type": "yarn",
                    "rope_theta": 10000.0,
                    "factor": 4.0,
                    "original_max_position_embeddings": 64,
                },
            ),
        }
        for name, hf_config in cases.items():
            with self.subTest(name):
                rotary = _build_and_init(hf_config).rotary_emb
                expected = type(rotary)(rotary.config)
                torch.testing.assert_close(rotary.inv_freq, expected.inv_freq)
                torch.testing.assert_close(
                    rotary.original_inv_freq, expected.original_inv_freq
                )
                self.assertEqual(rotary.attention_scaling, expected.attention_scaling)


if __name__ == "__main__":
    unittest.main()
