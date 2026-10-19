# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.llama3 import build_model_config
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# rope_scaling from the published config.json of each checkpoint
# (meta-llama/Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B).
_PUBLISHED_ROPE_SCALING = {
    "1B": 32.0,
    "3B": 32.0,
    "8B": 8.0,
}


class TestLlama3RopeScaling(unittest.TestCase):
    def test_rope_frequencies_match_published_checkpoints(self):
        for flavor, factor in _PUBLISHED_ROPE_SCALING.items():
            with self.subTest(flavor=flavor):
                config = build_model_config(
                    flavor, parallelism=ParallelismConfig(), seq_len=2
                )
                rope = config.layers[0].attention.rope.build()
                # The cache holds exp(i * position * inv_freq); position 1
                # gives the scaled inverse frequencies.
                inv_freq = torch.angle(rope._precompute_cache()[1])

                hf_config = LlamaConfig(
                    hidden_size=config.dim,
                    num_attention_heads=config.layers[0].attention.n_heads,
                    max_position_embeddings=131072,
                    rope_theta=500000.0,
                    rope_scaling={
                        "rope_type": "llama3",
                        "factor": factor,
                        "low_freq_factor": 1.0,
                        "high_freq_factor": 4.0,
                        "original_max_position_embeddings": 8192,
                    },
                )
                expected, _ = ROPE_INIT_FUNCTIONS["llama3"](hf_config, "cpu")
                torch.testing.assert_close(
                    inv_freq, expected.float(), rtol=1e-5, atol=0
                )


if __name__ == "__main__":
    unittest.main()
