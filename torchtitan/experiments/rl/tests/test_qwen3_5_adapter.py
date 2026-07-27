# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for Qwen35StateDictAdapter weight-tying behavior.

Qwen3.5 ties the LM head to the token embedding (HF tie_word_embeddings=true),
so ``to_hf`` must drop ``lm_head.weight`` when ``enable_weight_tying`` is set --
otherwise the natively-tied vLLM model would reject the extra key.
"""

import dataclasses
import unittest

import torch


class TestQwen35AdapterTying(unittest.TestCase):
    def test_to_hf_skips_lm_head_when_tied(self):
        try:
            from torchtitan.models.qwen3_5 import qwen3_5_configs
            from torchtitan.models.qwen3_5.state_dict_adapter import (
                Qwen35StateDictAdapter,
            )
        except ImportError as e:
            # qwen3_5 imports flash-linear-attention; skip where it is absent.
            self.skipTest(f"qwen3_5 dependencies unavailable: {e}")

        model_config = dataclasses.replace(
            qwen3_5_configs["debugmodel"](attn_backend="varlen"),
            enable_weight_tying=True,
        )
        adapter = Qwen35StateDictAdapter(model_config, hf_assets_path=None)

        # to_hf passes tok_embeddings/lm_head through unchanged (no transpose),
        # so the tensor shape is irrelevant to the tying skip under test.
        w = torch.zeros(2, 2)
        hf = adapter.to_hf({"tok_embeddings.weight": w, "lm_head.weight": w})

        self.assertNotIn("lm_head.weight", hf)
        self.assertIn("model.language_model.embed_tokens.weight", hf)


if __name__ == "__main__":
    unittest.main()
