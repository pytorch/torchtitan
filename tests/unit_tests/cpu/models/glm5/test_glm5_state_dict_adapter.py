# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glm5 import build_model_config
from torchtitan.models.glm5.state_dict_adapter import Glm5StateDictAdapter


class TestGlm5StateDictAdapter(unittest.TestCase):
    def _assert_hf_roundtrip(self, flavor: str) -> None:
        config = build_model_config(
            flavor,
            parallelism=ParallelismConfig(),
            seq_len=8,
        )
        model = config.build()
        model.init_states()
        native = model.state_dict()

        adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        restored = adapter.from_hf(adapter.to_hf(native))

        self.assertEqual(native.keys(), restored.keys())
        for key in native:
            torch.testing.assert_close(native[key], restored[key], rtol=0, atol=0)

    def test_debug_model_hf_roundtrip(self) -> None:
        self._assert_hf_roundtrip("glm5_2_debugmodel")

    def test_mtp_debug_model_hf_roundtrip(self) -> None:
        flavor = "glm5_2_debugmodel_mtp"
        config = build_model_config(
            flavor,
            parallelism=ParallelismConfig(),
            seq_len=8,
        )
        model = config.build()
        model.init_states()
        native = model.state_dict()

        adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        hf_state = adapter.to_hf(native)
        mtp_layer = len(config.layers)

        # The released GLM-5.2 checkpoint stores MTP as the next layer in the
        # normal ``model.layers`` namespace.  It is a full DSA/MoE block with
        # DeepSeek-style embedding/hidden fusion and a shared-head norm.  The
        # vocabulary projection itself remains the top-level shared lm_head.
        expected_mtp_keys = {
            f"model.layers.{mtp_layer}.enorm.weight",
            f"model.layers.{mtp_layer}.hnorm.weight",
            f"model.layers.{mtp_layer}.eh_proj.weight",
            f"model.layers.{mtp_layer}.shared_head.norm.weight",
            f"model.layers.{mtp_layer}.self_attn.q_a_proj.weight",
            f"model.layers.{mtp_layer}.self_attn.indexer.wq_b.weight",
            f"model.layers.{mtp_layer}.mlp.gate.weight",
        }
        self.assertTrue(expected_mtp_keys.issubset(hf_state))
        self.assertNotIn(
            f"model.layers.{mtp_layer}.shared_head.head.weight",
            hf_state,
        )

        restored = adapter.from_hf(hf_state)
        self.assertEqual(native.keys(), restored.keys())
        for key in native:
            torch.testing.assert_close(native[key], restored[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
