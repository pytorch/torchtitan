# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Provisional K3 2.8T-A50B flavor: config-level construction only.

Meta-build (no materialization) verifying the parameterized generator
emits the K3-scale MoE (896 experts / 16 active). The EP@896 runtime
mesh smoke lives in scripts (needs 8 GPUs); this locks the config.
"""

import unittest

import torch

from torchtitan.models.kimi_k3 import config_registry, model_registry


class TestKimi2p8tFlavor(unittest.TestCase):
    def test_generator_emits_k3_scale_moe(self):
        kc = config_registry.build_kimi_linear_config("2p8t")
        self.assertEqual(kc.num_experts, 896)  # K3 blog
        self.assertEqual(kc.num_experts_per_token, 16)  # K3 blog

    def test_meta_build(self):
        # The config-registry function carries a "_provisional" suffix; the
        # model flavor it builds does not, and model_registry parses
        # <size>_<variant> with variant in baseline/block_attn_res/full_attn_res.
        spec = model_registry("kimi_k3_2p8t_block_attn_res")
        with torch.device("meta"):
            model = spec.model.build()
        moe_layers = [
            layer for layer in model.layers.values() if getattr(layer, "is_moe", False)
        ]
        self.assertGreater(len(moe_layers), 0)
        # rough total > 1T (provisional; exact reconciles at 7.27)
        n = sum(p.numel() for p in model.parameters())
        self.assertGreater(n, 1_000_000_000_000)


if __name__ == "__main__":
    unittest.main()
