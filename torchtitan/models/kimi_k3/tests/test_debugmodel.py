# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CI smoke for the kimi_k3_debugmodel flavor.

Config build + a forward/backward through the real module tree (KDA
via fla -- triton on GPU boxes, CPU fallback otherwise -- MLA SDPA,
8-expert MoE, Block AttnRes) in a few seconds. The GPU train smoke lives in the launcher docs:
``--module kimi_k3 --config kimi_k3_debugmodel`` (10 steps).
"""

import unittest

import torch

from torchtitan.models.kimi_k3 import config_registry

from torchtitan.models.kimi_k3.tests.kda_shmem import skip_reason_if_insufficient


class TestKimiDebugModel(unittest.TestCase):
    def test_trainer_config_builds(self):
        cfg = config_registry.kimi_k3_debugmodel()
        self.assertEqual(cfg.model_spec.flavor, "kimi_k3_debugmodel")
        kimi = cfg.model_spec.model.kimi_config
        self.assertEqual(kimi.num_hidden_layers, 4)
        self.assertEqual(kimi.vocab_size, 2016)
        self.assertEqual(kimi.num_experts, 8)

    def test_forward_backward(self):
        # fla's KDA kernel at kda_head_dim=64 outgrows consumer Blackwell's
        # shared memory under triton 3.8; see kda_shmem for the numbers.
        reason = skip_reason_if_insufficient()
        if reason:
            self.skipTest(reason)
        # fla dispatches to triton whenever CUDA is available (even for
        # CPU tensors), so run on GPU when present and only exercise the
        # CPU fallback on CUDA-less boxes.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = config_registry.kimi_k3_debugmodel()
        with torch.device(device):
            model = cfg.model_spec.model.build()
            model.init_weights()
            # KDA training path requires chunk mode (seq > 64).
            tokens = torch.randint(0, 2016, (1, 128))
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (1, 128, 2016))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        # AttnRes projections get gradients (zero-init but on the path).
        for name, p in model.named_parameters():
            if name.endswith("attention_res_proj.weight"):
                self.assertIsNotNone(p.grad, f"no grad at {name}")
                break


if __name__ == "__main__":
    unittest.main()
