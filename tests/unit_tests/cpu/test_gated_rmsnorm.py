# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from torchtitan.models.common.nn_modules import GatedRMSNorm
from torchtitan.models.kimi_k3.kda import KimiGatedRMSNorm, KimiRMSNormGated
from torchtitan.models.qwen3_5.gdn import Qwen35GatedRMSNorm, RMSNormGated


class TestGatedRMSNorm(unittest.TestCase):
    def test_legacy_names_are_preserved(self):
        self.assertIs(KimiRMSNormGated, KimiGatedRMSNorm)
        self.assertIs(RMSNormGated, Qwen35GatedRMSNorm)

    def test_model_specific_defaults_use_shared_implementation(self):
        kimi_config = KimiGatedRMSNorm.Config(dim=8)
        qwen_config = Qwen35GatedRMSNorm.Config(dim=8)
        kimi = kimi_config.build()
        qwen = qwen_config.build()

        self.assertIs(kimi.activation_fn, torch.sigmoid)
        self.assertEqual(kimi_config.eps, 1e-5)
        self.assertFalse(kimi.round_normalized_to_input_dtype)
        self.assertIs(qwen.activation_fn, F.silu)
        self.assertEqual(qwen_config.eps, 1e-6)
        self.assertTrue(qwen.round_normalized_to_input_dtype)
        self.assertTrue(issubclass(KimiGatedRMSNorm, GatedRMSNorm))
        self.assertTrue(issubclass(Qwen35GatedRMSNorm, GatedRMSNorm))

    def test_model_specific_modules_match_shared_formula(self):
        input = torch.randn(4, 3, 8)
        gate = torch.randn_like(input)

        for config in (
            KimiGatedRMSNorm.Config(dim=8),
            Qwen35GatedRMSNorm.Config(dim=8),
        ):
            with self.subTest(config=type(config).__qualname__):
                module = config.build()
                with torch.no_grad():
                    module.weight.normal_()
                expected = F.rms_norm(
                    input.float(),
                    (input.shape[-1],),
                    module.weight.float(),
                    module.eps,
                )
                if module.round_normalized_to_input_dtype:
                    expected = expected.to(input.dtype)
                expected = expected * module.activation_fn(gate.float())
                torch.testing.assert_close(
                    module(input, gate),
                    expected.to(input.dtype),
                )
                self.assertEqual(list(module.state_dict()), ["weight"])


if __name__ == "__main__":
    unittest.main()
