# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from torchtitan.models.common import GatedRMSNorm, Sigmoid, SiLU


class TestGatedRMSNorm(unittest.TestCase):
    def test_configurable_activation_matches_reference(self):
        x = torch.randn(4, 3, 8)
        gate = torch.randn_like(x)

        for activation_config, activation_fn in (
            (Sigmoid.Config(), torch.sigmoid),
            (SiLU.Config(), F.silu),
        ):
            with self.subTest(activation=type(activation_config).__qualname__):
                module = GatedRMSNorm.Config(
                    dim=8,
                    eps=1e-5,
                    activation_fn=activation_config,
                ).build()
                with torch.no_grad():
                    module.weight.normal_()

                eager_forward = type(module).forward._torchdynamo_orig_callable
                actual = eager_forward(module, x, gate)
                expected = F.rms_norm(
                    x.float(),
                    (x.shape[-1],),
                    module.weight.float(),
                    module.eps,
                )
                expected = (expected * activation_fn(gate.float())).to(x.dtype)

                torch.testing.assert_close(actual, expected)
                self.assertEqual(list(module.state_dict()), ["weight"])


if __name__ == "__main__":
    unittest.main()
