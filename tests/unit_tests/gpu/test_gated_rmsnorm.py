# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common import GatedRMSNorm, Sigmoid


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGatedRMSNormCompile(unittest.TestCase):
    def tearDown(self):
        torch._dynamo.reset()

    @staticmethod
    def _make_module() -> GatedRMSNorm:
        return (
            GatedRMSNorm.Config(
                dim=128,
                eps=1e-5,
                activation_fn=Sigmoid.Config(),
            )
            .build()
            .cuda()
            .to(torch.bfloat16)
        )

    def test_default_forward_and_backward_emit_triton(self):
        from torch._inductor.utils import run_fw_bw_and_get_code

        module = self._make_module()
        x = torch.randn(
            8,
            6,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        gate = torch.randn_like(x, requires_grad=True)

        _, codes = run_fw_bw_and_get_code(lambda: module(x, gate))

        self.assertGreaterEqual(sum("triton" in code for code in codes), 2)
        self.assertTrue(any("sigmoid" in code and "rsqrt" in code for code in codes))


if __name__ == "__main__":
    unittest.main()
