# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common import GatedRMSNorm, Sigmoid, SiLU


class TestGatedRMSNorm(unittest.TestCase):
    def setUp(self):
        apply_local_compile([])

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

                actual = module(x, gate)
                expected = F.rms_norm(
                    x.float(),
                    (x.shape[-1],),
                    module.weight.float(),
                    module.eps,
                )
                expected = (expected * activation_fn(gate.float())).to(x.dtype)

                torch.testing.assert_close(actual, expected)
                self.assertEqual(list(module.state_dict()), ["weight"])

    def test_cast_before_gate_forward_and_gradients(self):
        """Match the reference's BF16 norm, weight, then FP32 gate operations."""
        torch.manual_seed(42)
        module = (
            GatedRMSNorm.Config(
                dim=128, eps=1e-6, activation_fn=Sigmoid.Config(), cast_before_gate=True
            )
            .build()
            .bfloat16()
        )
        with torch.no_grad():
            module.weight.uniform_(0.5, 1.5)
        x = torch.randn(4, 3, 128, dtype=torch.bfloat16, requires_grad=True)
        gate = torch.randn_like(x, requires_grad=True)
        weight = module.weight.detach().clone().requires_grad_()
        actual = module(x, gate)
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        expected = ((normalized.to(x.dtype) * weight) * gate.float().sigmoid()).to(
            x.dtype
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        upstream = torch.randn_like(actual)
        gradients = torch.autograd.grad(
            (actual * upstream).sum(), (x, gate, module.weight), retain_graph=True
        )
        expected_gradients = torch.autograd.grad(
            (expected * upstream).sum(), (x, gate, weight)
        )
        for actual_grad, expected_grad in zip(gradients, expected_gradients):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
