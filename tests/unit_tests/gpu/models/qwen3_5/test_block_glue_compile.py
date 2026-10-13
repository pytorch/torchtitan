# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.qwen3_5 import model as qwen3_5


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBlockGlueLocalCompile(unittest.TestCase):
    def setUp(self):
        # A local generator leaves the global RNG stream unchanged for later tests.
        self.generator = torch.Generator(device="cuda").manual_seed(0)

    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, generator=self.generator
        )

    def test_residual_add_norm_matches_separate_ops(self):
        norm = qwen3_5.OffsetRMSNorm(qwen3_5.OffsetRMSNorm.Config(dim=2048)).cuda()
        with torch.no_grad():
            norm.weight.copy_(self.randn(2048))
        for num_tokens in (1, 34, 4096):
            x, h = self.randn(num_tokens, 2048), self.randn(num_tokens, 2048)
            with torch.no_grad():
                # Before: a separate add, then the compiled offset_rmsnorm region.
                apply_local_compile(["offset_rmsnorm"])
                expected_x = x + h
                expected_norm = norm(expected_x)
                apply_local_compile(["offset_rmsnorm", "residual_norm"])
                actual_x, actual_norm = qwen3_5._residual_add_norm(x, h, norm)
            torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=0)
            torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)

    def test_residual_add_norm_trains_as_accurately_as_separate_ops(self):
        norm = qwen3_5.OffsetRMSNorm(qwen3_5.OffsetRMSNorm.Config(dim=2048)).cuda()
        with torch.no_grad():
            norm.weight.copy_(self.randn(2048))
        x0, h0, grad_x, grad_norm = (self.randn(4096, 2048) for _ in range(4))

        def run(regions: list[str] | None) -> list[torch.Tensor]:
            norm.weight.grad = None
            if regions is None:  # FP64 reference
                weight = norm.weight.detach().double().requires_grad_()
                x, h = (t.double().requires_grad_() for t in (x0, h0))
                summed = x + h
                rms = torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + norm.eps)
                normed = summed * rms * (1 + weight)
                (
                    summed * grad_x.double() + normed * grad_norm.double()
                ).sum().backward()
                return [summed, normed, x.grad, h.grad, weight.grad]
            apply_local_compile(regions)
            x, h = (t.clone().requires_grad_() for t in (x0, h0))
            summed, normed = qwen3_5._residual_add_norm(x, h, norm)
            (summed * grad_x + normed * grad_norm).float().sum().backward()
            return [summed, normed, x.grad, h.grad, norm.weight.grad]

        reference = run(None)
        separate = run(["offset_rmsnorm"])
        fused = run(["offset_rmsnorm", "residual_norm"])
        for name, ref, sep, fus in zip(
            ("sum", "norm", "grad_x", "grad_h", "grad_weight"),
            reference,
            separate,
            fused,
        ):
            sep_err = ((sep.double() - ref).norm() / ref.norm()).item()
            fus_err = ((fus.double() - ref).norm() / ref.norm()).item()
            # One FP32 sum instead of two BF16 roundings: never less accurate.
            self.assertLessEqual(fus_err, sep_err * 1.01, name)


if __name__ == "__main__":
    unittest.main()
