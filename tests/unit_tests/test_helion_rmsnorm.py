# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest

import torch
import torch.nn.functional as F


_HAS_HELION = importlib.util.find_spec("helion") is not None

if _HAS_HELION:
    from torchtitan.experiments.helion_kernels.rmsnorm import (
        rms_norm_helion,
        rms_norm_helion_raw,
    )


@unittest.skipUnless(_HAS_HELION, "requires Helion")
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestRMSNormHelion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _assert_matches_reference(self, shape: tuple[int, ...]) -> None:
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            expected = F.rms_norm(x, (shape[-1],), weight, 1e-5)
            actual = rms_norm_helion(x, weight, 1e-5)
            raw = rms_norm_helion_raw(x, weight, 1e-5)

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(raw, expected, rtol=1e-2, atol=1e-2)

    def test_forward_matches_pytorch_reference_for_qwen_shapes(self):
        for shape in (
            (1, 1, 5120),
            (1, 192, 5120),
            (1, 1088, 5120),
            (1, 8, 128),
            (1088, 8, 128),
        ):
            with self.subTest(shape=shape):
                self._assert_matches_reference(shape)

    def test_training_falls_back_to_autograd_safe_pytorch_path(self):
        x = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        x.requires_grad_(True)
        weight.requires_grad_(True)

        actual = rms_norm_helion(x, weight, 1e-5)
        expected = F.rms_norm(x, (128,), weight, 1e-5)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

        actual.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

    def test_unsupported_no_weight_case_falls_back(self):
        x = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)

        actual = rms_norm_helion(x, None, 1e-5)
        expected = F.rms_norm(x, (128,), None, 1e-5)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
