# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

from torchtitan.components.loss import ChunkedLossWrapper, MSELoss
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.linear import Linear


class TestLinearCrossEntropy(unittest.TestCase):
    def setUp(self):
        apply_local_compile([])
        torch.manual_seed(42)

    def tearDown(self):
        apply_local_compile([])

    def test_compiled_loss_and_gradients(self):
        heads = [
            Linear.Config(in_features=16, out_features=31, bias=True).build()
            for _ in range(2)
        ]
        heads[1].load_state_dict(heads[0].state_dict())
        x = torch.randn(16, 16)
        labels = torch.randint(0, 31, (16,))
        labels[:4] = -100
        results = []
        apply_local_compile(["loss"])
        for joint, head in zip((False, True), heads, strict=True):
            inputs = x.clone().requires_grad_()
            loss_fn = ChunkedLossWrapper.Config(
                num_chunks=4, linear_cross_entropy=joint
            ).build()
            loss_fn.set_lm_head(head)
            loss, _ = loss_fn(inputs, labels, torch.tensor(29))
            loss.backward()
            results.append((loss, inputs.grad, head.weight.grad, head.bias.grad))
        for reference, actual in zip(*results, strict=True):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)

    def test_matches_chunked_loss_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            for ignored in (False, True):
                for requires_grad in (False, True):
                    with self.subTest(dtype=dtype, ignored=ignored, grad=requires_grad):
                        heads = [
                            Linear.Config(in_features=16, out_features=31, bias=True)
                            .build()
                            .to(dtype)
                            for _ in range(2)
                        ]
                        heads[1].load_state_dict(heads[0].state_dict())
                        # A strided input exercises the wrapper's chunk layout.
                        x = torch.randn(16, 32, dtype=dtype)[:, ::2]
                        labels = torch.randint(0, 31, (16,))
                        if ignored:
                            labels[:4] = -100
                            labels[7] = -100
                        results = []
                        for fused, head in zip((False, True), heads, strict=True):
                            inputs = x.detach().requires_grad_(requires_grad)
                            loss_fn = ChunkedLossWrapper.Config(
                                num_chunks=4, linear_cross_entropy=fused
                            ).build()
                            loss_fn.set_lm_head(head)
                            loss, metrics = loss_fn(inputs, labels, torch.tensor(29))
                            self.assertEqual(metrics, {})
                            if requires_grad:
                                loss.backward()
                            results.append((loss, inputs.grad))
                        torch.testing.assert_close(
                            results[0][0], results[1][0], rtol=0, atol=0
                        )
                        if requires_grad:
                            torch.testing.assert_close(
                                results[0][1], results[1][1], rtol=0, atol=0
                            )
                            for a, b in zip(
                                heads[0].parameters(),
                                heads[1].parameters(),
                                strict=True,
                            ):
                                torch.testing.assert_close(
                                    a.grad, b.grad, rtol=0, atol=0
                                )

    def test_rejects_unsupported_contracts(self):
        with self.assertRaisesRegex(ValueError, "CrossEntropyLoss"):
            ChunkedLossWrapper.Config(
                linear_cross_entropy=True, loss_fn=MSELoss.Config()
            ).build()
        loss_fn = ChunkedLossWrapper.Config(linear_cross_entropy=True).build()
        for head in (
            HiMidLoLinear.Config(in_features=16, out_features=31).build(),
            Linear.Config(in_features=16, out_features=31, num_linears=2).build(),
        ):
            with self.assertRaisesRegex(ValueError, "ordinary Linear"):
                loss_fn.set_lm_head(head)
        loss_fn.set_lm_head(Linear.Config(in_features=16, out_features=31).build())
        x, y = torch.randn(16, 16), torch.zeros(16, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "one prediction"):
            loss_fn((x,), (y,))
        with patch("torchtitan.components.loss.spmd_mesh_size", return_value=2):
            with self.assertRaisesRegex(ValueError, "TP1"):
                loss_fn(x, y)


if __name__ == "__main__":
    unittest.main()
