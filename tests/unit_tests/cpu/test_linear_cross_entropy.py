# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from torchtitan.components.loss import LinearCrossEntropyLoss
from torchtitan.models.common.linear import CastLinear, Linear


class LinearCrossEntropyLossTest(unittest.TestCase):
    def test_loss_and_gradients_match_projection_and_cross_entropy(self):
        torch.manual_seed(5)
        head = Linear.Config(in_features=16, out_features=32, bias=True).build()
        loss_fn = LinearCrossEntropyLoss.Config(batch_chunk_size=3).build()
        loss_fn.set_lm_head(head)
        x = torch.randn(7, 16, requires_grad=True)
        labels = torch.tensor([2, 1, -100, 5, 4, 11, 0])
        expected = F.cross_entropy(head(x).float(), labels, reduction="sum") / 6
        expected_grads = torch.autograd.grad(expected, (x, head.weight, head.bias))
        actual, metrics = loss_fn(x, labels, torch.tensor(6))
        actual_grads = torch.autograd.grad(actual, (x, head.weight, head.bias))
        torch.testing.assert_close(actual, expected)
        for result, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(result, reference)
        self.assertEqual(metrics, {})
        self.assertEqual(set(head.state_dict()), {"weight", "bias"})

    def test_rejects_custom_compute_and_stacked_heads(self):
        for config in (
            CastLinear.Config(in_features=16, out_features=32),
            Linear.Config(in_features=16, out_features=32, num_linears=2),
        ):
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, "ordinary"
            ):
                LinearCrossEntropyLoss.Config().build().set_lm_head(config.build())

    def test_requires_positive_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            LinearCrossEntropyLoss.Config(batch_chunk_size=0).build()


if __name__ == "__main__":
    unittest.main()
