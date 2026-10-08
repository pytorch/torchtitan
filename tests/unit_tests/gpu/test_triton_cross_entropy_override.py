# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from torchtitan.components.loss import IGNORE_INDEX

from torchtitan_recipes.overrides.triton_cross_entropy import triton_cross_entropy_loss


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestTritonCrossEntropy(unittest.TestCase):
    def test_forward_and_backward_match_eager(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(
            4,
            128256,
            device="cuda",
            dtype=torch.bfloat16,
        )
        labels = torch.tensor([0, 128255, IGNORE_INDEX, 1234], device="cuda")
        eager_logits = logits.detach().clone().requires_grad_()
        triton_logits = logits.detach().clone().requires_grad_()

        eager_loss = F.cross_entropy(
            eager_logits.float(),
            labels,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        triton_loss = triton_cross_entropy_loss(triton_logits, labels)
        eager_loss.backward()
        triton_loss.backward()

        torch.testing.assert_close(triton_loss, eager_loss, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            triton_logits.grad,
            eager_logits.grad,
            rtol=2e-2,
            atol=1e-3,
        )
