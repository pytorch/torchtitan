# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A pipeline stage's input gradients are dense even when its first op is a cat."""

import unittest

import torch

from torchtitan.models.kimi_k3.model import _DenseGradient


def _is_dense(t: torch.Tensor) -> bool:
    return t.is_contiguous()


class TestDenseGradient(unittest.TestCase):
    def test_cat_backward_hands_views_and_the_boundary_makes_them_dense(self):
        h = torch.randn(4, 8, requires_grad=True)
        stack = torch.randn(4, 3, 8, requires_grad=True)
        values = torch.cat((stack, h.unsqueeze(1)), dim=1)
        gh, gs = torch.autograd.grad(values.float().pow(2).sum(), (h, stack))
        # This is what pipelining would size the P2P buffers from.
        self.assertFalse(_is_dense(gh))
        self.assertFalse(_is_dense(gs))

        h2 = torch.randn(4, 8, requires_grad=True)
        stack2 = torch.randn(4, 3, 8, requires_grad=True)
        hd, sd = _DenseGradient.apply(h2), _DenseGradient.apply(stack2)
        values2 = torch.cat((sd, hd.unsqueeze(1)), dim=1)
        gh2, gs2 = torch.autograd.grad(values2.float().pow(2).sum(), (h2, stack2))
        self.assertTrue(_is_dense(gh2))
        self.assertTrue(_is_dense(gs2))
        # Same values, and the forward is the identity.
        self.assertTrue(torch.equal(hd, h2))
        self.assertTrue(torch.equal(sd, stack2))
        self.assertTrue(torch.equal(gh2, 2 * h2))
        self.assertTrue(torch.equal(gs2, 2 * stack2))


if __name__ == "__main__":
    unittest.main()
