# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import pytest
import torch

from torchtitan.models.deepseek_v4.mhc import HcPost, HcSplitSinkhorn


def _reference_hc_post(x, residual, post, comb):
    """``Block.hc_post`` of DeepSeek-V4's reference inference/model.py, on its
    ``[B, S, ...]`` layout: x ``[B, S, D]``, residual ``[B, S, hc, D]``, post
    ``[B, S, hc]``, comb ``[B, S, hc, hc]``."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


class TestHcPost(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(0)
        num_tokens, hc_mult, dim = 5, 4, 8
        self.x = torch.randn(num_tokens, dim, generator=gen)
        self.residual = torch.randn(num_tokens, hc_mult, dim, generator=gen)
        self.post = torch.rand(num_tokens, hc_mult, generator=gen)
        # Not symmetric, so the test also tells comb.T @ residual from comb @ residual.
        self.comb = torch.rand(num_tokens, hc_mult, hc_mult, generator=gen)

    def test_matches_reference(self):
        out = HcPost.Config().build()(self.x, self.residual, self.post, self.comb)
        expected = _reference_hc_post(
            self.x[None], self.residual[None], self.post[None], self.comb[None]
        )[0]
        torch.testing.assert_close(out, expected)

    def test_mixes_the_residual_branches(self):
        out = HcPost.Config().build()(self.x, self.residual, self.post, self.comb)
        # y[j] = post[j] * x + sum_i comb[i, j] * residual[i]
        expected = (
            self.post.unsqueeze(-1) * self.x.unsqueeze(-2)
            + self.comb.transpose(-1, -2) @ self.residual
        )
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("sinkhorn_iters", [20, 1, 0])
def test_sinkhorn_per_entry_matches_tensor_form(sinkhorn_iters):
    """The compiled path's per-entry Sinkhorn equals eager's ``[T, hc, hc]`` loop, bitwise in fp64."""
    torch.manual_seed(0)
    sinkhorn = HcSplitSinkhorn.Config(hc_mult=4, sinkhorn_iters=sinkhorn_iters).build()
    mixes = torch.randn(64, (2 + 4) * 4, dtype=torch.float64)
    hc_scale = torch.rand(3, dtype=torch.float64)
    hc_base = torch.randn((2 + 4) * 4, dtype=torch.float64)

    expected = sinkhorn(mixes, hc_scale, hc_base)
    with mock.patch("torch.compiler.is_compiling", return_value=True):
        actual = sinkhorn(mixes, hc_scale, hc_base)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)
