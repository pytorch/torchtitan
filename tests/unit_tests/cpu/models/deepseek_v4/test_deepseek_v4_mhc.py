# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

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


def _reference_sinkhorn_comb(comb, sinkhorn_iters, eps):
    """Sinkhorn part of ``hc_split_sinkhorn_kernel`` in DeepSeek-V4's reference
    inference/kernel.py; comb ``[..., hc, hc]`` is the scaled and biased logits.

    https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L401
    """
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return comb


class TestHcSplitSinkhorn(unittest.TestCase):
    def test_comb_matches_reference(self):
        gen = torch.Generator().manual_seed(0)
        num_tokens, hc_mult = 5, 4
        config = HcSplitSinkhorn.Config(hc_mult=hc_mult)
        mixes = torch.randn(
            num_tokens, (2 + hc_mult) * hc_mult, generator=gen, dtype=torch.float64
        )
        # Token 0: columns 1.. of every comb row underflow exp. The reference
        # adds eps after the row softmax, so those columns recover to ~1/hc_mult
        # instead of staying zero.
        mixes[0, 2 * hc_mult :].view(hc_mult, hc_mult)[:, 1:] = -1000.0
        hc_scale = torch.ones(3, dtype=torch.float64)
        hc_base = torch.zeros((2 + hc_mult) * hc_mult, dtype=torch.float64)

        _, _, comb = config.build()(mixes, hc_scale, hc_base)
        expected = _reference_sinkhorn_comb(
            mixes[:, 2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult)),
            config.sinkhorn_iters,
            config.eps,
        )
        torch.testing.assert_close(comb, expected)


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
