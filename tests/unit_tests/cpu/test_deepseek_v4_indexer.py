# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU test for DeepSeek-V4 CSA top-k selection (``Indexer.select``)."""

import unittest

import torch

from torchtitan.models.deepseek_v4.compressor import Indexer


class TestIndexerSelect(unittest.TestCase):
    def test_indexer_select_matches_old_topk(self):
        torch.manual_seed(0)
        device = torch.device("cpu")
        seqlen, ratio, topk = 512, 4, 16
        n_cmp = seqlen // ratio
        g = torch.Generator(device).manual_seed(11)
        idx_q = torch.randn(seqlen, 8, 32, generator=g, device=device)
        idx_k = torch.randn(n_cmp, 32, generator=g, device=device)
        idx_w = torch.randn(seqlen, 8, generator=g, device=device)

        selected = Indexer.select(
            idx_q, idx_k, idx_w, seqlen=seqlen, ratio=ratio, topk=topk
        )

        # Old formulation: causal-masked scores -> topk -> map invalid to -1.
        scores = torch.einsum("shd,td->sht", idx_q, idx_k)
        scores = scores.relu_() * idx_w.unsqueeze(-1)
        scores = scores.sum(dim=1)
        causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        mask = torch.arange(n_cmp, device=device).repeat(seqlen, 1) >= causal_limit
        scores = scores + torch.where(mask, torch.finfo(idx_q.dtype).min, 0)
        _, topk_idxs = scores.topk(min(topk, n_cmp), dim=-1)
        old = torch.where(topk_idxs >= causal_limit, -1, topk_idxs)

        # lightning_indexer marks causally unselectable slots -1 itself and
        # leaves order unspecified, so compare each row as a sorted set.
        self.assertEqual(selected.shape, (seqlen, topk))
        self.assertEqual(selected.dtype, torch.int32)
        self.assertTrue(
            torch.equal(selected.long().sort(dim=-1).values, old.sort(dim=-1).values)
        )


if __name__ == "__main__":
    unittest.main()
