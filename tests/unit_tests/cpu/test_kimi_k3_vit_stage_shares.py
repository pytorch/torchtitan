# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The tower's DEP share decomposition must equal the unsplit tower.

Report 5.2.3 clause 2 splits the tower across pipeline stages. The split is only
a scheduling change, so head + body + tail must reproduce the single call
exactly. Ported in intent from the reference tree's test_vit_stage_roles.
"""

import unittest

import torch

from torchtitan.models.kimi_k3.config_registry import model_registry
from torchtitan.models.kimi_k3.vision_encoder import KimiK3VisionEncoder


def _tower():
    spec = model_registry("debugmodel")
    cfg = spec.model.vision_encoder
    torch.manual_seed(0)
    tower = cfg.build()
    tower.eval()
    return tower


def _batch(tower, h=4, w=4):
    grid = torch.tensor([[1, h, w]], dtype=torch.int32)
    n = int(grid.prod(dim=-1).sum())
    torch.manual_seed(1)
    dim_in = tower.patch_embed.weight.shape[1]
    return torch.randn(n, dim_in, dtype=tower.patch_embed.weight.dtype), grid


class TestViTStageShares(unittest.TestCase):
    def test_block_bounds_cover_every_block_exactly_once(self):
        tower = _tower()
        n = len(tower.layers)
        for k in range(1, n + 1):
            b = tower.block_bounds(k)
            self.assertEqual(b[0][0], 0)
            self.assertEqual(b[-1][1], n)
            for i in range(len(b) - 1):
                self.assertEqual(b[i][1], b[i + 1][0])
            sizes = [hi - lo for lo, hi in b]
            self.assertTrue(all(s > 0 for s in sizes))
            self.assertLessEqual(max(sizes) - min(sizes), 1)
            # The remainder goes to the LAST shares: share 0 also carries
            # patch_embed, so giving it an extra block unbalances the worst stage.
            self.assertEqual(sizes, sorted(sizes))

    def test_block_bounds_rejects_more_shares_than_blocks(self):
        tower = _tower()
        with self.assertRaises(ValueError):
            tower.block_bounds(len(tower.layers) + 1)

    def _assert_shares_match(self, num_shares):
        tower = _tower()
        x, grid = _batch(tower)
        with torch.no_grad():
            want = tower(x, grid_thw=grid)
            bounds = tower.block_bounds(num_shares)
            h = tower(x, grid_thw=grid, part="head", upto_block=bounds[0][1])
            for lo, hi in bounds[1:-1]:
                h = tower(h, grid_thw=grid, part="body", lo=lo, hi=hi)
            got = tower(h, grid_thw=grid, part="tail", from_block=bounds[-1][0])
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_head_tail_equals_single_stage(self):
        self._assert_shares_match(2)

    def test_head_body_tail_equals_single_stage(self):
        self._assert_shares_match(3)

    def test_four_shares_equal_single_stage(self):
        self._assert_shares_match(4)

    def test_unknown_part_is_rejected(self):
        tower = _tower()
        x, grid = _batch(tower)
        with self.assertRaises(ValueError):
            tower(x, grid_thw=grid, part="middle")


if __name__ == "__main__":
    unittest.main()
