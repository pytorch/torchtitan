# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the shared, model-agnostic vision-side building blocks
reused across the torchtitan VLMs (kimi_k2_5, qwen3_5):

- vision<->text scatter (fusing vision features into the token sequence),
- the vision attention block-mask and the ``VisionAttention`` config check.

These are pure-tensor logic, so they run on CPU. Model-specific behavior and the
text decoder are covered by the per-model integration and numerics-parity tests,
not here.
"""

import unittest

import torch

from torchtitan.models.common import Linear
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.common.vision_encoder import (
    get_vision_block_mask_mod,
    VisionAttention,
)

_PLACEHOLDER = 99


class TestVisionScatter(unittest.TestCase):
    def test_placement_and_order(self):
        dim = 4
        # Runs in order: 1 token at pos 1 (item 0), 2 tokens at pos 3-4 (item 1).
        tokens = torch.tensor([[5, _PLACEHOLDER, 6, _PLACEHOLDER, _PLACEHOLDER, 7]])
        num_tokens_per_item = torch.tensor([1, 2])

        positions = get_vision_positions(tokens, num_tokens_per_item, _PLACEHOLDER)
        self.assertEqual(positions, [(0, 0, 1, 1), (1, 0, 3, 2)])

        inputs = torch.zeros(1, 6, dim)
        merged = torch.zeros(2, 2, dim)  # (num_items, max_tokens, dim)
        merged[0, 0, :] = 10
        merged[1, 0, :] = 20
        merged[1, 1, :] = 21

        out = scatter_vision_embeds(
            inputs, vision_embeds=merged, vision_positions=positions
        )
        # Vision features land at the placeholder rows, in item/token order.
        self.assertTrue(torch.equal(out[0, 1], torch.full((dim,), 10.0)))
        self.assertTrue(torch.equal(out[0, 3], torch.full((dim,), 20.0)))
        self.assertTrue(torch.equal(out[0, 4], torch.full((dim,), 21.0)))
        # Text rows are untouched.
        self.assertTrue(torch.equal(out[0, 0], torch.zeros(dim)))
        self.assertTrue(torch.equal(out[0, 2], torch.zeros(dim)))

    def test_run_count_mismatch_raises(self):
        tokens = torch.tensor([[5, _PLACEHOLDER, 6]])  # one run
        with self.assertRaises(ValueError):
            get_vision_positions(tokens, torch.tensor([1, 1]), _PLACEHOLDER)

    def test_run_length_mismatch_raises(self):
        tokens = torch.tensor([[5, _PLACEHOLDER, 6]])  # one run of length 1
        with self.assertRaises(ValueError):
            get_vision_positions(tokens, torch.tensor([2]), _PLACEHOLDER)

    def test_runs_do_not_merge_across_batch_rows(self):
        # Sample 0 ends with a placeholder run; sample 1 starts with one. These
        # are two distinct items and must not merge across the flattened row
        # boundary (the bug a naive flat shift would introduce).
        tokens = torch.tensor([[_PLACEHOLDER, _PLACEHOLDER], [_PLACEHOLDER, 5]])
        positions = get_vision_positions(tokens, torch.tensor([2, 1]), _PLACEHOLDER)
        self.assertEqual(positions, [(0, 0, 0, 2), (1, 1, 0, 1)])


class TestVisionBlockMask(unittest.TestCase):
    def test_mask_mod_is_block_diagonal(self):
        # item 0 has 2 valid patches, item 1 has 1; everything else is padding.
        mask_mod = get_vision_block_mask_mod(torch.tensor([2, 1]))
        z = torch.tensor(0)
        # Within item 0: valid patches attend; padding (idx 2) does not.
        self.assertTrue(bool(mask_mod(0, z, torch.tensor(0), torch.tensor(1))))
        self.assertFalse(bool(mask_mod(0, z, torch.tensor(2), torch.tensor(0))))
        # Item 1: only patch 0 is valid.
        self.assertTrue(bool(mask_mod(1, z, torch.tensor(0), torch.tensor(0))))
        self.assertFalse(bool(mask_mod(1, z, torch.tensor(1), torch.tensor(0))))


class TestVisionAttentionConfig(unittest.TestCase):
    def test_dim_not_divisible_by_num_heads_raises(self):
        lin = Linear.Config(in_features=10, out_features=10)
        cfg = VisionAttention.Config(
            dim=10, num_heads=3, wq=lin, wk=lin, wv=lin, proj=lin
        )
        with self.assertRaises(ValueError):
            VisionAttention(cfg)


if __name__ == "__main__":
    unittest.main()
