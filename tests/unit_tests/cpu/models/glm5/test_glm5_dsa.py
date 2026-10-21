# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.glm5.dsa import create_dsa_causal_mask, selected_mask


class TestGlm5Dsa(unittest.TestCase):
    def test_packed_document_causal_mask(self) -> None:
        positions = torch.tensor([0, 1, 2, 0, 1])
        mask = create_dsa_causal_mask(positions, dtype=torch.float32)
        allowed = mask[0] == 0
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(allowed, expected)

    def test_selected_mask_intersects_topk_and_causality(self) -> None:
        positions = torch.arange(4)
        additive = create_dsa_causal_mask(positions, dtype=torch.float32)
        indices = torch.tensor([[0, -1], [1, 0], [3, 1], [3, 3]])
        actual = selected_mask(indices, additive)
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(actual, expected)

    def test_padding_mask_excludes_queries_and_keys(self) -> None:
        mask = create_dsa_causal_mask(
            torch.arange(4),
            dtype=torch.float32,
            padding_mask_T=torch.tensor([True, True, False, False]),
        )
        allowed = mask[0] == 0
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(allowed, expected)


if __name__ == "__main__":
    unittest.main()
