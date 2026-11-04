# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mrope import get_mrope_positions

TEXT, IMAGE, VIDEO = 1, 9, 8


def _collator_positions(tokens: list[int], grids: list[tuple[int, int, int]]):
    collator = MultiModalCollator.__new__(MultiModalCollator)
    collator.patch_order = "block"
    collator.spatial_merge_size = 2
    return collator._build_mrope_positions(
        torch.tensor(tokens),
        torch.tensor(grids),
        None,
        None,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
    )


def test_matches_collator_for_one_document():
    # 3 text, a 1x4x6 image (6 tokens), 2 text, a 1x8x4 image (8 tokens), 4 text.
    tokens = [TEXT] * 3 + [IMAGE] * 6 + [TEXT] * 2 + [IMAGE] * 8 + [TEXT] * 4
    grids = [(1, 4, 6), (1, 8, 4)]

    positions = get_mrope_positions(
        len(tokens), [(3, grids[0]), (11, grids[1])], spatial_merge_size=2
    )

    assert torch.equal(positions, _collator_positions(tokens, grids))


def test_docstring_example():
    positions = get_mrope_positions(9, [(2, (1, 4, 6))], spatial_merge_size=2)

    assert positions.T.tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5],
        [0, 1, 2, 2, 2, 3, 3, 3, 5],
        [0, 1, 2, 3, 4, 2, 3, 4, 5],
    ]


def test_text_only_positions_repeat_the_token_index():
    positions = get_mrope_positions(5, [], spatial_merge_size=2)

    assert torch.equal(positions, torch.arange(5).unsqueeze(1).expand(5, 3))


def test_rejects_an_image_past_the_sequence():
    with pytest.raises(ValueError, match="ends past"):
        get_mrope_positions(8, [(4, (1, 4, 6))], spatial_merge_size=2)
