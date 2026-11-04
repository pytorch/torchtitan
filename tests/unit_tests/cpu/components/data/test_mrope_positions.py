# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.data.multimodal.image_processor import ImageProcessor
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mrope import get_mrope_positions
from torchtitan.components.data.multimodal.video_processor import VideoProcessor
from torchtitan.models.qwen3_5 import QWEN3_5_IMAGE_PROCESSOR, QWEN3_5_VIDEO_PROCESSOR

TEXT, IMAGE, VIDEO = 1, 9, 8


def _collator_positions(tokens, image_grids, video_grids=None, positions=None):
    # Qwen3.5's processors merge 2x2 patches into a token.
    collator = MultiModalCollator.__new__(MultiModalCollator)
    collator.image_processor = ImageProcessor.Config(**QWEN3_5_IMAGE_PROCESSOR).build()
    collator.video_processor = VideoProcessor.Config(**QWEN3_5_VIDEO_PROCESSOR).build()
    return collator._build_mrope_positions(
        torch.tensor(tokens),
        torch.tensor(image_grids),
        torch.tensor(video_grids) if video_grids else None,
        torch.tensor(positions) if positions else None,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
    )


def test_matches_collator_for_one_document():
    # 3 text, a 1x4x6 image (1x2x3 tokens), 2 text, a 1x8x4 image (1x4x2 tokens),
    # 4 text.
    tokens = [TEXT] * 3 + [IMAGE] * 6 + [TEXT] * 2 + [IMAGE] * 8 + [TEXT] * 4

    positions = get_mrope_positions(len(tokens), [(3, (1, 2, 3)), (11, (1, 4, 2))])

    assert torch.equal(positions, _collator_positions(tokens, [(1, 4, 6), (1, 8, 4)]))


def test_collator_restarts_positions_per_packed_document():
    # Document 1: 2 text, a 1x4x4 image (1x2x2 tokens), 1 text. Document 2: 1 text,
    # a 2x4x4 video (2x2x2 tokens), 2 text.
    first = [TEXT] * 2 + [IMAGE] * 4 + [TEXT]
    second = [TEXT] + [VIDEO] * 8 + [TEXT] * 2

    positions = _collator_positions(
        first + second,
        [(1, 4, 4)],
        [(2, 4, 4)],
        positions=list(range(len(first))) + list(range(len(second))),
    )

    assert torch.equal(
        positions,
        torch.cat(
            [
                get_mrope_positions(7, [(2, (1, 2, 2))]),
                get_mrope_positions(11, [(1, (2, 2, 2))]),
            ]
        ),
    )


def test_docstring_example():
    positions = get_mrope_positions(9, [(2, (1, 2, 3))])

    assert positions.T.tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5],
        [0, 1, 2, 2, 2, 3, 3, 3, 5],
        [0, 1, 2, 3, 4, 2, 3, 4, 5],
    ]


def test_text_only_positions_repeat_the_token_index():
    positions = get_mrope_positions(5, [])

    assert torch.equal(positions, torch.arange(5).unsqueeze(1).expand(5, 3))


def test_rejects_an_image_past_the_sequence():
    with pytest.raises(ValueError, match="ends past"):
        get_mrope_positions(8, [(4, (1, 2, 3))])
