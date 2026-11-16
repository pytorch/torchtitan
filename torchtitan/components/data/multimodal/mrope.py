# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""3D M-RoPE positions (Qwen-VL style) from placeholder offsets and patch grids."""

from collections.abc import Sequence

import torch


def get_mrope_positions(
    num_tokens: int,
    vision_items: Sequence[tuple[int, tuple[int, int, int]]],
) -> torch.Tensor:
    """Return the ``(num_tokens, 3)`` temporal/height/width positions of one sequence.

    ``vision_items`` lists ``(offset, (t, h, w))`` per image or video: where its
    placeholder run starts and its token grid, the patch grid after spatial merging
    (e.g. ``(t, h // 2, w // 2)`` with a 2x2 merge). A text token advances all three
    axes by one. A placeholder run of ``t * h * w`` tokens takes the
    ``(t, row, column)`` coordinates of its tokens in row-major order, shifted by the
    position after the preceding text; the text after it resumes one past the largest
    coordinate. A completion appended after the prompt continues the last text run.

    Callers find the offsets their own way: the multimodal collator
    (``MultiModalCollator._build_mrope_positions``) scans each packed document's token ids
    for placeholder runs; RL takes them from the renderer (vLLM passes them to
    ``get_mrope_input_positions``), because a placeholder id the policy samples in a
    completion is text, not an image.

    Example: 2 text tokens, an image with token grid 1 x 2 x 3 (a 1 x 4 x 6 patch
    grid, merge 2), 1 text token::

        token     t0  t1  i0  i1  i2  i3  i4  i5  t2
        temporal   0   1   2   2   2   2   2   2   5
        height     0   1   2   2   2   3   3   3   5
        width      0   1   2   3   4   2   3   4   5
    """
    positions = torch.empty(num_tokens, 3, dtype=torch.int64)
    cursor = 0  # next token index
    next_position = 0
    for offset, (t, h, w) in sorted(vision_items):
        length = t * h * w
        if offset < cursor or offset + length > num_tokens:
            raise ValueError(
                f"Vision item at offset {offset} with {length} tokens overlaps another "
                f"item or ends past the {num_tokens}-token sequence."
            )
        text = torch.arange(next_position, next_position + offset - cursor)
        positions[cursor:offset] = text.unsqueeze(1)
        next_position += offset - cursor
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"
            ),
            dim=-1,
        )
        positions[offset : offset + length] = grid.reshape(length, 3) + next_position
        next_position += max(t, h, w)
        cursor = offset + length
    positions[cursor:] = torch.arange(
        next_position, next_position + num_tokens - cursor
    ).unsqueeze(1)
    return positions
