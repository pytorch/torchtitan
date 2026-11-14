# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Segment padding shared by supervised and RL token packing."""

import numpy as np

from torchtitan.tools.utils import round_up


def append_padded_segment(
    parts: dict[str, list[np.ndarray]],
    fields: dict[str, np.ndarray],
    *,
    per_sample_pad_multiple: int,
    pad_values: dict[str, int | float | bool],
    padding_mask: np.ndarray | None,
) -> int:
    """Append one segment's fields, positions and padding mask; return its length.

    Callers shift labels, split documents, choose field-specific fill values,
    and pad row tails. ``fields`` contains equally long 1-D arrays, including
    ``input_ids``; ``parts`` also holds ``positions`` and ``padding_mask`` lists.
    Five tokens padded to eight have positions 0..7; the last three are masked,
    and existing padding stays masked. Keep tokens and padding separate until
    row assembly to avoid copying tokens twice.
    """
    num_tokens = len(fields["input_ids"])
    if any(len(values) != num_tokens for values in fields.values()):
        raise ValueError("Segment fields must share one length")
    padded_length = round_up(num_tokens, per_sample_pad_multiple)
    num_padding = padded_length - num_tokens
    for key, values in fields.items():
        parts[key].append(values)
        if num_padding:
            parts[key].append(np.full(num_padding, pad_values[key], dtype=values.dtype))
    parts["positions"].append(np.arange(padded_length, dtype=np.int64))
    mask = np.ones(padded_length, dtype=np.bool_)
    mask[:num_tokens] = False if padding_mask is None else padding_mask
    parts["padding_mask"].append(mask)
    return padded_length
