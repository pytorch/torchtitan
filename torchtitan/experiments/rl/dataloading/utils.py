# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Iterable, Iterator

import torch

logger = logging.getLogger(__name__)


def _packed_field_dtype(pad_value: int | float | bool) -> torch.dtype:
    if isinstance(pad_value, bool):
        return torch.bool
    if isinstance(pad_value, int):
        return torch.long
    return torch.float32


def pack(
    samples: Iterable[dict[str, list]],
    max_seq_length: int,
    pad_values: dict[str, int | float | bool],
) -> Iterator[dict[str, torch.Tensor]]:
    """Greedy-pack variable-length samples into [1, max_seq_length] sequences.

    Takes an iterable of samples, appends each to a buffer. When the next
    sample doesn't fit, pads and yields the current buffer, then starts a
    new one with that sample. Yields remaining buffer at the end.

    Args:
        samples: Iterable of dicts. Each dict maps field names to lists of
            the same length. E.g. {"input_ids": [1,2,3], "loss_mask": [0,0,1]}.
        max_seq_length: Maximum tokens per packed sequence.
        pad_values: Pad value for each field to pack. Keys determine which
            fields are packed (concat + pad). The first key determines
            sample length.

    Yields:
        Dict with:
        - Token field tensors [1, max_seq_length] for each key in pad_values
        - "positions" tensor [1, max_seq_length] with per-document resets
        - "seq_lens" list[int] -- length of each sample in this row
    """
    keys = list(pad_values.keys())
    buffer: dict[str, list] = {key: [] for key in keys}
    position_buffer: list[int] = []
    seq_lens_buffer: list[int] = []
    buffer_length = 0

    def _flush() -> dict:
        nonlocal buffer, position_buffer, seq_lens_buffer, buffer_length
        pad_length = max_seq_length - buffer_length
        if pad_length > 0:
            for key in keys:
                buffer[key].extend([pad_values[key]] * pad_length)
            position_buffer.extend(range(pad_length))

        result: dict = {
            key: torch.tensor(
                buffer[key],
                dtype=_packed_field_dtype(pad_values[key]),
            ).unsqueeze(0)
            for key in keys
        }
        result["positions"] = torch.tensor(position_buffer, dtype=torch.long).unsqueeze(
            0
        )
        result["seq_lens"] = list(seq_lens_buffer)

        buffer = {key: [] for key in keys}
        position_buffer = []
        seq_lens_buffer = []
        buffer_length = 0
        return result

    for sample in samples:
        sample_length = len(sample[keys[0]])

        if sample_length > max_seq_length:
            logger.warning(
                "Dropping sample with length %d exceeding max_seq_length %d",
                sample_length,
                max_seq_length,
            )
            continue

        if buffer_length > 0 and buffer_length + sample_length > max_seq_length:
            yield _flush()

        for key in keys:
            buffer[key].extend(sample[key])
        position_buffer.extend(range(sample_length))
        seq_lens_buffer.append(sample_length)
        buffer_length += sample_length

    if buffer_length > 0:
        yield _flush()
