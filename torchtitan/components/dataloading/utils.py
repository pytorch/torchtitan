# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable, Iterator

import torch


def pack(
    samples: Iterable[dict[str, list]],
    token_keys: list[str],
    max_seq_length: int,
    pad_values: dict[str, int | float],
) -> Iterator[dict[str, torch.Tensor]]:
    """Greedy-pack variable-length samples into [1, max_seq_length] sequences.

    Takes an iterable of samples, appends each to a buffer. When the next
    sample doesn't fit, pads and yields the current buffer, then starts a
    new one with that sample. Yields remaining buffer at the end.

    Args:
        samples: Iterable of dicts. Each dict maps field names to lists of
            the same length. E.g. {"input_ids": [1,2,3], "loss_mask": [0,0,1]}.
        token_keys: Which fields to pack (concat + pad). Must all have the
            same length within a sample. The first key determines sample length.
        max_seq_length: Maximum tokens per packed sequence.
        pad_values: Pad value for each token_keys field.

    Yields:
        Dict with:
        - Token field tensors [1, max_seq_length] for each key in token_keys
        - "positions" tensor [1, max_seq_length] with per-document resets
        - "seq_lens" list[int] — length of each sample in this row
    """
    buffer: dict[str, list] = {key: [] for key in token_keys}
    position_buffer: list[int] = []
    seq_lens_buffer: list[int] = []
    buffer_length = 0

    def _flush() -> dict:
        nonlocal buffer, position_buffer, seq_lens_buffer, buffer_length
        pad_length = max_seq_length - buffer_length
        if pad_length > 0:
            for key in token_keys:
                buffer[key].extend([pad_values[key]] * pad_length)
            position_buffer.extend(range(pad_length))

        result: dict = {
            key: torch.tensor(
                buffer[key],
                dtype=torch.long if key.endswith("_ids") else torch.float32,
            ).unsqueeze(0)
            for key in token_keys
        }
        result["positions"] = torch.tensor(position_buffer, dtype=torch.long).unsqueeze(
            0
        )
        result["seq_lens"] = list(seq_lens_buffer)

        buffer = {key: [] for key in token_keys}
        position_buffer = []
        seq_lens_buffer = []
        buffer_length = 0
        return result

    for sample in samples:
        sample_length = len(sample[token_keys[0]])

        if sample_length > max_seq_length:
            continue

        if buffer_length > 0 and buffer_length + sample_length > max_seq_length:
            yield _flush()

        for key in token_keys:
            buffer[key].extend(sample[key])
        position_buffer.extend(range(sample_length))
        seq_lens_buffer.append(sample_length)
        buffer_length += sample_length

    if buffer_length > 0:
        yield _flush()
