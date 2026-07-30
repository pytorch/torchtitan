# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for efficient sample packing in multimodal datasets.

Uses a scan-and-pick algorithm: for each packed sequence, the entire buffer
is scanned to greedily select any sample that fits the remaining capacity,
producing tighter packing than a single-pass sorted approach.
"""

from collections import deque
from typing import Any

import torch


class MMSamplePacker:
    """Packs multiple samples to maximize sequence length utilization.

    Samples are accumulated in an internal buffer.  When the buffer reaches
    ``buffer_size``, a scan-and-pick pass packs samples into sequences of up
    to ``max_seq_length``. Partial sequences wait for more samples until the
    lookahead buffer fills, then are emitted to guarantee progress.
    """

    def __init__(
        self,
        max_seq_length: int,
        buffer_size: int = 100,
    ):
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size

        self._sample_buffer: dict[int, dict[str, Any]] = {}
        self._next_id: int = 0
        self.packed_samples: deque = deque()

    def _pack_buffered_samples(self, flush: bool = False) -> None:
        """Pack buffered samples into sequences using scan-and-pick.

        Repeatedly scans the buffer to greedily fill each packed sequence.
        When ``flush=False``, an incomplete sequence stays buffered until the
        lookahead buffer fills.

        O(N * K) where N = buffer size, K = number of packed sequences.
        Negligible vs data loading and model forward for typical buffer sizes.
        """
        while self._sample_buffer:
            picked_ids: list[int] = []
            current_length = 0

            for sid, sample in self._sample_buffer.items():
                length = len(sample["input_ids"])
                if current_length + length <= self.max_seq_length:
                    picked_ids.append(sid)
                    current_length += length

            # Oversized rows are filtered before entering the packer.
            if not picked_ids:
                raise RuntimeError(
                    "multimodal packer received a row longer than max_seq_length"
                )

            # Keep a partial row only while more lookahead slots remain.
            if (
                not flush
                and current_length < self.max_seq_length
                and len(self._sample_buffer) < self.buffer_size
            ):
                break

            samples = [self._sample_buffer.pop(sid) for sid in picked_ids]
            self.packed_samples.append(self._merge_samples(samples))

    @staticmethod
    def _merge_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {
            "input_ids": torch.cat([s["input_ids"] for s in samples]),
            "labels": torch.cat([s["labels"] for s in samples]),
            "positions": torch.cat([s["positions"] for s in samples]),
            "pixel_values": [img for s in samples for img in s.get("pixel_values", [])],
            "pixel_values_videos": [
                vid for s in samples for vid in s.get("pixel_values_videos", [])
            ],
        }
        return merged

    def add_sample(self, sample: dict[str, Any]) -> None:
        """Add a sample to the buffer. Triggers packing when buffer is full."""
        sid = self._next_id
        self._next_id += 1
        self._sample_buffer[sid] = sample
        if len(self._sample_buffer) >= self.buffer_size:
            self._pack_buffered_samples()

    def flush(self) -> None:
        """Pack and yield all remaining samples, including leftovers."""
        self._pack_buffered_samples(flush=True)
