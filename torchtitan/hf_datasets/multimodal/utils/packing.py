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

from torchtitan.tools.logging import logger


class MMSamplePacker:
    """Packs multiple samples to maximize sequence length utilization.

    Samples are accumulated in an internal buffer.  When the buffer reaches
    ``buffer_size``, a scan-and-pick pass packs samples into sequences of up
    to ``max_seq_length``.  Samples that cannot form a complete sequence stay
    in the buffer to be combined with future arrivals (unless ``flush=True``).
    """

    def __init__(
        self,
        max_seq_length: int,
        buffer_size: int = 100,
        batch_size: int = 8,
    ):
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self._sample_buffer: dict[int, dict[str, Any]] = {}
        self._next_id: int = 0
        self.packed_samples: deque = deque()

    def _pack_buffered_samples(self, flush: bool = False) -> None:
        """Pack buffered samples into sequences using scan-and-pick.

        Repeatedly scans the buffer to greedily fill each packed sequence.
        When ``flush=False``, an incomplete sequence stays in the buffer
        for future combination.

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

            # Nothing fit — every remaining sample exceeds max_seq_length
            if not picked_ids:
                logger.warning(
                    f"Dropping {len(self._sample_buffer)} samples that"
                    f" exceed max_seq_length {self.max_seq_length}"
                )
                self._sample_buffer.clear()
                break

            # Incomplete sequence — keep in buffer for future samples
            if not flush and current_length < self.max_seq_length:
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

    def has_batch_ready(self) -> bool:
        return len(self.packed_samples) >= self.batch_size

    def get_next_batch(self) -> list[dict[str, Any]] | None:
        """Get next batch of packed samples if a full batch is available."""
        if not self.has_batch_ready():
            return None
        return [self.packed_samples.popleft() for _ in range(self.batch_size)]

    def flush(self) -> None:
        """Pack and yield all remaining samples, including leftovers."""
        self._pack_buffered_samples(flush=True)
