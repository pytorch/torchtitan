# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for efficient sample packing in Qwen3-VL datasets."""

from collections import deque
from typing import Any

import torch


class SamplePacker:
    """Packs multiple samples together to maximize sequence length utilization.

    Samples are accumulated in a buffer. When the buffer reaches buffer_size,
    samples are sorted by length and greedily packed into sequences up to
    max_seq_length. The last (incomplete) group of samples is kept in the
    buffer to be combined with future samples, avoiding short packed sequences.
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

        self.sample_buffer: deque = deque()  # raw samples
        self.packed_samples: deque = deque()  # packed samples ready to be yielded

    def _pack_buffered_samples(self, flush: bool = False) -> None:
        """Pack buffered samples into sequences.

        When flush=False, the last incomplete group stays in the buffer
        to be combined with future samples.
        """
        if not self.sample_buffer:
            return

        samples = sorted(
            self.sample_buffer, key=lambda x: len(x["input_ids"]), reverse=True
        )
        self.sample_buffer.clear()

        current_sequence = []
        current_length = 0

        for sample in samples:
            sample_length = len(sample["input_ids"])

            if current_sequence and (
                current_length + sample_length > self.max_seq_length
            ):
                self.packed_samples.append(self._merge_samples(current_sequence))
                current_sequence = []
                current_length = 0

            current_sequence.append(sample)
            current_length += sample_length

        if current_sequence:
            if flush:
                self.packed_samples.append(self._merge_samples(current_sequence))
            else:
                self.sample_buffer.extend(current_sequence)

    @staticmethod
    def _merge_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_ids": torch.cat([s["input_ids"] for s in samples]),
            "labels": torch.cat([s["labels"] for s in samples]),
            "pixel_values": [img for s in samples for img in s["pixel_values"]],
        }

    def add_sample(self, sample: dict[str, Any]) -> None:
        """Add a sample to the buffer. Triggers packing when buffer is full."""
        self.sample_buffer.append(sample)
        if len(self.sample_buffer) >= self.buffer_size:
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
