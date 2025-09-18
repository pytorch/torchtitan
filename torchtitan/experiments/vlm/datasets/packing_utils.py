# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for efficient sample packing in multimodal datasets."""

from collections import deque
from typing import Any

import torch
from torchtitan.tools.logging import logger


class SamplePacker:
    """Packs multiple samples together to maximize sequence length utilization."""

    def __init__(
        self,
        max_seq_length: int,
        buffer_size: int = 100,
        batch_size: int = 8,
    ):
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Initialize buffers
        self.sample_buffer: deque = deque()
        self.packed_samples: deque = deque()

    def _pack_buffered_samples(self) -> list[dict[str, Any]]:
        """Pack buffered samples into optimal sequences."""
        if not self.sample_buffer:
            return []

        # Sort samples by length for better packing
        samples = sorted(
            self.sample_buffer, key=lambda x: len(x["input_ids"]), reverse=True
        )

        packed_sequences = []
        current_sequence = []
        current_length = 0

        for sample in samples:
            sample_length = len(sample["input_ids"])

            # Skip very long samples
            if sample_length > self.max_seq_length:
                logger.warning(
                    f"Sample length {sample_length} exceeds max_seq_length "
                    f"{self.max_seq_length}, will be skipped"
                )
                continue

            # Check if adding this sample would exceed max length
            if current_sequence and (
                current_length + sample_length > self.max_seq_length
            ):
                # Current sequence is full, create packed sample
                packed_sequences.append(
                    {
                        "input_ids": torch.cat(
                            [s["input_ids"] for s in current_sequence]
                        ),
                        "labels": torch.cat([s["labels"] for s in current_sequence]),
                        "pixel_values": [
                            img for s in current_sequence for img in s["pixel_values"]
                        ],
                    }
                )
                current_sequence = []
                current_length = 0

            # Add sample to current sequence
            current_sequence.append(sample)
            current_length += sample_length

        # Handle remaining sequence
        if current_sequence:
            packed_sequences.append(
                {
                    "input_ids": torch.cat([s["input_ids"] for s in current_sequence]),
                    "labels": torch.cat([s["labels"] for s in current_sequence]),
                    "pixel_values": [
                        img for s in current_sequence for img in s["pixel_values"]
                    ],
                }
            )

        # Clear buffer
        self.sample_buffer.clear()
        return packed_sequences

    def add_sample(self, sample: dict[str, Any]) -> None:
        """Add a sample to the buffer."""
        self.sample_buffer.append(sample)

        if len(self.sample_buffer) >= self.buffer_size:
            packed = self._pack_buffered_samples()
            self.packed_samples.extend(packed)

    def has_batch_ready(self) -> bool:
        """Check if a full batch is ready."""
        return len(self.packed_samples) >= self.batch_size

    def get_next_batch(self) -> list[dict[str, Any]] | None:
        """Get next batch of packed samples if available."""
        if not self.has_batch_ready():
            # Try to pack any remaining samples
            if self.sample_buffer:
                packed = self._pack_buffered_samples()
                self.packed_samples.extend(packed)

            if not self.has_batch_ready():
                return None

        batch = []
        for _ in range(self.batch_size):
            if not self.packed_samples:
                break
            batch.append(self.packed_samples.popleft())

        return batch
