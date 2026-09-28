# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packed-text inspection and attention cost estimation."""

from dataclasses import dataclass

import torch

from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class PackedTextMicrobatchMetadata:
    """Structural metadata for one packed text microbatch."""

    segment_lengths: tuple[int, ...]
    num_tokens: int
    num_non_padding_tokens: int
    payload_bytes: int

    @property
    def num_documents(self) -> int:
        """Return the number of reset-delimited document segments."""
        return len(self.segment_lengths)


class QuadraticAttentionCost(Configurable):
    """Estimate full-attention work as the sum of squared segment lengths."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def estimate(self, metadata: PackedTextMicrobatchMetadata) -> int:
        """Return the additive integer cost for one packed microbatch."""
        # TODO: Include padding segments, which varlen attention executes as
        # separate sequences and can therefore affect relative batch cost.
        return sum(length * length for length in metadata.segment_lengths)


class TokenizedTextPackingAdapter(Configurable):
    """Inspect canonical output produced by the built-in text data pipeline."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def inspect_microbatch(
        self, microbatch: TrainingMicrobatch
    ) -> PackedTextMicrobatchMetadata:
        """Validate and describe one canonical tokenized text microbatch."""
        if type(microbatch) is not TokenizedTrainingMicrobatch:
            raise ValueError(
                "load balancing requires a standard TokenizedTrainingMicrobatch"
            )

        padding = microbatch.padding_mask.tolist()
        try:
            num_non_padding_tokens = padding.index(True)
        except ValueError:
            num_non_padding_tokens = len(padding)
        if not all(padding[num_non_padding_tokens:]):
            raise ValueError("padding_mask must be one trailing suffix")

        segment_lengths = self._segment_lengths(
            microbatch.positions[:num_non_padding_tokens]
        )
        payload_values = (
            microbatch.input,
            microbatch.labels,
            microbatch.positions,
            microbatch.padding_mask,
            *microbatch.model_kwargs.values(),
        )

        return PackedTextMicrobatchMetadata(
            segment_lengths=segment_lengths,
            num_tokens=microbatch.input.numel(),
            num_non_padding_tokens=num_non_padding_tokens,
            payload_bytes=sum(
                value.numel() * value.element_size()
                for value in payload_values
                if isinstance(value, torch.Tensor)
            ),
        )

    def _segment_lengths(self, real_positions: torch.Tensor) -> tuple[int, ...]:
        positions = real_positions.tolist()
        segment_lengths: list[int] = []
        current_length = 0
        for position in positions:
            if position == 0:
                if current_length:
                    segment_lengths.append(current_length)
                current_length = 1
            elif position == current_length:
                current_length += 1
            else:
                raise ValueError("real-token positions are not canonical")

        if current_length:
            segment_lengths.append(current_length)
        return tuple(segment_lengths)
