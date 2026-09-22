# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Strict packed-text inspection and attention cost estimation."""

from dataclasses import dataclass

import torch

from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class PackedTextMicrobatchMetadata:
    """Validated structural metadata for one packed text microbatch."""

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
        return sum(length * length for length in metadata.segment_lengths)


class TokenizedTextPackingAdapter(Configurable):
    """Inspect canonical output produced by the built-in text data pipeline."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        num_tokens_per_microbatch: int,
        max_context_length: int,
        max_num_documents: int | None,
        expect_pinned_memory: bool,
    ) -> None:
        del config
        if num_tokens_per_microbatch <= 0:
            raise ValueError("num_tokens_per_microbatch must be greater than 0")
        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")
        if max_num_documents is not None and max_num_documents <= 0:
            raise ValueError("max_num_documents must be greater than 0")
        self._num_tokens_per_microbatch = num_tokens_per_microbatch
        self._max_context_length = max_context_length
        self._max_num_documents = max_num_documents
        self._expect_pinned_memory = expect_pinned_memory

    def inspect_microbatch(
        self, microbatch: TrainingMicrobatch
    ) -> PackedTextMicrobatchMetadata:
        """Validate and describe one canonical tokenized text microbatch."""
        if type(microbatch) is not TokenizedTrainingMicrobatch:
            raise ValueError(
                "load balancing requires a standard TokenizedTrainingMicrobatch"
            )

        tensors = {
            "input": microbatch.input,
            "labels": microbatch.labels,
            "positions": microbatch.positions,
            "padding_mask": microbatch.padding_mask,
        }
        for name, tensor in tensors.items():
            expected_dtype = torch.bool if name == "padding_mask" else torch.int64
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must be on CPU")
            if tensor.dtype != expected_dtype:
                raise ValueError(f"{name} must have dtype {expected_dtype}")
            expected_shape = (self._num_tokens_per_microbatch,)
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
            if tensor.is_pinned() != self._expect_pinned_memory:
                raise ValueError(
                    f"{name} pinned-memory state does not match the text collator"
                )

        if microbatch.model_kwargs:
            raise ValueError("model_kwargs must be empty for packed text balancing")

        padding = microbatch.padding_mask.tolist()
        try:
            num_non_padding_tokens = padding.index(True)
        except ValueError:
            num_non_padding_tokens = len(padding)
        if any(padding[:num_non_padding_tokens]) or not all(
            padding[num_non_padding_tokens:]
        ):
            raise ValueError("padding_mask must be one trailing suffix")
        if num_non_padding_tokens == 0:
            raise ValueError(
                "a packed microbatch must contain at least one non-padding token"
            )

        padding_slice = slice(num_non_padding_tokens, None)
        if torch.count_nonzero(microbatch.input[padding_slice]).item() != 0:
            raise ValueError("padding input tokens must be zero")
        if not torch.all(microbatch.labels[padding_slice] == IGNORE_INDEX).item():
            raise ValueError("padding labels must equal IGNORE_INDEX")
        expected_padding_positions = (
            torch.arange(
                self._num_tokens_per_microbatch - num_non_padding_tokens,
                dtype=torch.int64,
            )
            % self._max_context_length
        )
        if not torch.equal(
            microbatch.positions[padding_slice], expected_padding_positions
        ):
            raise ValueError("padding positions are not canonical")

        segment_lengths = self._segment_lengths(
            microbatch.positions[:num_non_padding_tokens]
        )
        if (
            self._max_num_documents is not None
            and len(segment_lengths) > self._max_num_documents
        ):
            raise ValueError(
                f"{len(segment_lengths)} documents exceeds "
                f"max_num_documents={self._max_num_documents}"
            )

        expected_num_valid_tokens = int(
            (microbatch.labels != IGNORE_INDEX).sum().item()
        )
        if microbatch.num_valid_tokens != expected_num_valid_tokens:
            raise ValueError(
                "num_valid_tokens does not match the non-ignored label count"
            )

        return PackedTextMicrobatchMetadata(
            segment_lengths=segment_lengths,
            num_tokens=self._num_tokens_per_microbatch,
            num_non_padding_tokens=num_non_padding_tokens,
            payload_bytes=sum(
                tensor.numel() * tensor.element_size() for tensor in tensors.values()
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

            if current_length > self._max_context_length:
                raise ValueError("a document segment exceeds max_context_length")

        if current_length:
            segment_lengths.append(current_length)
        return tuple(segment_lengths)
