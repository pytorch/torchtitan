# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared data-pipeline types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import grain.python as grain
import torch

from torchtitan.components.tokenizer import BaseTokenizer


class TrainingMicrobatch(ABC):
    """A data-parallel-rank-local input to one training forward/backward."""

    __slots__ = ()

    labels: torch.Tensor
    num_valid_tokens: int

    @abstractmethod
    def as_input_dict(self) -> dict[str, Any]:
        """Return the model-facing representation used by ``preprocess_inputs``."""
        ...

    def to_input_dict(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> dict[str, Any]:
        """Return model inputs with top-level tensors moved to ``device``."""
        return {
            key: (
                value.to(device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in self.as_input_dict().items()
        }

    def loss_kwargs(self) -> dict[str, Any]:
        """Return additional keyword arguments for the loss function."""
        return {}

    def to_loss_kwargs(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> dict[str, Any]:
        """Return loss arguments with top-level tensors moved to ``device``."""
        return {
            key: (
                value.to(device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in self.loss_kwargs().items()
        }


@dataclass(kw_only=True, slots=True)
class TokenizedTrainingMicrobatch(TrainingMicrobatch):
    """One fixed-size, data-parallel-rank-local token microbatch.

    Each data-parallel rank receives a distinct instance. A training step may
    consume one or more of these through gradient accumulation and pipeline
    parallel microbatching.
    """

    input: torch.Tensor
    labels: torch.Tensor
    positions: torch.Tensor
    padding_mask: torch.Tensor
    num_valid_tokens: int
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def as_input_dict(self) -> dict[str, Any]:
        """Return the model-facing representation used by ``preprocess_inputs``."""
        return {
            "input": self.input,
            "labels": self.labels,
            "positions": self.positions,
            "padding_mask": self.padding_mask,
            **self.model_kwargs,
        }


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetBuildContext:
    """Runtime values shared while building the data pipeline."""

    tokenizer: BaseTokenizer
    max_context_length: int
    num_tokens_per_microbatch: int
    read_options: grain.ReadOptions
    max_num_documents: int | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetIterationPolicy:
    """Controls dataset order, repetition, and data-parallel ownership."""

    seed: int
    shuffle: bool
    repeat: bool
    dp_rank: int
    dp_world_size: int
    streaming_shuffle_buffer_size: int
