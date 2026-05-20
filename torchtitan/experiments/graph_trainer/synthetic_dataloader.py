# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic dataloader for precompilation and benchmarking.

Generates random token IDs with the correct shapes and dtypes, avoiding
the cost of initializing a real dataset (e.g. downloading from HuggingFace
or loading sixlib). Only input shapes matter for tracing/compilation, so
real data is unnecessary for precompile workflows.

Usage:
    --dataloader synthetic --dataloader.vocab_size 128256

For precompile, vocab_size is read from the model config automatically
when using the graph_trainer config_registry functions.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader


class SyntheticDataset(IterableDataset, Stateful):
    """Infinite dataset of random token ID sequences.

    Each sample is a pair of (input_dict, labels) matching the format
    produced by HuggingFaceTextDataset: input_dict contains an "input"
    tensor of shape (seq_len,) and labels is a tensor of shape (seq_len,).
    Token IDs are sampled uniformly from [0, vocab_size).
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {vocab_size}. "
                "Set --dataloader.vocab_size or use a config that sets it."
            )
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self._step = 0

    def __iter__(self):
        while True:
            # Use a different seed per step for variety, but this is
            # synthetic data so exact reproducibility is not critical.
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            labels = torch.randint(0, self.vocab_size, (self.seq_len,))
            self._step += 1
            yield {"input": input_ids}, labels

    def state_dict(self) -> dict[str, Any]:
        return {"step": self._step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._step = state_dict.get("step", 0)


class SyntheticDataLoader(ParallelAwareDataloader):
    """Dataloader that generates synthetic random token sequences.

    Designed for precompilation and benchmarking where real data is
    unnecessary -- only input shapes matter for graph tracing.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "synthetic"
        """Dataset name (always 'synthetic' for this dataloader)."""

        vocab_size: int = 0
        """Vocabulary size for random token generation. Set to 0 to require
        explicit configuration (typically auto-set from the model config)."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        vocab_size = config.vocab_size
        if vocab_size <= 0:
            raise ValueError(
                "SyntheticDataLoader requires vocab_size > 0. "
                "Set --dataloader.vocab_size explicitly or use a config "
                "function that sets it from the model config."
            )

        ds = SyntheticDataset(
            vocab_size=vocab_size,
            seq_len=seq_len,
        )

        super().__init__(
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
        )
