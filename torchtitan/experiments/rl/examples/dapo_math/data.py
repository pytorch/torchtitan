# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset

from torchtitan.config import Configurable


_AIME_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


@dataclass(frozen=True, kw_only=True, slots=True)
class DapoMathSample:
    """A math prompt paired with its expected final answer."""

    prompt: str
    ground_truth: str


# TODO: Share this cycling iterator with other RL datasets instead of keeping
# per-environment implementations.
class _CyclingDataset(Configurable):
    """Provides an endless, resumable stream over a finite sample list."""

    def __init__(
        self,
        samples: list[DapoMathSample],
        *,
        seed: int,
        shuffle: bool,
    ) -> None:
        if not samples:
            raise ValueError("math dataset must contain at least one sample")
        self._samples = samples
        self._rng = random.Random(seed)
        self._shuffle = shuffle
        self._order = list(range(len(samples)))
        if shuffle:
            self._rng.shuffle(self._order)
        self._position = 0

    def __iter__(self) -> Iterator[DapoMathSample]:
        return self

    def __next__(self) -> DapoMathSample:
        if self._position == len(self._order):
            # Rollout production consumes an endless stream; crossing the dataset
            # boundary starts a new epoch. Training reshuffles; validation does not.
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._position = 0
        sample_index = self._order[self._position]
        self._position += 1
        return self._samples[sample_index]

    def state_dict(self) -> dict:
        """Snapshot row order and position so resume continues the same stream."""
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "position": self._position,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state returned by `state_dict`."""
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._position = state_dict["position"]


class DapoMathDataset(_CyclingDataset):
    """Provides filtered DAPO-Math problems in the original `Answer:` format."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        repo_id: str = "hamishivi/DAPO-Math-17k-Processed_filtered"
        split: str = "train"
        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        dataset = load_dataset(config.repo_id, split=config.split)
        samples: list[DapoMathSample] = []
        for row in dataset:
            prompt_messages = row["source_prompt"]
            if len(prompt_messages) != 1 or prompt_messages[0]["role"] != "user":
                raise ValueError("DAPO-Math rows must contain exactly one user prompt")
            samples.append(
                DapoMathSample(
                    prompt=prompt_messages[0]["content"],
                    ground_truth=str(row["ground_truth"]),
                )
            )
        super().__init__(samples, seed=config.seed, shuffle=config.shuffle)


class AIME2025Dataset(_CyclingDataset):
    """Provides AIME 2025 I+II problems using the DAPO answer format."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        repo_id: str = "opencompass/AIME2025"
        subsets: tuple[str, ...] = ("AIME2025-I", "AIME2025-II")
        split: str = "test"
        seed: int = 99
        shuffle: bool = False

    def __init__(self, config: Config) -> None:
        samples: list[DapoMathSample] = []
        for subset in config.subsets:
            dataset = load_dataset(config.repo_id, subset, split=config.split)
            samples.extend(
                DapoMathSample(
                    prompt=_AIME_PROMPT_TEMPLATE.format(problem=row["question"]),
                    ground_truth=str(row["answer"]),
                )
                for row in dataset
            )
        super().__init__(samples, seed=config.seed, shuffle=config.shuffle)
