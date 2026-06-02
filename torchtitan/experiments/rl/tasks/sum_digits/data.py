# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from dataclasses import dataclass

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class SumDigitsInput:
    """Typed payload for one SumDigits problem."""

    numbers: list[int]  # [N_numbers]
    """Numbers the model must digit-sum."""

    target: int
    """Ground-truth total digit sum."""


class SumDigitsDataset(Configurable):
    """Stateful, seeded RNG dataset of SumDigits problems.

    Example:

        ds = SumDigitsDataset(SumDigitsDataset.Config(seed=42))
        ex = ds.sample_example()
        # ex.env_input is a SumDigitsInput
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 42

    def __init__(self, config: Config) -> None:
        self._rng = random.Random(config.seed)

    def sample_example(self) -> SumDigitsInput:
        """Sample one SumDigits problem."""
        n = self._rng.randint(2, 4)
        numbers = [self._rng.randint(10, 99) for _ in range(n)]
        target = sum(int(d) for num in numbers for d in str(num))
        return SumDigitsInput(numbers=numbers, target=target)
