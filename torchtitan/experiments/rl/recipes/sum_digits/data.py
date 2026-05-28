# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollouts.types import DatasetOutput


@dataclass(frozen=True, kw_only=True, slots=True)
class SumDigitsInput:
    """Typed payload for one SumDigits problem.

    Args:
        numbers: Two to four two-digit numbers the model must digit-sum.
        target: Ground-truth total digit sum.
    """

    numbers: list[int]  # [N_numbers]
    target: int


class SumDigitsDataset(Configurable):
    """Stateful, seeded RNG dataset of SumDigits problems.

    Example:

        ds = SumDigitsDataset(SumDigitsDataset.Config(seed=42))
        ex = ds.sample_example()
        # ex.task == "sum_digits"
        # ex.env_input is a SumDigitsInput
    """

    TASK_NAME = "sum_digits"

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Config for `SumDigitsDataset`.

        Args:
            seed: Seed for the per-row RNG.
        """

        seed: int = 42

    def __init__(self, config: Config) -> None:
        self._rng = random.Random(config.seed)

    def sample_example(self) -> DatasetOutput:
        """Sample one SumDigits problem.

        Returns:
            Dataset row whose `env_input` is a `SumDigitsInput`. `task` is
            the constant `"sum_digits"` so the controller routes this row
            to a `SumDigitsTask` in its task map.
        """
        n = self._rng.randint(2, 4)
        numbers = [self._rng.randint(10, 99) for _ in range(n)]
        target = sum(int(d) for num in numbers for d in str(num))
        return DatasetOutput(
            task=self.TASK_NAME,
            env_input=SumDigitsInput(numbers=numbers, target=target),
        )
