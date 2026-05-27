# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from dataclasses import dataclass

from torchtitan.experiments.rl.envs import DatasetOutput


@dataclass(frozen=True, kw_only=True, slots=True)
class SumDigitsInput:
    """Typed payload for one SumDigits problem.

    Args:
        numbers: 2-4 two-digit numbers the model must digit-sum.
        target: Ground-truth total digit sum.
    """

    numbers: list[int]
    target: int


class SumDigitsDataset:
    """Stateful, seeded RNG dataset of SumDigits problems.

    Example:

        ds = SumDigitsDataset(seed=42)
        ex = ds.sample_example()
        # ex.env_name == "sum_digits"
        # ex.env_input is a SumDigitsInput
    """

    ENV_NAME = "sum_digits"

    def __init__(self, *, seed: int) -> None:
        self._rng = random.Random(seed)

    def sample_example(self) -> DatasetOutput:
        n = self._rng.randint(2, 4)
        numbers = [self._rng.randint(10, 99) for _ in range(n)]
        target = sum(int(d) for num in numbers for d in str(num))
        return DatasetOutput(
            env_name=self.ENV_NAME,
            env_input=SumDigitsInput(numbers=numbers, target=target),
        )
