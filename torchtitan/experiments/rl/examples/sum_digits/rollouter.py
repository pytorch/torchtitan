# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.rl.examples.sum_digits.data import SumDigitsDataset
from torchtitan.experiments.rl.examples.sum_digits.env import SumDigitsEnv
from torchtitan.experiments.rl.examples.sum_digits.rubric import (
    RewardCorrect,
    RewardFormat,
)

from torchtitan.experiments.rl.rollout.rollouter import Rollouter

from torchtitan.experiments.rl.rubrics import Rubric


class SumDigitsRollouter(Rollouter):
    """The SumDigits rollouter: digit-sum train/val datasets, env, and a
    correctness + format rubric. Pure config — all behavior (`make_env_group`,
    `sample_*`, `score_group`) is inherited from `Rollouter`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: SumDigitsDataset.Config = field(
            default_factory=lambda: SumDigitsDataset.Config(seed=42)
        )
        validation_dataset: SumDigitsDataset.Config = field(
            default_factory=lambda: SumDigitsDataset.Config(seed=99)
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[
                    RewardCorrect.Config(weight=1.0),
                    RewardFormat.Config(weight=0.3),
                ]
            )
        )
        message_env: SumDigitsEnv.Config = field(default_factory=SumDigitsEnv.Config)
