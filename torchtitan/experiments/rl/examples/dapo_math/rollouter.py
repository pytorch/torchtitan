# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.dapo_math.data import (
    AIME2025Dataset,
    DapoMathDataset,
)
from torchtitan.experiments.rl.examples.dapo_math.env import DapoMathEnv
from torchtitan.experiments.rl.examples.dapo_math.rubric import RewardMathVerify
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rubrics import Rubric


class DapoMathRollouter(Rollouter):
    """Provides DAPO-Math training with AIME 2025 validation and math rewards."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: DapoMathDataset.Config = field(
            default_factory=DapoMathDataset.Config
        )
        validation_dataset: AIME2025Dataset.Config = field(
            default_factory=AIME2025Dataset.Config
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardMathVerify.Config(weight=1.0)],
                error_reward=0.0,
            )
        )
        message_env: DapoMathEnv.Config = field(default_factory=DapoMathEnv.Config)
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(
                max_rollout_tokens=10240,
                max_num_turns=1,
            )
        )
        advantage: AdvantageEstimator.Config = field(
            default_factory=lambda: AdvantageEstimator.Config(
                should_std_normalize=False
            )
        )
