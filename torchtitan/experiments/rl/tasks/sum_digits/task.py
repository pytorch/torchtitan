# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from renderers import Renderer

from torchtitan.experiments.rl.env_types import RendererWrapperEnv
from torchtitan.experiments.rl.rubrics import Rubric
from torchtitan.experiments.rl.tasks import Task
from torchtitan.experiments.rl.tasks.sum_digits.data import (
    SumDigitsDataset,
    SumDigitsInput,
)
from torchtitan.experiments.rl.tasks.sum_digits.env import SumDigitsEnv
from torchtitan.experiments.rl.tasks.sum_digits.rubric import (
    RewardCorrect,
    RewardFormat,
)


class SumDigitsTask(Task):
    """SumDigits task: have the model sum a sequence of digits."""

    @dataclass(kw_only=True, slots=True)
    class Config(Task.Config):
        train_dataset: SumDigitsDataset.Config = field(
            default_factory=lambda: SumDigitsDataset.Config(seed=42)
        )
        val_dataset: SumDigitsDataset.Config = field(
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
        env_config: RendererWrapperEnv.Config = field(
            default_factory=RendererWrapperEnv.Config
        )
        """Renderer-wrapper limits, e.g. `max_rollout_tokens`."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)  # builds self.rubric from config.rubric
        self._train_dataset = config.train_dataset.build()
        self._val_dataset = config.val_dataset.build()
        self._env_config = config.env_config

    def sample_train_example(self) -> SumDigitsInput:
        return self._train_dataset.sample_example()

    def sample_val_example(self) -> SumDigitsInput:
        return self._val_dataset.sample_example()

    def make_envs(
        self,
        *,
        example: SumDigitsInput,
        group_size: int,
        renderer: Renderer,
    ) -> list[RendererWrapperEnv]:
        """Construct SumDigits envs for one prompt group."""
        return [
            RendererWrapperEnv(
                message_env=SumDigitsEnv(env_input=example),
                renderer=renderer,
                config=self._env_config,
            )
            for _ in range(group_size)
        ]
