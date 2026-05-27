# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from renderers import Renderer

from torchtitan.experiments.rl.envs import EnvLimits, RendererEnv
from torchtitan.experiments.rl.recipes import Task
from torchtitan.experiments.rl.recipes.sum_digits.data import SumDigitsDataset
from torchtitan.experiments.rl.recipes.sum_digits.env import SumDigitsEnv
from torchtitan.experiments.rl.recipes.sum_digits.grader import SumDigitsRubric
from torchtitan.experiments.rl.rollouts.types import DatasetOutput


class SumDigitsTask(Task):
    """SumDigits task: seeded dataset + 2-component rubric + env builder."""

    @dataclass(kw_only=True, slots=True)
    class Config(Task.Config):
        dataset: SumDigitsDataset.Config = field(
            default_factory=SumDigitsDataset.Config
        )
        rubric: SumDigitsRubric.Config = field(default_factory=SumDigitsRubric.Config)
        env_limits: EnvLimits = field(default_factory=EnvLimits)

    def __init__(self, config: Config) -> None:
        self.dataset = config.dataset.build()
        self.rubric = config.rubric.build()
        self.env_limits = config.env_limits

    def make_envs(
        self,
        *,
        example: DatasetOutput,
        group_size: int,
        renderer: Renderer,
    ) -> list[RendererEnv]:
        return [
            RendererEnv(
                message_env=SumDigitsEnv(env_input=example.env_input),
                renderer=renderer,
                limits=self.env_limits,
            )
            for _ in range(group_size)
        ]
