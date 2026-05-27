# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from renderers import Renderer

from torchtitan.experiments.rl.envs import DatasetOutput, RendererEnv, RendererEnvConfig
from torchtitan.experiments.rl.recipes import Task
from torchtitan.experiments.rl.recipes.sum_digits.data import SumDigitsDataset
from torchtitan.experiments.rl.recipes.sum_digits.env import SumDigitsEnv
from torchtitan.experiments.rl.recipes.sum_digits.grader import (
    reward_correct,
    reward_format,
)
from torchtitan.experiments.rl.rubrics import Rubric


class SumDigitsTask(Task):
    """SumDigits task: seeded dataset + 2-component rubric + env builder."""

    @dataclass(kw_only=True, slots=True)
    class Config(Task.Config):
        seed: int = 42
        correctness_weight: float = 1.0
        format_weight: float = 0.3
        renderer_env_config: RendererEnvConfig = field(
            default_factory=RendererEnvConfig
        )

    def __init__(self, config: Config) -> None:
        self._config = config
        self.dataset = SumDigitsDataset(seed=config.seed)
        self.rubric = Rubric(
            funcs=[reward_correct, reward_format],
            weights=[config.correctness_weight, config.format_weight],
        )
        self.renderer_env_config = config.renderer_env_config

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
                config=self.renderer_env_config,
            )
            for _ in range(group_size)
        ]
