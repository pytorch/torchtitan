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
from torchtitan.experiments.rl.recipes.sum_digits.env import SumDigitsEnv
from torchtitan.experiments.rl.recipes.sum_digits.grader import SumDigitsRubric
from torchtitan.experiments.rl.rollouts.types import DatasetOutput


class SumDigitsTask(Task):
    """SumDigits task: 2-component rubric + env builder.

    Dataset lives on `RLTrainer.Config` (not here); rows are routed to
    this task by `DatasetOutput.task == "sum_digits"`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Task.Config):
        """Config for `SumDigitsTask`.

        Args:
            rubric: SumDigits rubric config.
            env_limits: Renderer-env operational limits.
        """

        rubric: SumDigitsRubric.Config = field(default_factory=SumDigitsRubric.Config)
        env_limits: EnvLimits = field(default_factory=EnvLimits)

    def __init__(self, config: Config) -> None:
        self.rubric = config.rubric.build()
        self.env_limits = config.env_limits

    def make_envs(
        self,
        *,
        example: DatasetOutput,
        group_size: int,
        renderer: Renderer,
    ) -> list[RendererEnv]:
        """Construct SumDigits renderer envs for one prompt group."""
        return [
            RendererEnv(
                message_env=SumDigitsEnv(env_input=example.env_input),
                renderer=renderer,
                limits=self.env_limits,
            )
            for _ in range(group_size)
        ]
