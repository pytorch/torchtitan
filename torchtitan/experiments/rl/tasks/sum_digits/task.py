# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from renderers import Renderer

from torchtitan.experiments.rl.env_types import RendererEnv, RendererEnvConfig
from torchtitan.experiments.rl.rollouts.types import DatasetOutput
from torchtitan.experiments.rl.tasks import Task
from torchtitan.experiments.rl.tasks.sum_digits.env import SumDigitsEnv
from torchtitan.experiments.rl.tasks.sum_digits.grader import SumDigitsRubric


class SumDigitsTask(Task):
    """SumDigits task: Have the model sum a sequence of digits."""

    @dataclass(kw_only=True, slots=True)
    class Config(Task.Config):
        rubric: SumDigitsRubric.Config = field(default_factory=SumDigitsRubric.Config)
        """SumDigits rubric config."""

        renderer_env_config: RendererEnvConfig = field(
            default_factory=RendererEnvConfig
        )
        """Renderer-env limits, e.g. `max_rollout_tokens`."""

    def __init__(self, config: Config) -> None:
        self.rubric = config.rubric.build()
        self.renderer_env_config = config.renderer_env_config

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
                config=self.renderer_env_config,
            )
            for _ in range(group_size)
        ]
