# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import Renderer

from torchtitan.config import Configurable
from torchtitan.experiments.rl.envs.renderer_env import RendererEnv, RendererEnvConfig
from torchtitan.experiments.rl.envs.types import DatasetOutput
from torchtitan.experiments.rl.rubrics import Rubric


class Task(Configurable):
    """Per-task bundle: dataset + rubric + env construction.

    Holds the components; the controller accesses fields directly
    (``task.dataset``, ``task.rubric``, ``task.renderer_env_config``).
    Scoring is owned by ``task.rubric.score_group``; Task has no
    scoring method.

    Subclass and set ``self.dataset`` / ``self.rubric`` /
    ``self.renderer_env_config`` in ``__init__``, then implement
    ``make_envs``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Empty base; concrete tasks add fields."""

    dataset: object
    rubric: Rubric
    renderer_env_config: RendererEnvConfig

    def make_envs(
        self,
        *,
        example: DatasetOutput,
        group_size: int,
        renderer: Renderer,
    ) -> list[RendererEnv]:
        """Construct ``group_size`` single-use envs from one example."""
        raise NotImplementedError
