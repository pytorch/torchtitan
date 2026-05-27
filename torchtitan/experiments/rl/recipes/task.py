# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import Renderer

from torchtitan.config import Configurable
from torchtitan.experiments.rl.envs.renderer_env import EnvLimits, RendererEnv
from torchtitan.experiments.rl.rollouts.types import DatasetOutput
from torchtitan.experiments.rl.rubrics import Rubric


class Task(Configurable):
    """Per-task bundle: dataset + rubric + env construction.

    Holds the components; the controller accesses fields directly
    (``task.dataset``, ``task.rubric``, ``task.env_limits``).
    Scoring is owned by ``task.rubric.score_group``; Task has no
    scoring method.

    Subclass and set ``self.dataset`` / ``self.rubric`` /
    ``self.env_limits`` in ``__init__``, then implement
    ``make_envs``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Empty base; concrete tasks add fields."""

    dataset: object
    rubric: Rubric
    env_limits: EnvLimits

    # TODO: evaluate whether the Renderer should be owned by the Task instead
    # of by the controller. Today RLTrainer builds renderer from hf_assets_path
    # and threads it through every make_envs call. Moving it to Task would let
    # different tasks use different renderers (multi-task mixing) but couples
    # Task construction to model-asset paths.
    def make_envs(
        self,
        *,
        example: DatasetOutput,
        group_size: int,
        renderer: Renderer,
    ) -> list[RendererEnv]:
        """Construct ``group_size`` single-use envs from one example."""
        raise NotImplementedError
