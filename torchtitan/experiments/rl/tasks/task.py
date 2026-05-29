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
from torchtitan.experiments.rl.rollouts.types import DatasetOutput, Rollout
from torchtitan.experiments.rl.rubrics import Reward, Rubric


# TODO: investigate whether Task should also hold its own dataset
# instead of dataset living on RLTrainer.
class Task(Configurable):
    """Per-task bundle: rubric + env construction + group scoring.

    The controller owns the dataset and the rollout loop (including the
    per-group step). Task contributes the per-task pieces: how to build
    envs for one example, and how to score one group's rollouts.

    Subclass and set `self.rubric` / `self.renderer_env_config` in `__init__`,
    then implement `make_envs`. `score_group` has a default impl that delegates
    to `self.rubric.score_group` and returns one `Reward` per rollout.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Empty base; concrete tasks add fields."""

    rubric: Rubric
    renderer_env_config: RendererEnvConfig

    # TODO: revisit the Renderer being injected into `make_envs` once we
    # know whether Task should own a Renderer (per-task chat templates).
    def make_envs(
        self,
        *,
        example: DatasetOutput,
        group_size: int,
        renderer: Renderer,
    ) -> list[RendererEnv]:
        """Construct `group_size` single-use envs from one dataset example.

        Args:
            example: Dataset row sampled from the controller's dataset.
            group_size: Number of sibling envs for this prompt group.
            renderer: Renderer shared by the rollout controller.

        Returns:
            `group_size` `RendererEnv` instances, each ready for one rollout.
        """
        raise NotImplementedError

    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[Reward]:
        """Score one group's rollouts; the controller applies the rewards.

        Default impl delegates to `self.rubric.score_group`. Override for
        cross-sibling scoring (judge, pairwise, diversity) or partial-credit
        reward shaping.

        Args:
            rollouts: Sibling rollouts in one prompt group, already stepped.
            env_input: Dataset payload shared by the group.

        Returns:
            One `Reward` per rollout, in input order.
        """
        return await self.rubric.score_group(rollouts, env_input)

    # TODO(continuous-batching): when VLLMGenerator gains continuous batching,
    # move the rollout loop onto Task as `do_single_rollout(example, client)
    # -> Rollout`, so each rollout drives its own generate calls, instead of
    # the controller's batched generate + `_do_single_rollout` fan-out in
    # `RLTrainer._run_rollouts`.
