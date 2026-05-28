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
from torchtitan.experiments.rl.rollouts.types import DatasetOutput, Rollout
from torchtitan.experiments.rl.rubrics import Rubric


# TODO: investigate whether Task should also hold its own dataset (Camp A,
# verifiers-shape) instead of dataset living on RLTrainer (Camp B,
# NeMo-RL-shape). Today the dataset lives on RLTrainer.Config and rows
# are routed to a Task by `DatasetOutput.task`. See `60_concrete_options.md`.
class Task(Configurable):
    """Per-task bundle: rubric + env construction + group scoring.

    The controller owns the dataset and the rollout loop (including the
    per-group step). Task contributes the per-task pieces: how to build
    envs for one example, and how to score one group's rollouts.

    Subclass and set `self.rubric` / `self.env_limits` in `__init__`, then
    implement `make_envs`. `score_group` has a default impl that delegates
    to `self.rubric.score_group` and fills `reward` / `reward_components`
    on each rollout.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Empty base; concrete tasks add fields."""

    rubric: Rubric
    env_limits: EnvLimits

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
        *,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[Rollout]:
        """Score one group's rollouts and fill reward + reward_components.

        Default impl delegates to `self.rubric.score_group`. Override for
        cross-sibling scoring (judge, pairwise, diversity) or partial-credit
        reward shaping.

        Args:
            rollouts: Sibling rollouts in one prompt group, already stepped.
            env_input: Dataset payload shared by the group.

        Returns:
            The same rollouts with `reward` and `reward_components` filled,
            in input order.
        """
        scored = await self.rubric.score_group(rollouts, env_input)
        for rollout, reward in zip(rollouts, scored, strict=True):
            rollout.reward = reward.reward
            rollout.reward_components = reward.components
        return rollouts

    # TODO(continuous-batching): when VLLMGenerator gains continuous batching,
    # add `do_single_rollout(example, client, renderer, sampling, group_id,
    # sample_idx) -> Rollout` and migrate `RLTrainer._run_rollouts` from
    # per-group fan-out (controller _do_group_step + task.score_group) to
    # per-rollout fan-out via `asyncio.gather` of `do_single_rollout` calls.
    # See `60_concrete_options.md` §A2.
