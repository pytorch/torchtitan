# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from dataclasses import dataclass

from renderers import Renderer

from torchtitan.config import Configurable
from torchtitan.experiments.rl.env_types import RendererWrapperEnv
from torchtitan.experiments.rl.rollouts.types import Rollout
from torchtitan.experiments.rl.rubrics import Reward, Rubric


# TODO(continuous-batching): when VLLMGenerator gains continuous batching,
# move the rollout loop onto Task as `run_rollout(example, client) -> Rollout`,
# so each rollout drives its own generate calls.
class Task(Configurable, abc.ABC):
    """Bundles everything needed to run and score one rollout: a dataset,
    how to build its envs, and a `Rubric`.

    The flow for one prompt group:

        sample  = task.sample_train_example()                   # the env input from the task's dataset
        envs    = task.make_envs(sample, group_size, renderer)  # MessageEnvs wrapped in RendererWrapperEnv
        run_rollout_fn(sampler, envs)                           # the controller runs the rollout loop
        rewards = task.score_group(rollouts, sample.env_input)  # the Rubric scores them

    `MessageEnv` works in messages; `RendererWrapperEnv` (what `make_envs` returns)
    adds the message <-> token plumbing. `score_group` defaults to per-rollout
    `rubric.score_group`; override it for cross-sibling scoring.

    Example:
        class MyTask(Task):
            @dataclass(kw_only=True, slots=True)
            class Config(Task.Config):
                train_dataset: MyDataset.Config = field(default_factory=MyDataset.Config)
                val_dataset: MyDataset.Config = field(default_factory=MyDataset.Config)
                rubric: MyRubric.Config = field(default_factory=MyRubric.Config)
                env_config: RendererWrapperEnv.Config = field(
                    default_factory=RendererWrapperEnv.Config
                )

            def __init__(self, config: Config) -> None:
                super().__init__(config)  # builds self.rubric from config.rubric
                self._train = config.train_dataset.build()
                self._val = config.val_dataset.build()
                self._env_config = config.env_config

            def sample_train_example(self) -> MyInput:
                return self._train.sample_example()

            def sample_val_example(self) -> MyInput:
                return self._val.sample_example()

            def make_envs(self, *, example, group_size, renderer):
                return [RendererWrapperEnv(...) for _ in range(group_size)]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rubric: Rubric.Config

    rubric: Rubric
    """Built from `config.rubric` by the base `__init__`; used by `score_group`."""

    def __init__(self, config: Config) -> None:
        self.rubric = config.rubric.build()

    @abc.abstractmethod
    def sample_train_example(self) -> object:
        """Sample one training example (the env input) from this task's dataset."""

    @abc.abstractmethod
    def sample_val_example(self) -> object:
        """Sample one validation example (the env input) from this task's dataset."""

    # TODO: revisit the Renderer being injected into `make_envs` once we
    # know whether Task should own a Renderer (per-task chat templates).
    @abc.abstractmethod
    def make_envs(
        self,
        *,
        example: object,
        group_size: int,
        renderer: Renderer,
    ) -> list[RendererWrapperEnv]:
        """Construct `group_size` single-use envs from one dataset example.

        Args:
            example: the env input from `sample_train_example` / `sample_val_example`.
            group_size: number of sibling envs for this prompt group.
            renderer: Renderer shared by the rollout controller.

        Returns:
            `group_size` `RendererWrapperEnv` instances, each ready for one rollout.
        """

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
