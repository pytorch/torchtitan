# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from renderers import Renderer

from torchtitan.config import Configurable
from torchtitan.experiments.rl.env_types import MessageEnv, RendererWrapperEnv
from torchtitan.experiments.rl.rollouts.types import Rollout
from torchtitan.experiments.rl.rubrics import Rubric, RubricOutput


# TODO(continuous-batching): when VLLMGenerator gains continuous batching,
# move the rollout loop onto Task as `run_rollout(example, client) -> Rollout`,
# so each rollout drives its own generate calls.
class Task(Configurable):
    """Bundles everything to run and score one prompt group: train and validation
    datasets, the `MessageEnv` to build per example, and a `Rubric`.

    Subclass only to override specific methods, such as `score_group` for cross-sibling scoring,
    or `make_envs` for custom logic, such as using a pool of envs instead of creating a new one.

    The flow for one prompt group:

        example = task.sample_train_example()       # one env input from the dataset
        envs    = task.make_envs(example=example, group_size=N, renderer=renderer)
        ...                                          # the controller runs the rollout loop
        outputs = task.score_group(rollouts, example)  # the Rubric scores them

    `MessageEnv` works in messages; `RendererWrapperEnv` (what `make_envs` returns)
    adds the message <-> token plumbing.

    Example:
        task = Task.Config(
            train_dataset=MyDataset.Config(seed=42),
            validation_dataset=MyDataset.Config(seed=99),
            rubric=Rubric.Config(
                reward_fns=[RewardCorrect.Config(), RewardFormat.Config(weight=0.3)]
            ),
            message_env=MyEnv.Config(),
        ).build()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        train_dataset: Configurable.Config
        """Dataset iterator for training (`next()` yields one env input)."""

        validation_dataset: Configurable.Config
        """Dataset iterator for validation."""

        rubric: Rubric.Config
        """Reward functions + weights used by `score_group`."""

        message_env: MessageEnv.Config
        """The env to build per example; `make_envs` calls `build(env_input=example)`."""

        env_wrapper_cfg: RendererWrapperEnv.Config = field(
            default_factory=RendererWrapperEnv.Config
        )
        """Renderer-wrapper limits (e.g. `max_rollout_tokens`) passed to `make_envs`."""

    rubric: Rubric
    """Built from `config.rubric` by `__init__`; used by `score_group`."""

    def __init__(self, config: Config) -> None:
        self._train_dataset = config.train_dataset.build()
        self._validation_dataset = config.validation_dataset.build()
        self.rubric = config.rubric.build()
        self._message_env_config = config.message_env
        self._env_wrapper_cfg = config.env_wrapper_cfg

    # TODO: revisit this abstraction: should it return a sample or a dataset or an iterator?
    def sample_train_example(self) -> object:
        """Sample one training example (the env input) from the train dataset."""
        return next(self._train_dataset)

    def sample_validation_example(self) -> object:
        """Sample one validation example (the env input) from the validation dataset."""
        return next(self._validation_dataset)

    # TODO: revisit the Renderer being injected into `make_envs` once we
    # know whether Task should own a Renderer (per-task chat templates).
    def make_envs(
        self,
        *,
        example: object,
        group_size: int,
        renderer: Renderer,
    ) -> list[RendererWrapperEnv]:
        """Construct `group_size` single-use envs from one dataset example.

        Args:
            example: the env input from `sample_train_example` / `sample_validation_example`.
            group_size: number of sibling envs for this prompt group.
            renderer: Renderer shared by the rollout controller.

        Returns:
            `group_size` `RendererWrapperEnv` instances, each ready for one rollout.
        """
        return [
            RendererWrapperEnv(
                message_env=self._message_env_config.build(env_input=example),
                renderer=renderer,
                config=self._env_wrapper_cfg,
            )
            for _ in range(group_size)
        ]

    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[RubricOutput]:
        """Score one group's rollouts; the controller applies the rewards.

        Default impl delegates to `self.rubric.score_group`. Override for
        cross-sibling scoring (judge, pairwise, diversity) or partial-credit
        reward shaping.

        Args:
            rollouts: Sibling rollouts in one prompt group, already stepped.
            env_input: Dataset payload shared by the group.

        Returns:
            One `RubricOutput` per rollout, in input order.
        """
        return await self.rubric.score_group(rollouts, env_input)
