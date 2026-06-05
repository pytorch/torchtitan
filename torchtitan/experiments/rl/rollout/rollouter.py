# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from renderers import Renderer

from torchtitan.config import Configurable
from torchtitan.experiments.rl.environment import MessageEnv, TokenEnv
from torchtitan.experiments.rl.rollout.types import Rollout
from torchtitan.experiments.rl.rubrics import Rubric, RubricOutput


# TODO(continuous-batching): when VLLMGenerator gains continuous batching,
# move the rollout loop onto Rollouter as `run_rollout(example, client) -> Rollout`,
# so each rollout drives its own generate calls.
class Rollouter(Configurable):
    """Turns a problem (train/val datasets, the `MessageEnv` to build per example, and a
    `Rubric`) into scored rollouts — the RL training data.

    Like a `Dataloader` turns a `Dataset` into training batches, a `Rollouter`
    turns a problem into rollouts: it builds the envs, the controller drives them against
    the inference engine, and `score_group` scores the results.

    Subclass only to override specific methods, such as `score_group` for cross-sibling scoring,
    or `make_env_group` for custom logic, such as using a pool of envs instead of creating a new one.

    The flow for one prompt group:

        sample = rollouter.get_training_sample()  # one sample from the dataset
        envs = rollouter.make_env_group(sample=sample, group_size=N, renderer=renderer)
        ...  # the controller runs the rollout loop
        outputs = rollouter.score_group(rollouts, sample)  # the Rubric scores them

    `MessageEnv` works in messages; `TokenEnv` (what `make_env_group` returns)
    adds the message <-> token plumbing.

    Example:
        rollouter = Rollouter.Config(
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
        """The env to build per sample; `make_env_group` calls `build(env_input=sample)`."""

        token_env: TokenEnv.Config = field(default_factory=TokenEnv.Config)
        """`TokenEnv` limits (e.g. `max_rollout_tokens`) passed to `make_env_group`."""

    def __init__(self, config: Config) -> None:
        self._train_dataset = config.train_dataset.build()
        self._validation_dataset = config.validation_dataset.build()
        self.rubric: Rubric = config.rubric.build()
        self._message_env_config = config.message_env
        self._token_env_config = config.token_env

    # TODO: revisit this abstraction: should it return a sample or a dataset or an iterator?
    def get_training_sample(self) -> object:
        """Get one training sample (the env input) from the training dataset."""
        return next(self._train_dataset)

    def get_validation_sample(self) -> object:
        """Get one validation sample (the env input) from the validation dataset."""
        return next(self._validation_dataset)

    # TODO: revisit the Renderer being injected into `make_env_group` once we
    # know whether Rollouter should own a Renderer (per-rollouter chat templates).
    def make_env_group(
        self,
        *,
        sample: object,
        group_size: int,
        renderer: Renderer,
    ) -> list[TokenEnv]:
        """Construct `group_size` single-use envs from one dataset sample.

        Args:
            sample: the dataset sample (the env input) from `get_training_sample` / `get_validation_sample`.
            group_size: number of sibling envs for this prompt group.
            renderer: Renderer shared by the rollout controller.

        Returns:
            `TokenEnv` * `group_size` instances, each ready for one rollout.
        """
        return [
            self._token_env_config.build(
                message_env=self._message_env_config.build(env_input=sample),
                renderer=renderer,
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
