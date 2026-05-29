# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import asyncio
import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollouts.types import Rollout
from torchtitan.observability import structured_logger as sl


@dataclass(frozen=True, kw_only=True, slots=True)
class RewardFn:
    """One reward function + its weight in the rubric's weighted sum."""

    fn: Callable[[Rollout, object], Awaitable[float]]
    """Async reward callable with shape `async (rollout, env_input) -> float`."""

    weight: float = 1.0
    """Unnormalized weight for reward averaging. The rubric normalizes weights the sum to 1.0."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Reward:
    """One rollout's reward + per-reward-fn output breakdown.

    Example:
        >>> Reward(reward=0.5, components={"reward_fn1": 0.2, "reward_fn2": 0.3})
    """

    reward: float
    """Final weighted reward for this rollout."""

    components: dict[str, float] = field(default_factory=dict)
    """Per-reward-fn raw output, keyed by `fn.__name__`."""


class Rubric(Configurable, abc.ABC):
    """Holds and calls reward functions.

    Subclass and implement `register_funcs` returning a list of `RewardFn`.
    Weights are normalized to sum to 1.0 across the list.

    `Config.truncation_reward` / `Config.error_reward` override the
    weighted-sum path on non-COMPLETED rollouts when set. When `None`, reward
    fns run on the partial response and can inspect `rollout.status`.

    Example:
        class MyRubric(Rubric):
            def register_funcs(self) -> list[RewardFn]:
                return [
                    RewardFn(fn=my_reward_fn1, weight=0.5),
                    RewardFn(fn=my_reward_fn2, weight=0.5),
                ]

        rubric = MyRubric(config=MyRubric.Config(truncation_reward=0.0))
        rewards = await rubric.score_group(my_rollouts, env_input)
        for reward, rollout in zip(rewards, my_rollouts):
            my_rollout.reward = reward.reward
            my_rollout.reward_components = reward.components
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):

        truncation_reward: float | None = None
        """Skip reward fns on truncated rollouts and returns this value instead.
        If `None`, reward fns run normally."""

        error_reward: float | None = None
        """Skip reward fns on errored rollouts and returns this value instead.
        If `None`, reward fns run normally."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Rubric.Config()

    @abc.abstractmethod
    def register_funcs(self) -> list[RewardFn]:
        """Return the reward fns + weights for this rubric.

        Example:
            class MyRubric(Rubric):
                def register_funcs(self) -> list[RewardFn]:
                    return [
                        RewardFn(fn=my_reward_fn1, weight=0.5),
                        RewardFn(fn=my_reward_fn2, weight=0.5),
                    ]
        """

    @functools.cached_property
    def _rwd_fns(self) -> list[RewardFn]:
        """Builds registered reward fns with weights normalized to sum
        to 1.0 on first scoring call (lazily)."""
        registered = self.register_funcs()

        # Sanity checks
        if not registered:
            raise ValueError("register_funcs returned no reward fns")

        names = [_fn_name(rwd_fn.fn) for rwd_fn in registered]
        if len(names) != len(set(names)):
            raise ValueError(f"reward fn names must be unique; got {names}")

        total_weight = sum(rwd_fn.weight for rwd_fn in registered)
        if total_weight <= 0:
            raise ValueError(
                f"rubric weights must sum to a positive value; got {total_weight}"
            )

        # Normalize weights
        return [
            RewardFn(fn=rwd_fn.fn, weight=rwd_fn.weight / total_weight)
            for rwd_fn in registered
        ]

    @sl.log_trace_span("score_one")
    async def _score_one(self, rollout: Rollout, env_input: object) -> Reward:
        """Score one rollout. Short-circuits on truncate and error if
        `Config.truncation_reward` / `error_reward` are set.

        Args:
            rollout: Rollout to score.
            env_input: Dataset payload originally used to construct the rollout env.
                It can contain valuable info for the reward, such as the target or metadata.

        Returns:
            Final weighted reward + per-fn raw components. S
        """
        # Short-circuit on truncate / error
        cfg = self._config
        if cfg.truncation_reward is not None and rollout.status.is_truncated():
            return Reward(
                reward=cfg.truncation_reward,
                components={"truncated": cfg.truncation_reward},
            )
        if cfg.error_reward is not None and rollout.status.is_error():
            return Reward(
                reward=cfg.error_reward,
                components={"errored": cfg.error_reward},
            )

        # Run all reward fns and weight-sum
        per_fn_rewards = await asyncio.gather(
            *(rwd_fn.fn(rollout, env_input) for rwd_fn in self._rwd_fns)
        )

        components = {}
        total_reward = 0.0
        for rwd_fn, r in zip(self._rwd_fns, per_fn_rewards, strict=True):
            fn_name = _fn_name(rwd_fn.fn)
            components[fn_name] = r
            total_reward += rwd_fn.weight * r

        return Reward(reward=total_reward, components=components)

    @sl.log_trace_span("score_group")
    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[Reward]:
        """Score every rollout in one prompt group.

        Override for cross-rollout rewards (pairwise comparison, diversity,
        rank normalization).

        Args:
            rollouts: Sibling rollouts sampled from one prompt group.
            env_input: Dataset payload originally used to construct the rollout env.
                It can contain valuable info for the reward, such as the target or metadata.

        Returns:
            One `Reward` per rollout, in input order.
        """
        return await asyncio.gather(*(self._score_one(r, env_input) for r in rollouts))


def _fn_name(fn: Callable) -> str:
    """Plain fns have `__name__`; callable instances use the class name."""
    return getattr(fn, "__name__", None) or type(fn).__name__
