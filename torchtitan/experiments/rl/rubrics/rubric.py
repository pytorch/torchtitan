# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollouts.types import Rollout
from torchtitan.observability import structured_logger as sl


@dataclass(frozen=True, kw_only=True, slots=True)
class RewardFn:
    """One reward function + its weight in the rubric's weighted sum.

    Args:
        fn: Async reward callable with shape `async (rollout, env_input) -> float`.
        weight: Raw weight. The rubric normalizes weights to sum to 1.0
            across the registered list, so only the ratio matters. Default 1.0.
    """

    fn: Callable[[Rollout, object], Awaitable[float]]
    weight: float = 1.0


@dataclass(frozen=True, kw_only=True, slots=True)
class Reward:
    """One rollout's reward + per-reward-fn output breakdown.

    Args:
        reward: Final weighted reward for this rollout.
        components: Per-reward-fn raw output, keyed by `fn.__name__`.

    Example:
        >>> Reward(reward=0.5, components={"reward_fn1": 0.2, "reward_fn2": 0.3})
    """

    reward: float
    components: dict[str, float] = field(default_factory=dict)


class Rubric(Configurable):
    """Weighted sum of reward functions.

    Subclass and implement `register_funcs` returning a list of `RewardFn`.
    Weights are normalized to sum to 1.0 across the list, so the final
    reward is bounded by the per-fn output range.

    `Config.truncation_reward` / `Config.error_reward` override the
    weighted-sum path on non-COMPLETED rollouts: when set, all reward fns
    are SKIPPED and the configured value is returned. When `None`, reward
    fns run on the partial response and can inspect `rollout.status`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Terminal-status reward overrides.

        Args:
            truncation_reward: If set, reward fns are skipped on truncated
                rollouts and this value is returned. `None` → reward fns run
                normally.
            error_reward: If set, reward fns are skipped on errored rollouts
                and this value is returned. `None` → reward fns run normally.
        """

        truncation_reward: float | None = None
        error_reward: float | None = None

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Rubric.Config()

    @functools.cached_property
    def _rwd_fns(self) -> list[RewardFn]:
        """Registered reward fns with weights normalized to sum to 1.0.

        Built lazily on first scoring call so subclass `__init__` may set
        instance state in any order.
        """
        registered = self.register_funcs()
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
        return [
            RewardFn(fn=rwd_fn.fn, weight=rwd_fn.weight / total_weight)
            for rwd_fn in registered
        ]

    def register_funcs(self) -> list[RewardFn]:
        """Return reward fns and raw weights for this rubric.

        Returns:
            List of `RewardFn(fn=..., weight=...)` entries. Weights are
            normalized by `Rubric` before scoring.
        """
        raise NotImplementedError

    @sl.log_trace_span("score_one")
    async def score_one(self, rollout: Rollout, env_input: object) -> Reward:
        """Score one rollout.

        Args:
            rollout: Rollout to score.
            env_input: Dataset payload used to construct the rollout env.

        Returns:
            Final weighted reward + per-fn raw components. Short-circuits on
            truncate / error if `Config.truncation_reward` / `error_reward`
            are set; otherwise runs every registered reward fn.
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
        components = {
            _fn_name(rwd_fn.fn): r
            for rwd_fn, r in zip(self._rwd_fns, per_fn_rewards, strict=True)
        }
        total_reward = sum(
            rwd_fn.weight * r
            for rwd_fn, r in zip(self._rwd_fns, per_fn_rewards, strict=True)
        )
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
            env_input: Dataset payload shared by the group.

        Returns:
            One `Reward` per rollout, in input order.
        """
        return await asyncio.gather(*(self.score_one(r, env_input) for r in rollouts))


def _fn_name(fn: Callable) -> str:
    """Plain fns have `__name__`; callable instances use the class name."""
    return getattr(fn, "__name__", None) or type(fn).__name__
