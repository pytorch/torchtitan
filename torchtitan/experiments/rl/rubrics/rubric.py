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
    """One reward function + its weight in the rubric's weighted sum."""

    fn: Callable[[Rollout, object], Awaitable[float]]
    """``async (rollout, env_input) -> float``. Plain ``async def`` functions
    match; callable instances (``class Foo: async def __call__``) also match,
    so stateful graders work without a base class."""

    weight: float = 1.0
    """Raw weight; the rubric normalizes weights to sum to 1.0 internally,
    so what matters is the ratio between weights."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Reward:
    """One rollout's reward + per-fn component breakdown."""

    reward: float
    """Final weighted reward for this rollout."""

    components: dict[str, float] = field(default_factory=dict)
    """Per-reward-fn raw output, keyed by ``fn.__name__``. Diagnostic;
    weighted sum of these is NOT the reward (weights aren't applied here)."""


class Rubric(Configurable):
    """Weighted sum of reward functions.

    Subclass and implement ``register_funcs()`` returning a list of
    ``RewardFn``. Weights are normalized to sum to 1.0 across the list,
    so the final reward is bounded by the per-fn output range.

    ``truncation_reward`` / ``error_reward`` (default ``None``) override
    the weighted-sum path for non-COMPLETED rollouts: when set, all
    reward fns are SKIPPED and the configured value is returned. When
    ``None``, reward fns run normally on the partial response and can
    inspect ``rollout.status`` themselves if they care.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        truncation_reward: float | None = None
        """If set, reward fns are skipped on truncated rollouts and this
        value is returned. ``None`` → reward fns run normally."""

        error_reward: float | None = None
        """If set, reward fns are skipped on errored rollouts and this
        value is returned. ``None`` → reward fns run normally."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Rubric.Config()

    @functools.cached_property
    def _rwd_fns(self) -> list[RewardFn]:
        """Registered reward fns with weights normalized to sum to 1.0.

        Built lazily on first scoring call so subclass ``__init__`` may
        set instance state in any order.
        """
        registered = self.register_funcs()
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
        """Subclass override: return ``[RewardFn(fn=..., weight=...), ...]``."""
        raise NotImplementedError

    @sl.log_trace_span("score_rollout")
    async def score_rollout(self, rollout: Rollout, env_input: object) -> Reward:
        """Score one rollout via the registered reward fns.

        Callers that want the truncation/error short-circuit should use
        ``score_group`` instead; this method always runs every reward fn.
        """
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
        """Score every rollout in a group. Short-circuits on truncate / error
        if ``Config.truncation_reward`` / ``error_reward`` are set; otherwise
        runs ``score_rollout`` on every rollout.

        Override for cross-rollout math (pairwise / diversity / rank-norm).
        """
        cfg = self._config

        async def _score(rollout: Rollout) -> Reward:
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
            return await self.score_rollout(rollout, env_input)

        return await asyncio.gather(*(_score(r) for r in rollouts))


def _fn_name(fn: Callable) -> str:
    """Plain fns have ``__name__``; callable instances use class name."""
    return getattr(fn, "__name__", None) or type(fn).__name__
