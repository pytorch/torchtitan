# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeAlias

from torchtitan.experiments.rl.envs.types import Rollout
from torchtitan.observability import structured_logger as sl


RewardFn: TypeAlias = Callable[[Rollout, object], Awaitable[float]]
"""``(rollout, env_input) -> float`` reward function.

Plain ``async def`` functions match. Callable instances
(``class Foo: async def __call__(self, rollout, env_input)``) also
match — Python's ``Callable`` protocol covers both, so stateful
graders (LLM judge, sandbox runner) work without a base class.
"""


@dataclass(frozen=True, kw_only=True, slots=True)
class Reward:
    """One rollout's reward + per-fn component breakdown."""

    reward: float
    components: dict[str, float] = field(default_factory=dict)


class Rubric:
    """Weighted reward functions for a task.

    Owns reward assignment end-to-end (doc 37 Option B): COMPLETED rollouts
    are scored via ``funcs``; TRUNCATED and ERROR statuses use the
    configured ``truncation_reward`` / ``error_reward`` policy with
    diagnostic components ``{"truncated": 1.0}`` / ``{"error": 1.0}``.

    Args:
        funcs: Reward functions matching ``RewardFn``: each takes
            ``(rollout, env_input)`` and returns one float (its component).
        weights: Same length as ``funcs``. Per-rollout reward is the
            weighted sum.
        truncation_reward: Reward assigned to rollouts whose status
            satisfies ``is_truncated()``.
        error_reward: Reward assigned to rollouts whose status
            satisfies ``is_error()``.

    Example:

        rubric = Rubric(
            funcs=[reward_correct, reward_format],
            weights=[1.0, 0.3],
        )
        r = await rubric.score_one(completed_rollout, env_input)
        # r.components: {"reward_correct": 1.0, "reward_format": 1.0}
        # r.reward:     1.0 * 1.0 + 0.3 * 1.0 == 1.3

    Group reranking (pairwise preference, diversity, rank-norm)
    subclasses ``Rubric`` and overrides ``score_group``.
    """

    def __init__(
        self,
        *,
        funcs: list[RewardFn],
        weights: list[float],
        truncation_reward: float = 0.0,
        error_reward: float = 0.0,
    ) -> None:
        if len(funcs) != len(weights):
            raise ValueError(
                f"funcs and weights must have the same length; "
                f"got {len(funcs)} and {len(weights)}"
            )
        self.funcs = list(funcs)
        self.weights = list(weights)
        self.truncation_reward = truncation_reward
        self.error_reward = error_reward

    @sl.log_trace_span("score_one")
    async def score_one(
        self,
        rollout: Rollout,
        env_input: object,
    ) -> Reward:
        """Score one COMPLETED rollout. All funcs evaluated in parallel.

        Callers (typically ``score_group``) are responsible for filtering
        out non-COMPLETED rollouts.
        """
        values = await asyncio.gather(*(fn(rollout, env_input) for fn in self.funcs))
        components = {_fn_name(fn): v for fn, v in zip(self.funcs, values, strict=True)}
        weighted = sum(w * v for w, v in zip(self.weights, values, strict=True))
        return Reward(reward=weighted, components=components)

    @sl.log_trace_span("score_group")
    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[Reward]:
        """Whole group. Branches on status:

        - ``is_truncated()`` → ``Reward(truncation_reward, {"truncated": 1.0})``
        - ``is_error()``     → ``Reward(error_reward,      {"error": 1.0})``
        - COMPLETED          → ``score_one``

        Override entirely for cross-rollout math (pairwise / diversity /
        rank-norm).
        """

        async def _one(rollout: Rollout) -> Reward:
            if rollout.status.is_truncated():
                return Reward(
                    reward=self.truncation_reward,
                    components={"truncated": 1.0},
                )
            if rollout.status.is_error():
                return Reward(
                    reward=self.error_reward,
                    components={"error": 1.0},
                )
            return await self.score_one(rollout, env_input)

        return await asyncio.gather(*(_one(r) for r in rollouts))


def _fn_name(fn: RewardFn) -> str:
    """Plain fns have ``__name__``; callable instances use class name."""
    return getattr(fn, "__name__", None) or type(fn).__name__
