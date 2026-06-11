# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollout.types import Rollout
from torchtitan.observability import structured_logger as sl


class RewardFn(Configurable, abc.ABC):
    """A single reward function, as a Configurable callable.

    Subclass and implement `__call__`. Its `Config` carries the `weight` used in
    the rubric's weighted sum, plus any args a stateful reward fn needs (a reward
    model path, an LLM-judge endpoint, a threshold, ...).

    Example:
        class RewardCorrect(RewardFn):
            @dataclass(kw_only=True, slots=True)
            class Config(RewardFn.Config):
                pass  # only needs `weight`

            async def __call__(self, rollout, env_input) -> float:
                ...
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        weight: float = 1.0
        """Relative weight in the rubric's weighted sum (normalized across fns)."""

    def __init__(self, config: Config) -> None:
        self.weight = config.weight

    @abc.abstractmethod
    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        """Return this fn's score for one rollout.

        Args:
            rollout: Rollout to score.
            env_input: Dataset payload used to build the env (target/metadata).
        """


class RewardCompletionHash(RewardFn):
    """Small deterministic reward for debug runs with random-init models.

    This is intended for CI configs where task rewards may be all zero because
    the model has not learned the response format. It should be assigned a low
    rubric weight alongside the real task reward.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        modulus: int = 11
        """Bucket count for the checksum reward. Must be greater than 1."""

        def __post_init__(self) -> None:
            if self.modulus <= 1:
                raise ValueError(f"modulus must be greater than 1; got {self.modulus}")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.modulus = config.modulus

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        completion_text = "\n".join(
            (turn.completion_message or {}).get("content") or ""
            for turn in rollout.turns
        )
        checksum = sum(completion_text.encode("utf-8", errors="ignore"))
        return (checksum % self.modulus) / (self.modulus - 1)


@dataclass(frozen=True, kw_only=True, slots=True)
class RubricOutput:
    """One rollout's reward, as returned by a `Rubric`.

    Example:
        >>> RubricOutput(reward=0.5, reward_breakdown={"RewardCorrect": 1.0, "RewardFormat": 0.0})
    """

    reward: float
    """Final scalar reward for this rollout; assigned to `Rollout.reward`."""

    reward_breakdown: dict[str, float] = field(default_factory=dict)
    """Per-reward-fn outputs (unweighted), keyed by reward-fn class name.
    The default `Rubric` computes `reward` from these; callers may also use them for per-reward
    advantage, reweighting, metrics, or inspection."""


class Rubric(Configurable):
    """Scores rollouts with a set of weighted reward functions.

    The reward fns and their weights live in config (`reward_fns`), so common
    cases need no subclass. Subclass and override `score_group` for cross-sibling
    scoring (pairwise comparison, diversity, rank normalization).

    Setting `truncation_reward` / `error_reward` short-circuits the reward fns for
    rollouts whose status is truncated / errored.

    Example:
        rubric = Rubric.Config(
            reward_fns=[RewardCorrect.Config(weight=1.0), RewardFormat.Config(weight=0.3)],
            truncation_reward=0.0,
        ).build()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        reward_fns: list[RewardFn.Config] = field(default_factory=list)
        """The rubric's reward fns + weights; built and weight-normalized at init."""

        truncation_reward: float | None = None
        """Reward for a truncated rollout. If set, the reward fns are SKIPPED and this fixed
        reward is used. If None, the reward fns run on the truncated rollout."""

        error_reward: float | None = None
        """Reward for a errored rollout. If set, the reward fns are SKIPPED and this fixed
        reward is used. If None, the reward fns run on the errored rollout."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._reward_fns = [rwd_cfg.build() for rwd_cfg in config.reward_fns]

        # Sanity checks
        if not self._reward_fns:
            raise ValueError("Rubric.Config.reward_fns must not be empty")
        names = [type(fn).__name__ for fn in self._reward_fns]
        if len(names) != len(set(names)):
            raise ValueError(f"reward fn names must be unique; got {names}")
        self._weight_sum = sum(fn.weight for fn in self._reward_fns)
        if self._weight_sum <= 0:
            raise ValueError(
                f"rubric weights must sum to a positive value; got {self._weight_sum}"
            )

    @sl.log_trace_span("score_single_rollout")
    async def _score_single_rollout(
        self, rollout: Rollout, env_input: object
    ) -> RubricOutput:
        """Score one rollout. Short-circuits to `truncation_reward` /
        `error_reward` when those are set and the rollout truncated / errored.

        Args:
            rollout: Rollout to score.
            env_input: Dataset payload used to build the env (target/metadata).

        Returns:
            Final weighted reward + per-fn raw breakdown.
        """
        # Short-circuit on truncate / error and return the configured reward. The
        # breakdown records the short-circuit reason so it shows up in metrics.
        cfg = self._config
        if cfg.truncation_reward is not None and rollout.status.is_truncated():
            return RubricOutput(
                reward=cfg.truncation_reward,
                reward_breakdown={"truncated": cfg.truncation_reward},
            )
        if cfg.error_reward is not None and rollout.status.is_error():
            return RubricOutput(
                reward=cfg.error_reward,
                reward_breakdown={"errored": cfg.error_reward},
            )

        # Run all reward fns and weight-sum (weights normalized to sum to 1.0).
        per_fn_rewards = await asyncio.gather(
            *(fn(rollout, env_input) for fn in self._reward_fns)
        )

        reward_breakdown = {}
        total_reward = 0.0
        for fn, r in zip(self._reward_fns, per_fn_rewards, strict=True):
            reward_breakdown[type(fn).__name__] = r
            total_reward += (fn.weight / self._weight_sum) * r

        return RubricOutput(reward=total_reward, reward_breakdown=reward_breakdown)

    @sl.log_trace_span("score_group")
    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[RubricOutput]:
        """Score every rollout in one prompt group.

        Override for cross-rollout rewards (pairwise comparison, diversity,
        rank normalization).

        Args:
            rollouts: Sibling rollouts sampled from one prompt group.
            env_input: Dataset payload originally used to construct the rollout env.

        Returns:
            One `RubricOutput` per rollout, in input order.
        """
        return await asyncio.gather(
            *(self._score_single_rollout(r, env_input) for r in rollouts)
        )
