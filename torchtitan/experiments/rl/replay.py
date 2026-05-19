# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollout -> :class:`ReplaySample` conversion + the :class:`ReplayBuffer`.

Three pieces:

- :func:`rollout_to_replay_samples` — walk a multi-turn rollout and
  emit one :class:`ReplaySample` per contiguous prefix-runnable span.
- :func:`compute_advantages` — group-mean baseline; returns
  ``{(group_id, sample_idx): advantage}``.
- :class:`ReplayBuffer` — bounded FIFO held on the controller (plain
  Python class, not a Monarch actor). Producer rollout tasks and the
  consumer training task share it through :class:`asyncio.Condition`
  so the trainer awaits until samples land rather than sleep-polling.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.types import ReplaySample, RolloutOutput

__all__ = [
    "BufferClosedError",
    "ReplayBuffer",
    "compute_advantages",
    "rollout_to_replay_samples",
]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def rollout_to_replay_samples(r: RolloutOutput) -> list[ReplaySample]:
    """Convert one :class:`RolloutOutput` to one or more :class:`ReplaySample`s.

    Walks turns in order. For each turn:

    - If the turn's ``prompt_tokens`` extends the running buffer as a
      strict prefix, append the extension as ``loss_mask=0`` and the
      response tokens as ``loss_mask=1``.
    - Otherwise, flush the running sample and start a new one with
      this turn's prompt + response.

    The first turn always starts a fresh sample. Single-turn rollouts
    always produce exactly one sample.

    ``advantage`` is set to 0.0 on each sample; the controller fills
    it from :func:`compute_advantages` at collate time. ``policy_version``
    is the minimum across the turns that contributed to the sample —
    the conservative choice when a sample spans a weight swap.
    """
    samples: list[ReplaySample] = []
    if not r.turns or r.reward is None:
        # Empty rollouts and ERROR-status rollouts (parse / timeout
        # failures) have no learning signal — skip them entirely.
        return samples

    tokens: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []
    turn_versions: list[int] = []

    def _flush() -> None:
        if not tokens:
            return
        samples.append(
            ReplaySample(
                token_ids=list(tokens),
                loss_mask=list(mask),
                behavior_logprobs=list(logprobs),
                advantage=0.0,
                group_id=r.group_id,
                sample_idx=r.sample_idx,
                policy_version=min(turn_versions),
                reward=float(r.reward),
                reward_components=dict(r.reward_components),
            )
        )

    for i, turn in enumerate(r.turns):
        is_continuation = i > 0 and turn.prompt_token_ids[: len(tokens)] == tokens
        if not is_continuation:
            _flush()
            tokens = list(turn.prompt_token_ids)
            mask = [0] * len(turn.prompt_token_ids)
            logprobs = [0.0] * len(turn.prompt_token_ids)
            turn_versions = [turn.policy_version]
        else:
            tail = turn.prompt_token_ids[len(tokens) :]
            tokens.extend(tail)
            mask.extend([0] * len(tail))
            logprobs.extend([0.0] * len(tail))
            turn_versions.append(turn.policy_version)

        tokens.extend(turn.response_token_ids)
        mask.extend([1] * len(turn.response_token_ids))
        logprobs.extend(turn.response_logprobs)

    _flush()
    return samples


def compute_advantages(
    rollouts: Sequence[RolloutOutput],
) -> dict[tuple[str, int], float]:
    """Group-mean baseline. Returns ``{(group_id, sample_idx): advantage}``.

    Groups rollouts by ``group_id`` and subtracts the per-group mean
    reward from each rollout's reward. Groups with one member produce
    zero advantage (trivially) — those rollouts have no learning signal
    and should be filtered upstream if needed.

    Standard deviation normalization is NOT applied here; per DAPO the
    consensus is mean-only is sufficient and std-normalization tends to
    over-attenuate signal on near-uniform-reward groups.
    """
    by_group: dict[str, list[RolloutOutput]] = defaultdict(list)
    for r in rollouts:
        if r.reward is None:
            continue  # skip ERROR-status rollouts; no learning signal
        by_group[r.group_id].append(r)

    out: dict[tuple[str, int], float] = {}
    for group in by_group.values():
        rewards = [float(r.reward) for r in group]
        mean = sum(rewards) / len(rewards)
        for r, rew in zip(group, rewards, strict=True):
            out[(r.group_id, r.sample_idx)] = rew - mean
    return out


# ---------------------------------------------------------------------------
# ReplayBuffer actor
# ---------------------------------------------------------------------------


class ReplayBuffer(Configurable):
    """Bounded FIFO between rollout producers and the trainer consumer.

    Lives on the controller process. ``add`` and ``sample`` synchronize
    through an :class:`asyncio.Condition`: the trainer's ``await sample``
    blocks until ``add`` notifies enough rollouts have landed, or the
    buffer is closed.

    Each ``sample`` call first evicts entries older than
    ``max_policy_age`` train steps. With the generator's drain-on-pull,
    a rollout is at most one weight version behind, so
    ``max_policy_age=1`` is the safe default.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        batch_size: int = 8
        """Number of :class:`ReplaySample`\\ s per DP rank per train step."""

        dp_size: int = 1
        """Trainer data-parallel degree; ``sample`` returns one batch per rank."""

        max_policy_age: int | None = 1
        """Drop samples whose ``policy_version`` is more than this many
        train steps behind ``curr_policy_version``. ``None`` disables aging."""

        max_buffer_size: int = 2048
        """Hard cap on stored samples; oldest fall off on overflow."""

        seed: int = 0
        """Seed for the in-actor sampling RNG."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._buffer: deque[ReplaySample] = deque(maxlen=config.max_buffer_size)
        self._rng = random.Random(config.seed)
        self._cv: asyncio.Condition | None = None  # initialized lazily
        self._closed: bool = False

    def _cond(self) -> asyncio.Condition:
        if self._cv is None:
            self._cv = asyncio.Condition()
        return self._cv

    async def add(self, samples: list[ReplaySample]) -> None:
        """Append samples; wake consumers parked in ``sample``."""
        cv = self._cond()
        async with cv:
            self._buffer.extend(samples)
            cv.notify_all()

    async def sample(self, *, curr_policy_version: int) -> list[list[ReplaySample]]:
        """Wait until ``dp_size * batch_size`` survivors are available.

        Returns shape ``[dp_size][batch_size]``. Evicts stale samples
        on every wake-up. Raises ``BufferClosedError`` when the buffer
        is closed mid-wait.
        """
        cv = self._cond()
        async with cv:
            while True:
                if self._closed:
                    raise BufferClosedError("ReplayBuffer closed during sample()")
                self._evict(curr_policy_version)
                total = self.config.dp_size * self.config.batch_size
                if len(self._buffer) >= total:
                    indices = self._rng.sample(range(len(self._buffer)), k=total)
                    chosen = [self._buffer[i] for i in indices]
                    return [
                        chosen[
                            r
                            * self.config.batch_size : (r + 1)
                            * self.config.batch_size
                        ]
                        for r in range(self.config.dp_size)
                    ]
                await cv.wait()

    async def close(self) -> None:
        """Wake all waiters with :class:`BufferClosedError`."""
        cv = self._cond()
        async with cv:
            self._closed = True
            cv.notify_all()

    def _evict(self, curr_policy_version: int) -> None:
        if self.config.max_policy_age is None:
            return
        cutoff = curr_policy_version - self.config.max_policy_age
        self._buffer = deque(
            (s for s in self._buffer if s.policy_version >= cutoff),
            maxlen=self.config.max_buffer_size,
        )


class BufferClosedError(RuntimeError):
    """Raised by ``ReplayBuffer.sample`` when ``close`` fires mid-wait."""
