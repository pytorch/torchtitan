# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollout → :class:`ReplaySample` conversion + the :class:`ReplayBuffer` actor.

Three pieces:

1. :func:`rollout_to_replay_samples` — walks a multi-turn rollout's
   turns left-to-right, accumulating into one :class:`ReplaySample`
   per contiguous prefix-runnable span. A turn whose ``prompt_tokens``
   doesn't extend the running buffer flushes the current sample and
   starts a new one (e.g. when a thinking-strip renderer breaks
   prefix continuity).
2. :func:`compute_advantages` — group-mean baseline. Maps
   ``(group_id, sample_idx) → advantage`` for a batch of rollouts;
   the controller copies these onto :class:`ReplaySample`s at collate
   time.
3. :class:`ReplayBuffer` — bounded FIFO Monarch actor. Producer-
   consumer between ``continuous_rollouts`` tasks and
   ``continuous_training``. Drops samples older than ``max_policy_age``
   training steps before sampling. Borrowed structurally from forge's
   ``ReplayBuffer`` (``forge/actors/replay_buffer.py``, BSD-licensed).

Why a buffer at all under TBR sync-checkpoint? Because the trainer's
weight-push drains the **generator** but does NOT stall the
**controller**: the next ``continuous_rollouts`` iteration begins
issuing requests against the freshly-loaded weights immediately,
while previous rollouts are still landing in the buffer stamped with
the older version. ``max_policy_age=1`` lets one-step-stale samples
flow into training; ``max_policy_age=0`` is strict on-policy at the
cost of buffer starvation.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass

from monarch.actor import Actor, endpoint

from torchtitan.config import Configurable
from torchtitan.experiments.rl.types import ReplaySample, RolloutOutput

__all__ = [
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
    if not r.turns:
        return samples

    tokens: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []
    turn_versions: list[int] = []
    expected_prefix: list[int] = []

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
                reward=float(r.reward or 0.0),
                reward_components=dict(r.reward_components),
            )
        )

    for i, turn in enumerate(r.turns):
        is_continuation = (
            i > 0 and turn.prompt_token_ids[: len(expected_prefix)] == expected_prefix
        )
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
        expected_prefix = list(tokens)

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
        by_group[r.group_id].append(r)

    out: dict[tuple[str, int], float] = {}
    for group in by_group.values():
        rewards = [float(r.reward or 0.0) for r in group]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        for r, rew in zip(group, rewards, strict=True):
            out[(r.group_id, r.sample_idx)] = rew - mean
    return out


# ---------------------------------------------------------------------------
# ReplayBuffer actor
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class _BufferEntry:
    sample: ReplaySample
    sample_count: int = 0


class ReplayBuffer(Actor, Configurable):
    """Bounded FIFO buffer mediating producer ``continuous_rollouts`` tasks
    and consumer ``continuous_training``.

    Lives as a Monarch actor so its ``add`` / ``sample`` endpoints
    serialize automatically — no controller-side lock needed. Returns
    ``None`` from ``sample`` when underfilled so the trainer can
    ``await asyncio.sleep(0.1); continue`` until more rollouts land.

    The eviction policy is age-based: every call to ``sample`` first
    drops entries whose ``policy_version`` is more than ``max_policy_age``
    behind the trainer's ``curr_policy_version``. With sync-checkpoint
    + per-rollout drain in the generator, a sample cannot span more
    than two weight versions; ``max_policy_age=1`` accepts one-step-
    stale samples (which is what you get when a rollout that started
    on V is added to the buffer right after the trainer pushed V+1).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        batch_size: int = 8
        """Number of :class:`ReplaySample`s per DP rank per train step."""

        dp_size: int = 1
        """Trainer data-parallel degree; ``sample`` returns one batch per rank."""

        max_policy_age: int | None = 1
        """Drop samples whose policy_version is more than this many train
        steps behind the trainer's current step. ``None`` disables aging."""

        max_buffer_size: int = 2048
        """Hard cap on stored samples; oldest are dropped on overflow."""

        seed: int = 0
        """Seed for the in-actor sampling RNG."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._buffer: deque[_BufferEntry] = deque(maxlen=config.max_buffer_size)
        self._rng = random.Random(config.seed)

    @endpoint
    async def add(self, samples: list[ReplaySample]) -> None:
        """Append samples (oldest fall off when ``max_buffer_size`` is exceeded)."""
        for s in samples:
            self._buffer.append(_BufferEntry(sample=s))

    @endpoint
    async def sample(
        self, *, curr_policy_version: int
    ) -> list[list[ReplaySample]] | None:
        """Pop a batch sharded by ``dp_size``, or ``None`` if underfilled.

        Evicts stale samples first, then random-samples
        ``dp_size * batch_size`` survivors. Returns
        ``list[list[ReplaySample]]`` of shape ``[dp_size][batch_size]``
        so the trainer can fan out one batch per DP rank.
        """
        self._evict(curr_policy_version)
        total = self.config.dp_size * self.config.batch_size
        if len(self._buffer) < total:
            return None

        indices = self._rng.sample(range(len(self._buffer)), k=total)
        chosen: list[ReplaySample] = []
        for i in indices:
            self._buffer[i].sample_count += 1
            chosen.append(self._buffer[i].sample)

        return [
            chosen[r * self.config.batch_size : (r + 1) * self.config.batch_size]
            for r in range(self.config.dp_size)
        ]

    @endpoint
    async def size(self) -> int:
        """Current buffer length (for logging / debugging)."""
        return len(self._buffer)

    @endpoint
    async def clear(self) -> None:
        """Drop everything. Used by validation to reset between runs."""
        self._buffer.clear()

    # ------------------------------------------------------------------ internal

    def _evict(self, curr_policy_version: int) -> None:
        if self.config.max_policy_age is None:
            return
        cutoff = curr_policy_version - self.config.max_policy_age
        self._buffer = deque(
            (e for e in self._buffer if e.sample.policy_version >= cutoff),
            maxlen=self.config.max_buffer_size,
        )
