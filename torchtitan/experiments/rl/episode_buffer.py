# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FIFO buffer between rollout workers and the batcher.

Rollout workers add episodes. The batcher waits until the buffer has enough fresh tokens for a
batch, packs the front, then removes the consumed episodes. Metrics are stored beside the episodes
and leave the buffer with the batch that consumes them.

A plain asyncio object in the controller's event loop (not a Monarch actor — it owns no GPU), with
two bounds: a version bound (`max_offpolicy_steps`, the staleness drop) and a depth bound
(`max_buffered_batches`, how far the producer banks ahead). It owns NO packing — the batcher decides
the cut; the buffer holds, gates on token totals, drops stale, and backpressures.

Episodes are held in FIFO `_BufferedEpisodes` — one per `add_episodes` (one rollout-collection
round) — so each round's metrics stay tied to its episodes: they ride out with the batch that
consumes the first episode (logged once, with data that trained), and a fully-stale round is dropped
whole, metrics included, so the logs never report rollouts that never trained.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from torchtitan.experiments.rl.batcher import Batcher
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Episode
from torchtitan.observability import structured_logger as sl


def oldest_sampled_version(episode: Episode) -> int:
    """The oldest policy version the episode's tokens were sampled at (conservative staleness).

    Reads the min over `version_intervals` so a packed multi-turn episode is as stale as its oldest
    turn; falls back to `policy_version` when intervals are absent.

    Example:

        oldest_sampled_version(Episode(version_intervals=[(0, 5), (40, 6)], ...))  # -> 5
    """
    if not episode.version_intervals:
        return episode.policy_version
    return min(version for _start, version in episode.version_intervals)


def token_weighted_avg_version(episode: Episode) -> float | None:
    """Mean of an episode's sampled versions, weighted by each turn's token span (None if absent).

    Example:

        # 6 tokens, intervals [(0, 5), (3, 6)]: 3 tokens at v5, 3 at v6 -> 5.5
        token_weighted_avg_version(Episode(token_ids=[...6...], version_intervals=[(0, 5), (3, 6)]))
        # -> 5.5
    """
    intervals = episode.version_intervals
    num_tokens = len(episode.token_ids)
    if not intervals or num_tokens == 0:
        return None
    weighted = 0.0
    for i, (start, version) in enumerate(intervals):
        end = intervals[i + 1][0] if i + 1 < len(intervals) else num_tokens
        weighted += version * (end - start)  # this turn's token span
    return weighted / num_tokens


@dataclass(slots=True)
class _BufferedEpisodes:
    """One `add_episodes` round's episodes plus the metrics that describe them.

    Episodes and metrics travel together so staleness drops and the metric log never disagree: drop
    the whole round and its metrics go too; consume its episodes and its metrics ride out with them.
    `metrics` is cleared once emitted, so a round split across two batches logs its metrics once.
    """

    episodes: list[Episode]
    metrics: list[m.Metric]


class EpisodeBuffer:
    """A FIFO of scored `Episode`s between the rollout workers and the batcher loop.

    Workers `add_episodes`; the batcher loop `wait_for_batch_episodes` once a full batch of fresh
    tokens is buffered, packs them, and `pop_consumed_episodes` for the consumed count. Stale episodes
    are dropped (not down-weighted) against the trainer's LIVE version; `add_episodes` backpressures at
    the depth bound.

    Args:
        batcher: owns packing + the batch token target (`target_batch_tokens`, `num_packed_tokens`).
        dp_degree: data-parallel degree, for the token target.
        max_offpolicy_steps: drop an episode whose oldest token is more than this many versions behind
            the trainer's version. 0 = strict on-policy; None = never drop (log staleness only).
        drop_rollout_group_if_any_stale: drop a whole `add_episodes` round (one GRPO group) if ANY of
            its episodes is stale, instead of dropping stale episodes individually.
        max_buffered_batches: batches the producer may bank ahead before `add_episodes` backpressures.
        train_version: getter for the trainer's CURRENT policy version. Read live at every drop so a
            version bump while the batcher loop waits is enforced immediately (banked data ages in
            place), and so the staleness METRIC matches the bound the drop enforced.
        on_round_drained: called once per round (= one GRPO group) that leaves the buffer (consumed or
            dropped), so an `AdmissionBudget` can release the group's permit.

    Example:

        buffer = EpisodeBuffer(batcher=batcher, dp_degree=2, max_offpolicy_steps=1,
                               max_buffered_batches=2, train_version=lambda: trainer.version)
        await buffer.add_episodes(episodes, metrics)         # rollout worker
        episodes = await buffer.wait_for_batch_episodes()    # batcher loop
        if episodes is not None:
            packed = batcher.pack_batch(episodes, dp_degree=2)
            batch_metrics = await buffer.pop_consumed_episodes(packed.num_episodes_consumed)
    """

    def __init__(
        self,
        *,
        batcher: Batcher,
        dp_degree: int,
        max_offpolicy_steps: int | None,
        max_buffered_batches: int,
        train_version: Callable[[], int],
        drop_rollout_group_if_any_stale: bool = False,
        on_round_drained: Callable[[], None] | None = None,
    ) -> None:
        if max_offpolicy_steps is not None and max_offpolicy_steps < 0:
            raise ValueError(
                f"max_offpolicy_steps must be >= 0 or None, got {max_offpolicy_steps}"
            )
        if max_buffered_batches < 1:
            raise ValueError(
                f"max_buffered_batches must be >= 1, got {max_buffered_batches}"
            )
        self._batcher = batcher
        self._dp_degree = dp_degree
        self._max_offpolicy_steps = max_offpolicy_steps
        self._drop_rollout_group_if_any_stale = drop_rollout_group_if_any_stale
        self._max_buffered_batches = max_buffered_batches
        # Called once per round (one `add_episodes`, one GRPO group) that leaves the buffer —
        # consumed or dropped — so an AdmissionBudget can release the group's permit.
        self._on_round_drained = on_round_drained or (lambda: None)
        self._train_version = train_version
        self._buffered: deque[_BufferedEpisodes] = deque()
        # Notified by add_episodes (data added) and by _drop_stale_episodes / pop_consumed_episodes
        # (space freed).
        self._cv = asyncio.Condition()
        self._closed = False
        self._num_stale_episodes_dropped_total = (
            0  # cumulative, for buffer/dropped_stale
        )
        self._num_stale_dropped_since_last_batch = (
            0  # numerator of stale_drop_rate; reset each pop_consumed_episodes
        )

    async def add_episodes(
        self, episodes: list[Episode], metrics: list[m.Metric]
    ) -> None:
        """Append one round's episodes + metrics, then backpressure on the depth bound.

        No `train_version`: the producer just appends; staleness is dropped at
        `wait_for_batch_episodes` time against the trainer's live version (one drop site). An
        empty-episode round still records its metrics (e.g. a fully-failed group), which drain out
        with the next `pop_consumed_episodes`.

        Args:
            episodes: scored episodes from one rollout-collection round (may be empty on a failure).
            metrics: rollout/generation metrics for these episodes; ride out with the batch that
                consumes them.
        """
        async with self._cv:
            self._buffered.append(
                _BufferedEpisodes(episodes=list(episodes), metrics=list(metrics))
            )
            self._cv.notify_all()  # wake the batcher loop: a full batch may be ready now
            # Backpressure: park until below the depth bound, so the producer banks at most
            # `max_buffered_batches` ahead.
            await self._cv.wait_for(lambda: self._closed or not self._at_depth_limit())

    @sl.log_trace_span("wait_for_batch_episodes")
    async def wait_for_batch_episodes(self) -> list[Episode] | None:
        """Block until a full batch of FRESH episodes is buffered; drop stale; snapshot the episodes.

        Returns the buffered episodes (oldest first), or `None` ONLY when closed and no full fresh
        batch remains. The batcher cuts one batch from the front of the snapshot and calls
        `pop_consumed_episodes`; metrics ride out from there, not here, so an `add_episodes` arriving
        during the off-loop pack stays with its own (next) batch.
        """
        async with self._cv:
            while True:
                # The version may have advanced (a swap landed) while we waited; re-drop the whole
                # buffer against the LIVE version. This is where the staleness drop does real work;
                # the trainer idles here meanwhile.
                self._drop_stale_episodes()
                if self._has_full_batch():
                    break
                if self._closed:
                    return None  # closed without a full batch: caller stops
                await self._cv.wait_for(lambda: self._closed or self._has_full_batch())
            return [
                episode for buffered in self._buffered for episode in buffered.episodes
            ]

    async def pop_consumed_episodes(self, num_consumed: int) -> list[m.Metric]:
        """Drop the `num_consumed` front episodes the batcher used and return this batch's metrics.

        Returns the metrics that ride out with this batch: the consumed rounds' rollout/generation
        metrics (each round's ride out once, with the batch that consumes its first episode) plus the
        staleness / depth / stale-drop metrics (depth measured POST-peel, staleness against the live
        version — the version this batch actually trains under).
        """
        async with self._cv:
            consumed, ride_out_metrics = self._pop_front(num_consumed)
            dropped_since_batch, self._num_stale_dropped_since_last_batch = (
                self._num_stale_dropped_since_last_batch,
                0,
            )
            depth_batches = self._buffered_tokens() / max(
                self._batch_tokens(), 1
            )  # POST-peel
            depth_episodes = sum(len(b.episodes) for b in self._buffered)
            version = self._train_version()
            self._cv.notify_all()  # backpressure released
        return ride_out_metrics + self._staleness_metrics(
            consumed, dropped_since_batch, depth_batches, depth_episodes, version
        )

    async def close(self) -> None:
        """Mark closed and wake blocked add_episodes / wait_for_batch_episodes callers."""
        async with self._cv:
            self._closed = True
            self._cv.notify_all()

    def _pop_front(self, num_consumed: int) -> tuple[list[Episode], list[m.Metric]]:
        """Pop the front `num_consumed` episodes the batcher used; return them and the metrics that
        ride out with this batch.

        A round rides its metrics out with the batch that consumes its FIRST episode, then clears them
        — so a round split across two batches is logged once. Leading metrics-only rounds (a failed
        round: zero episodes) drain here too.
        """
        consumed: list[Episode] = []
        ride_out: list[m.Metric] = []
        remaining = num_consumed
        while self._buffered:
            buffered = self._buffered[0]
            if (
                len(buffered.episodes) <= remaining
            ):  # whole round consumed (or a 0-episode round)
                self._buffered.popleft()
                self._on_round_drained()  # this round (group) is fully consumed
                consumed.extend(buffered.episodes)
                ride_out.extend(buffered.metrics)
                remaining -= len(buffered.episodes)
            elif remaining > 0:  # this batch took only the front of the round
                consumed.extend(buffered.episodes[:remaining])
                ride_out.extend(
                    buffered.metrics
                )  # this batch started the round -> ride now
                buffered.episodes = buffered.episodes[remaining:]
                buffered.metrics = (
                    []
                )  # already rode out; the tail trains under the next batch
                remaining = 0
            else:  # remaining == 0 and the front round still holds episodes for the next batch
                break
        return consumed, ride_out

    # --- token-total gates (the buffer's only use of the batcher: accounting, never packing) ---

    def _buffered_tokens(self) -> int:
        return sum(
            self._batcher.num_packed_tokens(episode)
            for buffered in self._buffered
            for episode in buffered.episodes
        )

    def _batch_tokens(self) -> int:
        return self._batcher.target_batch_tokens(self._dp_degree)

    def _has_full_batch(self) -> bool:
        # Readiness: enough fresh tokens for one batch (gates wait_for_batch_episodes).
        return self._buffered_tokens() >= self._batch_tokens()

    def _at_depth_limit(self) -> bool:
        # Backpressure: banked >= max_buffered_batches worth of tokens (gates add_episodes).
        return (
            self._buffered_tokens() >= self._max_buffered_batches * self._batch_tokens()
        )

    # --- staleness drop (one site, called from wait_for_batch_episodes) ---

    def _episode_is_fresh(self, episode: Episode, version: int) -> bool:
        # On-policy enough to keep iff the episode's oldest token is within `max_offpolicy_steps`
        # versions of the trainer's live `version`. None = no bound (everything is fresh).
        if self._max_offpolicy_steps is None:
            return True
        return version - oldest_sampled_version(episode) <= self._max_offpolicy_steps

    def _drop_stale_episodes(self) -> None:
        """Drop episodes (or whole groups, per `drop_rollout_group_if_any_stale`) stale past the bound."""
        # Banked episodes age as the version advances, so re-scan the WHOLE buffer. Count first so the
        # common case (nothing stale, the hot path) makes no new deque; rebuild + notify the producer
        # only when something dropped. A round whose every episode went stale is dropped whole (its
        # metrics go too — that data never trains, so it never logs).
        if self._max_offpolicy_steps is None:  # measurement-only: never drop
            return
        version = self._train_version()
        # drop_rollout_group_if_any_stale: a round IS one GRPO group, so a stale episode condemns its
        # whole round (keeps groups intact for mean-baseline advantage).
        if self._drop_rollout_group_if_any_stale:
            surviving = deque()
            dropped = 0
            for buffered in self._buffered:
                if any(
                    not self._episode_is_fresh(e, version) for e in buffered.episodes
                ):
                    dropped += len(
                        buffered.episodes
                    )  # one stale episode condemns its group
                    self._on_round_drained()  # the whole group leaves
                else:
                    surviving.append(buffered)  # all fresh, or metrics-only round
            if not dropped:
                return
            self._buffered = surviving
        else:
            dropped = sum(
                1
                for buffered in self._buffered
                for episode in buffered.episodes
                if not self._episode_is_fresh(episode, version)
            )
            if not dropped:
                return
            surviving: deque[_BufferedEpisodes] = deque()
            for buffered in self._buffered:
                fresh = [
                    e for e in buffered.episodes if self._episode_is_fresh(e, version)
                ]
                if (
                    fresh or not buffered.episodes
                ):  # survivors, or metrics-only round: keep
                    buffered.episodes = fresh
                    surviving.append(buffered)
                else:  # every episode in this round went stale -> drop the round, metrics included
                    self._on_round_drained()
            self._buffered = surviving
        self._num_stale_episodes_dropped_total += dropped
        self._num_stale_dropped_since_last_batch += dropped
        self._cv.notify_all()

    def _staleness_metrics(
        self,
        consumed: list[Episode],
        dropped_since_batch: int,
        depth_batches: float,
        depth_episodes: int,
        version: int,
    ) -> list[m.Metric]:
        """Build this batch's staleness / depth / born-stale metrics from the consumed episodes."""
        # Panel keys live under perf/ (the 12-metric panel); drill-down keys under buffer/.
        staleness = [version - oldest_sampled_version(episode) for episode in consumed]
        denom = dropped_since_batch + len(consumed)
        stale_drop_rate = dropped_since_batch / denom if denom else 0.0

        # Born-stale: derived from each episode's carried version_intervals (one per turn). A weight
        # update that landed mid-generation shows up as an extra interval, so an episode can be born
        # stale even when first sampled on-policy. initial = vs the first turn's version, final = vs
        # the last turn's version.
        with_intervals = [e for e in consumed if e.version_intervals]
        num_inflight_weight_updates = [
            len(e.version_intervals) - 1 for e in with_intervals
        ]
        initial_staleness = [
            version - e.version_intervals[0][1] for e in with_intervals
        ]
        final_staleness = [version - e.version_intervals[-1][1] for e in with_intervals]
        avg_staleness = [
            version - token_weighted_avg_version(e) for e in with_intervals
        ]

        return [
            m.Metric(
                "perf/buffer_staleness_max",
                m.NoReduce(float(max(staleness, default=0))),
            ),
            m.Metric("perf/buffer_depth_batches", m.NoReduce(depth_batches)),
            m.Metric("perf/buffer_stale_drop_rate", m.NoReduce(stale_drop_rate)),
            m.Metric("buffer/staleness", m.Mean.from_list(staleness)),
            m.Metric(
                "buffer/dropped_stale",
                m.NoReduce(float(self._num_stale_episodes_dropped_total)),
            ),
            m.Metric("buffer/depth_episodes", m.NoReduce(float(depth_episodes))),
            m.Metric(
                "buffer/num_inflight_weight_updates",
                m.Mean.from_list(num_inflight_weight_updates),
            ),
            m.Metric(
                "buffer/num_inflight_weight_updates",
                m.Max.from_list(num_inflight_weight_updates),
            ),
            m.Metric("buffer/initial_staleness", m.Mean.from_list(initial_staleness)),
            m.Metric("buffer/final_staleness", m.Mean.from_list(final_staleness)),
            m.Metric("buffer/avg_staleness", m.Mean.from_list(avg_staleness)),
        ]
