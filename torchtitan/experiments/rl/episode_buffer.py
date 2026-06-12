# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-process episode buffer between the rollout producer and the trainer.

A plain asyncio object in the controller's event loop (not a Monarch actor — it owns no GPU). It is a
FIFO of scored `Episode`s plus two bounds: a version bound (`max_offpolicy_steps`, the staleness drop)
and a depth bound (`max_buffered_batches`, how far the producer banks ahead).

It owns NO packing. Producers `put` episodes; the pack loop `take_full_batch`es a snapshot, packs it
via the `Batcher`, and `commit`s how many the batcher used. The batcher decides the cut; the buffer
only holds, gates on token totals, drops stale, and backpressures.

Episodes are held in FIFO `_BufferedGroup`s — one per `put` (one rollout-collection round) — so each
round's rollout/generation metrics stay tied to its episodes. A group's metrics ride out with the
batch that consumes its first episode (logged once, with data that trained); a fully-stale group is
dropped whole, metrics included, so the logs never report rollouts that never trained.
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


def earliest_version(episode: Episode) -> int:
    """The oldest policy version the episode's tokens were sampled at (conservative staleness).

    Reads the min over `version_intervals` so a packed multi-turn episode is as stale as its oldest
    turn; falls back to `policy_version` when intervals are absent.

    Example:

        earliest_version(Episode(version_intervals=[(0, 5), (40, 6)], ...))  # -> 5
    """
    if not episode.version_intervals:
        return episode.policy_version
    return min(version for _start, version in episode.version_intervals)


@dataclass(slots=True)
class _BufferedGroup:
    """One `put`'s episodes (a rollout-collection round) plus the metrics that describe them.

    Episodes and metrics travel together so staleness drops and the metric log never disagree: drop
    the whole group and its metrics go too; consume its episodes and its metrics ride out with them.
    `metrics` is cleared once emitted, so a group split across two batches logs its metrics once.
    """

    episodes: list[Episode]
    metrics: list[m.Metric]


class EpisodeBuffer:
    """A FIFO of scored `Episode`s between the rollout producer and the trainer's pack loop.

    Producer `put`s episodes; the pack loop `take_full_batch`es a snapshot once a full batch of fresh
    tokens is buffered, packs it, and `commit`s the consumed count. Stale episodes are dropped (not
    down-weighted) against the trainer's LIVE version; `put` backpressures at the depth bound.

    Args:
        batcher: owns packing + the batch token target (`num_tokens_target`, `trainable_tokens`).
        dp_degree: data-parallel degree, for the token target.
        max_offpolicy_steps: drop an episode whose oldest token is more than this many versions behind
            the trainer's version. 0 = strict on-policy.
        max_buffered_batches: batches the producer may bank ahead before `put` backpressures.
        train_version: getter for the trainer's CURRENT policy version. Read live at every drop so a
            version bump while the pack loop waits is enforced immediately (banked data ages in place),
            and so the staleness METRIC matches the bound the drop enforced.

    Example:

        buffer = EpisodeBuffer(batcher=batcher, dp_degree=2, max_offpolicy_steps=1,
                               max_buffered_batches=2, train_version=lambda: trainer.version)
        await buffer.put(episodes, metrics)                  # producer
        episodes = await buffer.take_full_batch()            # pack loop
        if episodes is not None:
            packed = batcher.pack_one_batch(episodes, dp_degree=2)
            ride_out_metrics = await buffer.commit(packed.num_episodes_consumed)
    """

    def __init__(
        self,
        *,
        batcher: Batcher,
        dp_degree: int,
        max_offpolicy_steps: int,
        max_buffered_batches: int,
        train_version: Callable[[], int],
    ) -> None:
        if max_offpolicy_steps < 0:
            raise ValueError(
                f"max_offpolicy_steps must be >= 0, got {max_offpolicy_steps}"
            )
        if max_buffered_batches < 1:
            raise ValueError(
                f"max_buffered_batches must be >= 1, got {max_buffered_batches}"
            )
        self._batcher = batcher
        self._dp_degree = dp_degree
        self._max_offpolicy_steps = max_offpolicy_steps
        self._max_buffered_batches = max_buffered_batches
        self._train_version = train_version
        self._groups: deque[_BufferedGroup] = deque()
        # Notified by put (data added) and by _drop_stale / commit (space freed).
        self._cv = asyncio.Condition()
        self._closed = False
        self._num_dropped_stale = 0  # cumulative, for buffer/dropped_stale
        self._dropped_since_commit = (
            0  # numerator of stale_drop_rate; reset each commit
        )

    async def put(self, episodes: list[Episode], metrics: list[m.Metric]) -> None:
        """Append one round's episodes + metrics as a group, then backpressure on the depth bound.

        No `train_version`: the producer just appends; staleness is dropped at `take_full_batch` time
        against the trainer's live version (one drop site). An empty-episode round still records its
        metrics (e.g. a fully-failed group), which drain out with the next `commit`.

        Args:
            episodes: scored episodes from one rollout-collection round (may be empty on a failure).
            metrics: rollout/generation metrics for these episodes; ride out with the batch that
                consumes them.
        """
        async with self._cv:
            self._groups.append(
                _BufferedGroup(episodes=list(episodes), metrics=list(metrics))
            )
            self._cv.notify_all()  # wake the pack loop: a full batch may be ready now
            # Backpressure: park until below the depth bound, so the producer banks at most
            # `max_buffered_batches` ahead.
            await self._cv.wait_for(lambda: self._closed or not self._at_capacity())

    @sl.log_trace_span("take_full_batch")
    async def take_full_batch(self) -> list[Episode] | None:
        """Block until a full batch of FRESH episodes is buffered; drop stale; snapshot the episodes.

        Returns the buffered episodes (oldest first), or `None` ONLY when closed and no full fresh
        batch remains. The packer cuts one batch from the front of the snapshot and calls `commit`;
        metrics ride out from `commit`, not here, so a `put` arriving during the off-loop pack stays
        with its own (next) batch.
        """
        async with self._cv:
            while True:
                # The version may have advanced (a swap landed) while we waited; re-drop the whole
                # buffer against the LIVE version. This is where the staleness drop does real work;
                # the trainer idles here meanwhile.
                self._drop_stale()
                if self._has_full_batch():
                    break
                if self._closed:
                    return None  # closed without a full batch: caller stops
                await self._cv.wait_for(lambda: self._closed or self._has_full_batch())
            return [episode for group in self._groups for episode in group.episodes]

    async def commit(self, num_consumed: int) -> list[m.Metric]:
        """Drop the `num_consumed` front episodes the packer used and return this batch's metrics.

        Returns the metrics that ride out with this batch: the consumed groups' rollout/generation
        metrics (each group's ride out once, with the batch that consumes its first episode) plus the
        staleness / depth / stale-drop metrics (depth measured POST-peel, staleness against the live
        version — the version this batch actually trains under).
        """
        async with self._cv:
            consumed, ride_out_metrics = self._consume_front(num_consumed)
            dropped_since_commit, self._dropped_since_commit = (
                self._dropped_since_commit,
                0,
            )
            depth_batches = self._buffered_tokens() / max(
                self._batch_tokens(), 1
            )  # POST-peel
            depth_episodes = sum(len(group.episodes) for group in self._groups)
            version = self._train_version()
            self._cv.notify_all()  # backpressure released
        return ride_out_metrics + self._staleness_metrics(
            consumed, dropped_since_commit, depth_batches, depth_episodes, version
        )

    async def close(self) -> None:
        """Mark the buffer closed and wake everyone, so blocked put / take_full_batch return."""
        async with self._cv:
            self._closed = True
            self._cv.notify_all()

    def _consume_front(self, num_consumed: int) -> tuple[list[Episode], list[m.Metric]]:
        """Pop the front `num_consumed` episodes the packer used; return them and the metrics that
        ride out with this batch.

        A group rides its metrics out with the batch that consumes its FIRST episode, then clears them
        — so a group split across two batches is logged once. Leading metrics-only groups (a failed
        round: zero episodes) drain here too.
        """
        consumed: list[Episode] = []
        ride_out: list[m.Metric] = []
        remaining = num_consumed
        while self._groups:
            group = self._groups[0]
            if (
                len(group.episodes) <= remaining
            ):  # whole group consumed (or a 0-episode group)
                self._groups.popleft()
                consumed.extend(group.episodes)
                ride_out.extend(group.metrics)
                remaining -= len(group.episodes)
            elif remaining > 0:  # this batch took only the front of the group
                consumed.extend(group.episodes[:remaining])
                ride_out.extend(
                    group.metrics
                )  # this batch started the group -> ride now
                group.episodes = group.episodes[remaining:]
                group.metrics = (
                    []
                )  # already rode out; the tail trains under the next batch
                remaining = 0
            else:  # remaining == 0 and the front group still holds episodes for the next batch
                break
        return consumed, ride_out

    # --- token-total gates (the buffer's only use of the batcher: accounting, never packing) ---

    def _buffered_tokens(self) -> int:
        return sum(
            self._batcher.trainable_tokens(episode)
            for group in self._groups
            for episode in group.episodes
        )

    def _batch_tokens(self) -> int:
        return self._batcher.num_tokens_target(self._dp_degree)

    def _has_full_batch(self) -> bool:
        return self._buffered_tokens() >= self._batch_tokens()

    def _at_capacity(self) -> bool:
        return (
            self._buffered_tokens() >= self._max_buffered_batches * self._batch_tokens()
        )

    # --- staleness drop (one site, called from take_full_batch) ---

    def _is_fresh(self, episode: Episode, version: int) -> bool:
        # On-policy enough to keep iff the episode's oldest token is within `max_offpolicy_steps`
        # versions of the trainer's live `version`.
        return version - earliest_version(episode) <= self._max_offpolicy_steps

    def _drop_stale(self) -> None:
        # Drop stale episodes from the WHOLE buffer (banked episodes age as the version advances).
        # Count first so the common case (nothing stale, the hot path) makes no new deque; rebuild +
        # notify the producer only when something actually dropped. A group whose every episode went
        # stale is dropped whole (its metrics go too — that data never trains, so it never logs).
        version = self._train_version()
        dropped = sum(
            1
            for group in self._groups
            for episode in group.episodes
            if not self._is_fresh(episode, version)
        )
        if not dropped:
            return
        surviving: deque[_BufferedGroup] = deque()
        for group in self._groups:
            fresh = [e for e in group.episodes if self._is_fresh(e, version)]
            if fresh or not group.episodes:  # survivors, or a metrics-only group: keep
                group.episodes = fresh
                surviving.append(group)
            # else: every episode in this group went stale -> drop the group, metrics included
        self._groups = surviving
        self._num_dropped_stale += dropped
        self._dropped_since_commit += dropped
        self._cv.notify_all()

    def _staleness_metrics(
        self,
        consumed: list[Episode],
        dropped_since_commit: int,
        depth_batches: float,
        depth_episodes: int,
        version: int,
    ) -> list[m.Metric]:
        # Panel keys live under perf/ (the 12-metric panel); drill-down keys under buffer/.
        staleness = [version - earliest_version(episode) for episode in consumed]
        denom = dropped_since_commit + len(consumed)
        stale_drop_rate = dropped_since_commit / denom if denom else 0.0
        return [
            m.Metric(
                "perf/buffer_staleness_max",
                m.NoReduce(float(max(staleness, default=0))),
            ),
            m.Metric("perf/buffer_depth_batches", m.NoReduce(depth_batches)),
            m.Metric("perf/buffer_stale_drop_rate", m.NoReduce(stale_drop_rate)),
            m.Metric("buffer/staleness", m.Mean.from_list(staleness)),
            m.Metric(
                "buffer/dropped_stale", m.NoReduce(float(self._num_dropped_stale))
            ),
            m.Metric("buffer/depth_episodes", m.NoReduce(float(depth_episodes))),
        ]
