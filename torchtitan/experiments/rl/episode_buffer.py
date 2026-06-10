# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-process episode buffer between the rollout producer and the trainer consumer.

A plain asyncio object in the controller's event loop (not a Monarch actor — it owns no GPU).
Two bounds shape the producer/consumer pipeline: a version bound (`max_offpolicy_steps`, the
staleness drop) and a depth bound (`max_buffered_batches`, how far the producer banks ahead).
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from torchtitan.experiments.rl.batcher import Batcher
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Episode, TrainingBatch
from torchtitan.observability import structured_logger as sl


def earliest_version(episode: Episode) -> int:
    """The oldest policy version the episode's tokens were sampled at (conservative staleness).

    Reads the min over `version_intervals` so a packed multi-turn episode is as stale as its
    oldest turn; falls back to `policy_version` when intervals are absent.

    Example:

        earliest_version(Episode(version_intervals=[(0, 5), (40, 6)], ...))  # -> 5
    """
    if not episode.version_intervals:
        return episode.policy_version
    return min(version for _start, version in episode.version_intervals)


@dataclass(frozen=True, slots=True)
class PackedEpisodeBatch:
    """One packed batch for the trainer: grad-accum microbatches, their global-token normalizer,
    and the rollout/episode/buffer metrics that ride out with this batch.

    Example:

        batch = await buffer.get_batch(train_version=3)
        if batch is None:  # buffer closed and drained
            return
        await trainer.forward_backward.call(batch.microbatches[0], batch.num_global_valid_tokens)
    """

    microbatches: list[list[TrainingBatch]]  # [grad_accum_steps][dp_degree]
    num_global_valid_tokens: int
    metrics: list[m.Metric]


class EpisodeBuffer:
    """A FIFO of scored `Episode`s the trainer consumes one packed batch at a time.

    Producer `put`s episodes; consumer `get_batch`es one packed batch once a full batch of fresh
    tokens is buffered. Stale episodes are dropped, not down-weighted; `put` backpressures at the
    depth bound.

    Args:
        batcher: packs peeled episodes into microbatches; also sets the batch token target.
        dp_degree: data-parallel degree, for the token target.
        max_offpolicy_steps: drop an episode whose oldest token is more than this many versions
            behind `train_version`. 0 = strict on-policy.
        max_buffered_batches: batches the producer may bank ahead before `put` backpressures.
            Set >= `max_offpolicy_steps + 1` to use the full staleness budget.

    Example:

        buffer = EpisodeBuffer(
            batcher=batcher, dp_degree=2, max_offpolicy_steps=1, max_buffered_batches=2
        )
        await buffer.put(episodes, metrics, train_version=3)        # producer
        batch = await buffer.get_batch(train_version=3)             # consumer -> PackedEpisodeBatch
    """

    def __init__(
        self,
        *,
        batcher: Batcher,
        dp_degree: int,
        max_offpolicy_steps: int,
        max_buffered_batches: int,
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
        self._episodes: deque[Episode] = deque()
        # Rollout/episode metrics waiting to ride out with the next consumed batch.
        self._pending_metrics: list[m.Metric] = []
        # Notified by put (data added) and by get_batch (space freed).
        self._cv = asyncio.Condition()
        self._closed = False
        self._num_dropped_stale = 0
        # Stale drops since the last consumed batch; the numerator of `buffer/stale_drop_rate`,
        # reset each `get_batch` so the rate is per-step, not cumulative.
        self._num_dropped_stale_since_last_get = 0

    def _episode_tokens(self, episode: Episode) -> int:
        # A packed episode contributes `len(token_ids) - 1` unpadded tokens (the [:-1]/[1:]
        # input/label split the batcher applies, see `Batcher._pack_episodes`).
        return len(episode.token_ids) - 1

    def _buffered_tokens(self) -> int:
        # Unpadded token total, compared against the batcher's padded-slot target; because packing
        # wastes slots, a peel sized to this token target can pack into more than `global_batch_size`
        # rows (see `_peel_one_batch`).
        return sum(self._episode_tokens(episode) for episode in self._episodes)

    def _batch_tokens(self) -> int:
        return self._batcher.num_tokens_target(self._dp_degree)

    def _has_full_batch(self) -> bool:
        return self._buffered_tokens() >= self._batch_tokens()

    def _at_capacity(self) -> bool:
        # The depth bound: stop the producer once it has banked `max_buffered_batches` ahead.
        return (
            self._buffered_tokens() >= self._max_buffered_batches * self._batch_tokens()
        )

    def _is_fresh(self, episode: Episode, train_version: int) -> bool:
        # The staleness-drop rule, named once: an episode is on-policy enough to keep iff its
        # oldest token is within `max_offpolicy_steps` versions of the consumer's `train_version`.
        return train_version - earliest_version(episode) <= self._max_offpolicy_steps

    def _fresh_episodes(
        self, episodes: Iterable[Episode], *, train_version: int
    ) -> tuple[list[Episode], int]:
        # Split `episodes` into the fresh ones (to keep, in order) and the count to drop as stale,
        # by the single `_is_fresh` rule. Shared by both drop sites: `put` (incoming episodes) and
        # `_drop_stale_buffered_episodes` (the whole buffer at consume time).
        fresh: list[Episode] = []
        dropped = 0
        for episode in episodes:
            if self._is_fresh(episode, train_version):
                fresh.append(episode)
            else:
                dropped += 1
        return fresh, dropped

    def _record_stale_drops(self, dropped: int) -> None:
        self._num_dropped_stale += dropped
        self._num_dropped_stale_since_last_get += dropped

    def _drop_stale_buffered_episodes(self, *, train_version: int) -> int:
        # Consume-time re-drop over the WHOLE buffer: episodes banked ahead age as `train_version`
        # advances, so a buffered episode that was fresh at `put` can be stale now. Frees space ->
        # notify the producer. (`put` itself only filters the incoming episodes, not the backlog.)
        fresh, dropped = self._fresh_episodes(
            self._episodes, train_version=train_version
        )
        if dropped:
            self._episodes = deque(fresh)
            self._record_stale_drops(dropped)
            self._cv.notify_all()
        return dropped

    def _peel_one_batch(self) -> list[Episode]:
        # Peel episodes from the front (oldest first, FIFO) up to the batcher's packed-row budget,
        # mirroring its greedy first-fit packing (`Batcher._pack_episodes`) so it receives exactly
        # `global_batch_size` rows and never truncates the surplus. The remainder stays buffered to
        # age toward the staleness bound. Counting rows (not raw tokens) is what keeps the peel
        # aligned with what the trainer actually consumes despite packing waste.
        # TODO: this re-implements (and reaches into) `Batcher`'s first-fit packing; expose a
        #   `Batcher.rows_for(episodes)` and peel against it so the two can't drift.
        seq_len = self._batcher.seq_len
        pad_multiple = self._batcher._per_sample_pad_multiple
        row_budget = max(
            1, self._batch_tokens() // seq_len
        )  # resolved global_batch_size rows
        peeled: list[Episode] = []
        rows_used = 0
        row_fill = 0
        while self._episodes:
            episode = self._episodes[0]
            sample_len = self._episode_tokens(episode)
            if pad_multiple:
                sample_len = (
                    (sample_len + pad_multiple - 1) // pad_multiple
                ) * pad_multiple
            opens_new_row = row_fill == 0 or row_fill + sample_len > seq_len
            if opens_new_row and rows_used + 1 > row_budget:
                break  # one more row would exceed the batcher's budget
            if opens_new_row:
                rows_used += 1
                row_fill = 0
            row_fill += sample_len
            peeled.append(self._episodes.popleft())
        return peeled

    async def put(
        self,
        episodes: list[Episode],
        metrics: list[m.Metric],
        *,
        train_version: int,
    ) -> None:
        """Add fresh episodes, then block (backpressure) until the buffer is below its depth.

        Args:
            episodes: Scored episodes from one rollout-collection round.
            metrics: Rollout/episode metrics for these episodes; surfaced to the consumer.
            train_version: The consumer's current policy version, for the staleness drop.
        """
        async with self._cv:
            # Admit only the fresh incoming episodes; the backlog is re-checked at consume time
            # (`_drop_stale_buffered_episodes`), which is what lets banked batches age in place.
            fresh, dropped = self._fresh_episodes(episodes, train_version=train_version)
            self._episodes.extend(fresh)
            self._record_stale_drops(dropped)
            self._pending_metrics.extend(metrics)
            self._cv.notify_all()  # wake the consumer: a full batch may be ready now
            # Backpressure: park until the buffer drops below the depth bound, so the producer
            # banks at most `max_buffered_batches` ahead (not arbitrarily far).
            await self._cv.wait_for(lambda: self._closed or not self._at_capacity())

    @sl.log_trace_span("get_batch")
    async def get_batch(self, *, train_version: int) -> PackedEpisodeBatch | None:
        """Block until a full batch of fresh episodes is buffered, then pack + return ONE batch.

        Returns a `PackedEpisodeBatch`, or `None` ONLY when closed and drained; a transient
        all-stale round re-waits for a top-up rather than returning `None`. Episodes that went
        stale (against `train_version`) while waiting are re-dropped here before packing. The
        returned metrics carry the rollout/episode metrics plus staleness/depth.
        """
        async with self._cv:
            while True:
                # The trainer idles here until enough fresh tokens are buffered (this wait is
                # the trainer-idle time the perf audit measures) — or the buffer is closed.
                await self._cv.wait_for(lambda: self._closed or self._has_full_batch())
                # Re-check staleness at consume time: train_version may have advanced while the
                # episodes waited (a swap landed, or they aged while banked), so buffered episodes
                # can now be stale. This is where the staleness drop does real work.
                self._drop_stale_buffered_episodes(train_version=train_version)
                if self._has_full_batch():
                    break
                if self._closed:
                    # Closed without a full batch: discard the partial remainder and signal done.
                    # Peeling it would under-fill the trainer's grad-accum microbatches.
                    return None
                # The re-drop left us short of a batch — wait for the producer to refill.
            buffered_tokens = self._buffered_tokens()
            buffered_episodes = len(self._episodes)
            batch = (
                self._peel_one_batch()
            )  # a full batch (the loop only breaks once full)
            dropped_since_get = self._num_dropped_stale_since_last_get
            self._num_dropped_stale_since_last_get = 0
            metrics = self._pending_metrics
            self._pending_metrics = []
            self._cv.notify_all()  # wake the producer: backpressure released
        # Fraction of episodes lost to staleness since the last batch (drops / (drops + consumed)).
        # ~0 is healthy; high means the producer is banking faster than the trainer consumes.
        consumed = len(batch)
        stale_drop_rate = (
            dropped_since_get / (dropped_since_get + consumed)
            if dropped_since_get + consumed
            else 0.0
        )
        staleness = [train_version - earliest_version(episode) for episode in batch]
        metrics += [
            m.Metric("buffer/staleness", m.Mean.from_list(staleness)),
            m.Metric("buffer/staleness", m.Max.from_list(staleness)),
            m.Metric(
                "buffer/dropped_stale", m.NoReduce(float(self._num_dropped_stale))
            ),
            m.Metric("buffer/stale_drop_rate", m.NoReduce(stale_drop_rate)),
            # Depth (measured before peeling): how many batches the producer banked ahead. ~0
            # means the producer is starving the trainer; near max_buffered_batches means it is
            # comfortably ahead and the trainer should rarely idle.
            m.Metric("buffer/depth_episodes", m.Max(float(buffered_episodes))),
            m.Metric(
                "buffer/depth_batches",
                m.Max(buffered_tokens / max(self._batch_tokens(), 1)),
            ),
        ]
        # Pack off the event loop (outside the lock): packing a large batch is CPU-bound, and
        # blocking the loop here would stall the rollout producer + generator polling. The
        # staleness drop already ran above, so the trainer trains exactly this version's batch.
        # TODO(perf): pre-pack the next batch in a standalone batcher coroutine feeding a 1-deep
        #   ready-batch queue so the trainer never waits on packing at all. Needs a drop-at-consume
        #   staleness re-check, since a speculatively packed batch's train version isn't fixed
        #   until the trainer advances.
        microbatches, num_global_valid_tokens, packing_metrics = await self._pack_batch(
            batch
        )
        return PackedEpisodeBatch(
            microbatches=microbatches,
            num_global_valid_tokens=num_global_valid_tokens,
            metrics=metrics + packing_metrics,
        )

    @sl.log_trace_span("_pack_batch")
    async def _pack_batch(
        self, batch: list[Episode]
    ) -> tuple[list[list[TrainingBatch]], int, list[m.Metric]]:
        return await asyncio.to_thread(
            self._batcher.batch, batch, dp_degree=self._dp_degree
        )

    async def close(self) -> None:
        """Mark the buffer closed and wake everyone, so blocked put/get_batch return."""
        async with self._cv:
            self._closed = True
            self._cv.notify_all()
