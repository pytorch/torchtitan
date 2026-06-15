# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.experiments.rl.episode_buffer.EpisodeBuffer`.

The buffer is a dumb FIFO: rollout workers `add_episodes`; the batcher loop `wait_for_batch_episodes`
(a snapshot), packs it via the batcher, and `pop_consumed_episodes` for the consumed count.
`_consume_one` below plays the batcher loop's role. The buffer reads the trainer's version through a
getter, so tests advance the version mid-flight to exercise the staleness drop.
"""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.batcher import PackedBatch
from torchtitan.experiments.rl.episode_buffer import (
    EpisodeBuffer,
    oldest_sampled_version,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Episode, RolloutID


class _FakeBatcher:
    """Stub batcher exposing the seam the buffer uses: `target_batch_tokens`, `num_packed_tokens`, and
    `pack_batch` (next-fit over episodes, stops at `target // seq_len` rows, reports consumed).

    The default huge `seq_len` packs every episode into one row (so one batch = all buffered episodes);
    tests that exercise the per-row peel pass a small `seq_len` to force episodes into separate rows.
    """

    def __init__(
        self, *, target: int, seq_len: int = 1_000_000, per_sample_pad_multiple=None
    ) -> None:
        self._target = target
        self.seq_len = seq_len
        self._per_sample_pad_multiple = per_sample_pad_multiple

    def target_batch_tokens(self, dp_degree: int) -> int:
        return self._target

    def num_packed_tokens(self, episode: Episode) -> int:
        length = len(episode.token_ids) - 1
        if self._per_sample_pad_multiple:
            align = self._per_sample_pad_multiple
            length = ((length + align - 1) // align) * align
        return length

    def pack_batch(self, episodes, *, dp_degree) -> PackedBatch:
        max_rows = max(1, self._target // self.seq_len)
        rows: list[list[Episode]] = []
        current: list[Episode] = []
        current_len = 0
        for episode in episodes:
            length = self.num_packed_tokens(episode)
            if current and current_len + length > self.seq_len:
                rows.append(current)
                if len(rows) == max_rows:
                    break
                current, current_len = [], 0
            current.append(episode)
            current_len += length
        else:
            if current and len(rows) < max_rows:
                rows.append(current)
        num_consumed = sum(len(row) for row in rows)
        # Stand-in num_global_valid_tokens: one "valid token" per consumed episode (tests assert counts).
        return PackedBatch(
            microbatches=["microbatch"],
            num_global_valid_tokens=num_consumed,
            num_episodes_consumed=num_consumed,
            metrics=[m.Metric("batcher/pct_pad_in_batch", m.NoReduce(0.0))],
        )


def _episode(*, version: int, prompt: int = 1, completion: int = 4) -> Episode:
    """A packed episode of `prompt + completion` tokens (so `num_packed_tokens` = prompt+completion-1)."""
    return Episode(
        policy_version=version,
        rollout_id=RolloutID(group_id="g", rollout_id=0, turn_id=0),
        token_ids=list(range(prompt + completion)),
        loss_mask=[False] * prompt + [True] * completion,
        logprobs=[0.0] * prompt + [-0.1] * completion,
        advantage=[0.0] * (prompt + completion),
        version_intervals=[(prompt, version)],
    )


def _make_buffer(
    batcher,
    *,
    max_offpolicy_steps,
    max_buffered_batches,
    version=0,
    drop_rollout_group_if_any_stale=False,
):
    """An EpisodeBuffer plus a mutable version box; set `box["v"] = N` to advance the trainer version."""
    box = {"v": version}
    buffer = EpisodeBuffer(
        batcher=batcher,
        dp_degree=1,
        max_offpolicy_steps=max_offpolicy_steps,
        drop_rollout_group_if_any_stale=drop_rollout_group_if_any_stale,
        max_buffered_batches=max_buffered_batches,
        train_version=lambda: box["v"],
    )
    return buffer, box


def _num_buffered(buffer: EpisodeBuffer) -> int:
    return sum(len(group.episodes) for group in buffer._buffered)


async def _consume_one(buffer: EpisodeBuffer, batcher: _FakeBatcher):
    """Play the pack loop: take a full fresh batch, pack it, commit. Returns (packed, metrics) or None."""
    episodes = await buffer.wait_for_batch_episodes()
    if episodes is None:
        return None
    packed = batcher.pack_batch(episodes, dp_degree=1)
    metrics = await buffer.pop_consumed_episodes(packed.num_episodes_consumed)
    return packed, metrics


def _metric_value(metrics, key, reducer_cls):
    for metric in metrics:
        if metric.key == key and isinstance(metric.value, reducer_cls):
            return metric.value.value
    return None


def test_oldest_sampled_version_uses_oldest_interval():
    assert oldest_sampled_version(_episode(version=5)) == 5
    multi = _episode(version=5)
    multi.version_intervals = [(0, 5), (100, 6)]
    assert oldest_sampled_version(multi) == 5  # oldest, conservative


def test_oldest_sampled_version_falls_back_to_policy_version():
    episode = _episode(version=7)
    episode.version_intervals = []
    assert oldest_sampled_version(episode) == 7


def test_rejects_invalid_config():
    for bad in ({"max_offpolicy_steps": -1}, {"max_buffered_batches": 0}):
        kwargs = {"max_offpolicy_steps": 1, "max_buffered_batches": 2, **bad}
        with pytest.raises(ValueError):
            _make_buffer(_FakeBatcher(target=5), **kwargs)


def test_take_blocks_until_full_then_packs():
    async def main():
        batcher = _FakeBatcher(target=10)
        buffer, _ = _make_buffer(batcher, max_offpolicy_steps=5, max_buffered_batches=2)
        take = asyncio.create_task(buffer.wait_for_batch_episodes())
        await asyncio.sleep(0.01)
        assert not take.done()  # blocked: buffer empty

        await buffer.add_episodes(
            [_episode(version=0, prompt=6, completion=5)], []
        )  # 6+5-1 = 10 >= target
        episodes = await take
        assert len(episodes) == 1

    asyncio.run(main())


def test_drops_stale_at_take_R1():
    async def main():
        batcher = _FakeBatcher(target=4)
        buffer, _ = _make_buffer(
            batcher, max_offpolicy_steps=1, max_buffered_batches=3, version=3
        )
        # version=3: v2 is 1 stale (kept), v1 is 2 stale (dropped). v2 alone is 5 tokens >= 4.
        await buffer.add_episodes(
            [_episode(version=2, completion=5), _episode(version=1, completion=5)], []
        )
        packed, _ = await _consume_one(buffer, batcher)
        assert packed.num_episodes_consumed == 1  # only the v2 episode survived
        assert buffer._num_stale_episodes_dropped_total == 1

    asyncio.run(main())


def test_max_offpolicy_none_never_drops():
    # max_offpolicy_steps=None: measurement-only. Stale episodes are kept (logged, never dropped).
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, box = _make_buffer(
            batcher, max_offpolicy_steps=None, max_buffered_batches=5
        )
        await buffer.add_episodes([_episode(version=0, completion=5)], [])
        box["v"] = 100  # wildly stale under any finite bound
        packed, _ = await _consume_one(buffer, batcher)
        assert packed.num_episodes_consumed == 1  # kept despite staleness
        assert buffer._num_stale_episodes_dropped_total == 0

    asyncio.run(main())


def test_drop_rollout_group_if_any_stale_drops_whole_round():
    # drop_rollout_group_if_any_stale: one stale episode condemns its whole add_episodes round (group),
    # so GRPO groups stay intact rather than losing only the stale siblings.
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, box = _make_buffer(
            batcher,
            max_offpolicy_steps=0,
            max_buffered_batches=5,
            drop_rollout_group_if_any_stale=True,
        )
        # round 1 (one group): a v0 + a v1 episode; round 2: a fresh v1 episode.
        await buffer.add_episodes(
            [_episode(version=0, completion=5), _episode(version=1, completion=5)], []
        )
        await buffer.add_episodes([_episode(version=1, completion=5)], [])
        box["v"] = 1  # v0 is now 1 stale (> 0) -> whole round 1 (both episodes) dropped
        episodes = await buffer.wait_for_batch_episodes()
        assert (
            buffer._num_stale_episodes_dropped_total == 2
        )  # the fresh v1 sibling went too
        assert len(episodes) == 1 and all(e.policy_version == 1 for e in episodes)

    asyncio.run(main())


def test_on_round_drained_fires_per_round_consumed_and_dropped():
    # The AdmissionBudget release hook fires once per round (group) that leaves the buffer — whether
    # consumed by a batch or dropped for staleness — so permits balance acquisitions.
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        box = {"v": 0}
        drained: list[int] = []
        buffer = EpisodeBuffer(
            batcher=batcher,
            dp_degree=1,
            max_offpolicy_steps=0,
            max_buffered_batches=5,
            train_version=lambda: box["v"],
            on_round_drained=lambda: drained.append(1),
        )
        await buffer.add_episodes([_episode(version=0, completion=5)], [])
        await _consume_one(buffer, batcher)  # round consumed -> drained once
        assert len(drained) == 1

        await buffer.add_episodes([_episode(version=0, completion=5)], [])  # a v0 round
        box["v"] = 1  # now stale; the next wait drops the whole round
        take = asyncio.create_task(buffer.wait_for_batch_episodes())
        await asyncio.sleep(0.01)
        assert len(drained) == 2  # the dropped round also fired the hook
        take.cancel()
        try:
            await take
        except asyncio.CancelledError:
            pass

    asyncio.run(main())


def test_drops_stale_against_live_version_during_wait_C1():
    # Codex blocker 1: while wait_for_batch_episodes waits, the trainer advances its version. The drop must use
    # the LIVE version, not the one at take-call time — otherwise an episode that aged past the bound
    # during the wait would still be packed and train (R1 broken).
    async def main():
        batcher = _FakeBatcher(
            target=10, seq_len=10
        )  # one row, needs two 5-token episodes
        buffer, box = _make_buffer(
            batcher, max_offpolicy_steps=0, max_buffered_batches=5
        )
        await buffer.add_episodes(
            [_episode(version=0, completion=5)], []
        )  # 5 tokens < 10 -> take waits
        take = asyncio.create_task(buffer.wait_for_batch_episodes())
        await asyncio.sleep(0.01)
        assert not take.done()

        box["v"] = 1  # version advances: the banked v0 episode is now 1 stale (> 0)
        await buffer.add_episodes(
            [_episode(version=1, completion=5)], []
        )  # wakes take; drop runs at v=1
        await asyncio.sleep(0.01)
        assert (
            not take.done()
        )  # v0 dropped (live version), only the fresh v1 left -> still < 10
        assert buffer._num_stale_episodes_dropped_total == 1

        await buffer.add_episodes(
            [_episode(version=1, completion=5)], []
        )  # second fresh v1 -> full
        episodes = await take
        assert len(episodes) == 2 and all(
            e.policy_version == 1 for e in episodes
        )  # no stale v0

    asyncio.run(main())


def test_peels_one_batch_leaves_remainder_R4():
    async def main():
        batcher = _FakeBatcher(
            target=5, seq_len=5
        )  # 1 row of 5 tokens -> one 5-token episode/batch
        buffer, _ = _make_buffer(batcher, max_offpolicy_steps=5, max_buffered_batches=5)
        await buffer.add_episodes(
            [_episode(version=0, completion=5), _episode(version=0, completion=5)], []
        )
        first, _ = await _consume_one(buffer, batcher)
        assert first.num_episodes_consumed == 1  # exactly ONE, not both
        assert _num_buffered(buffer) == 1  # remainder left buffered
        second, _ = await _consume_one(buffer, batcher)
        assert second.num_episodes_consumed == 1

    asyncio.run(main())


def test_pack_respects_pad_multiple_R5():
    async def main():
        batcher = _FakeBatcher(target=10, seq_len=10, per_sample_pad_multiple=8)
        buffer, _ = _make_buffer(batcher, max_offpolicy_steps=5, max_buffered_batches=5)
        # Two 5-token episodes share one row unpadded; padded to 8 they need two rows (budget=1).
        await buffer.add_episodes(
            [_episode(version=0, completion=5), _episode(version=0, completion=5)], []
        )
        packed, _ = await _consume_one(buffer, batcher)
        assert (
            packed.num_episodes_consumed == 1
        )  # only one padded episode fits the single row
        assert _num_buffered(buffer) == 1

    asyncio.run(main())


def test_producer_running_ahead_makes_staleness_exceed_one():
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, box = _make_buffer(
            batcher, max_offpolicy_steps=5, max_buffered_batches=5
        )
        for _ in range(3):  # bank 3 batches ahead, all at v0
            await buffer.add_episodes([_episode(version=0, completion=5)], [])
        _, m0 = await _consume_one(buffer, batcher)
        box["v"] = 1
        _, m1 = await _consume_one(buffer, batcher)
        box["v"] = 2
        _, m2 = await _consume_one(buffer, batcher)
        assert _metric_value(m0, "perf/buffer_staleness_max", m.NoReduce) == 0
        assert _metric_value(m1, "perf/buffer_staleness_max", m.NoReduce) == 1
        assert _metric_value(m2, "perf/buffer_staleness_max", m.NoReduce) == 2

    asyncio.run(main())


def test_batch_aged_past_bound_is_dropped_at_consume():
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, box = _make_buffer(
            batcher, max_offpolicy_steps=1, max_buffered_batches=5
        )
        for _ in range(2):
            await buffer.add_episodes([_episode(version=0, completion=5)], [])
        await _consume_one(buffer, batcher)  # consume one fresh batch at v0
        box["v"] = 2  # the remaining v0 batch is now 2 versions stale (> 1)
        take = asyncio.create_task(
            buffer.wait_for_batch_episodes()
        )  # re-dropped, take re-waits
        await asyncio.sleep(0.01)
        assert not take.done()
        assert buffer._num_stale_episodes_dropped_total == 1
        await buffer.add_episodes(
            [_episode(version=2, completion=5)], []
        )  # fresh v2 unblocks
        episodes = await take
        assert len(episodes) == 1

    asyncio.run(main())


def test_close_returns_none_when_drained():
    async def main():
        buffer, _ = _make_buffer(
            _FakeBatcher(target=100), max_offpolicy_steps=5, max_buffered_batches=2
        )
        await buffer.close()
        assert await buffer.wait_for_batch_episodes() is None

    asyncio.run(main())


def test_partial_on_close_returns_none_R2():
    async def main():
        buffer, _ = _make_buffer(
            _FakeBatcher(target=100), max_offpolicy_steps=5, max_buffered_batches=2
        )
        await buffer.add_episodes(
            [_episode(version=0, completion=5)], []
        )  # far below target
        await buffer.close()
        assert await buffer.wait_for_batch_episodes() is None  # partial discarded

    asyncio.run(main())


def test_producer_consumer_no_deadlock_R3():
    async def main():
        batcher = _FakeBatcher(target=10)
        buffer, _ = _make_buffer(batcher, max_offpolicy_steps=5, max_buffered_batches=2)
        consumed = 0

        async def producer():
            for _ in range(100):
                await buffer.add_episodes(
                    [_episode(version=0, prompt=6, completion=5)], []
                )

        async def consumer():
            nonlocal consumed
            for _ in range(3):
                assert await _consume_one(buffer, batcher) is not None
                consumed += 1

        producer_task = asyncio.create_task(producer())
        await asyncio.wait_for(consumer(), timeout=2.0)
        await buffer.close()
        producer_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer_task
        assert consumed == 3

    asyncio.run(main())


def test_all_stale_redrop_keeps_waiting_not_none():
    async def main():
        batcher = _FakeBatcher(target=5)
        buffer, _ = _make_buffer(
            batcher, max_offpolicy_steps=0, max_buffered_batches=2, version=1
        )
        take = asyncio.create_task(buffer.wait_for_batch_episodes())
        await buffer.add_episodes([_episode(version=0, prompt=1, completion=5)], [])
        await asyncio.sleep(0.01)
        assert not take.done()  # all-stale re-drop re-waits, does NOT return None
        await buffer.add_episodes([_episode(version=1, prompt=1, completion=5)], [])
        episodes = await take
        assert len(episodes) == 1  # the fresh v1 episode

    asyncio.run(main())


def test_commit_reports_depth_R10():
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, _ = _make_buffer(batcher, max_offpolicy_steps=5, max_buffered_batches=5)
        for _ in range(3):  # 3 batches buffered
            await buffer.add_episodes([_episode(version=0, completion=5)], [])
        _, metrics = await _consume_one(buffer, batcher)
        # one batch consumed -> 2 batches (10 tokens) of 5-token target remain
        assert _metric_value(metrics, "perf/buffer_depth_batches", m.NoReduce) == 2.0
        assert _metric_value(metrics, "buffer/depth_episodes", m.NoReduce) == 2.0

    asyncio.run(main())


def test_stale_drop_rate_R7():
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, box = _make_buffer(
            batcher, max_offpolicy_steps=0, max_buffered_batches=10
        )
        await buffer.add_episodes([_episode(version=0, completion=5)], [])
        await buffer.add_episodes(
            [_episode(version=1, completion=5)], []
        )  # put leaves the aged v0 in place
        assert (
            _num_buffered(buffer) == 2 and buffer._num_stale_episodes_dropped_total == 0
        )
        box["v"] = 1
        _, metrics = await _consume_one(buffer, batcher)  # drops v0, keeps v1
        assert buffer._num_stale_episodes_dropped_total == 1
        # 1 dropped / (1 dropped + 1 consumed) since the last commit.
        assert _metric_value(metrics, "perf/buffer_stale_drop_rate", m.NoReduce) == 0.5

    asyncio.run(main())


def test_metrics_ride_with_their_batch_R13():
    # Metrics ride out at commit, with the batch that consumes their group. A put arriving between take
    # and commit must ride with its OWN (next) batch, not be swept out with this one.
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, _ = _make_buffer(batcher, max_offpolicy_steps=5, max_buffered_batches=5)
        await buffer.add_episodes(
            [_episode(version=0, completion=5)], [m.Metric("rollout/a", m.Sum(1.0))]
        )
        episodes = await buffer.wait_for_batch_episodes()
        # A put lands DURING the (here, simulated) pack, before commit.
        await buffer.add_episodes(
            [_episode(version=0, completion=5)], [m.Metric("rollout/b", m.Sum(1.0))]
        )
        packed = batcher.pack_batch(episodes, dp_degree=1)
        ride_out = await buffer.pop_consumed_episodes(packed.num_episodes_consumed)
        assert "rollout/a" in [metric.key for metric in ride_out]  # this batch's group
        assert "rollout/b" not in [
            metric.key for metric in ride_out
        ]  # not the next group's
        # The second batch carries rollout/b (not lost).
        _, ride_out2 = await _consume_one(buffer, batcher)
        assert "rollout/b" in [metric.key for metric in ride_out2]

    asyncio.run(main())


def test_dropped_group_metrics_never_emitted_C2():
    # Codex blocker 2: a group dropped for staleness must NOT log its rollout metrics — that data never
    # trained, so reporting its reward would make the logs lie about what ran.
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, box = _make_buffer(
            batcher, max_offpolicy_steps=0, max_buffered_batches=5
        )
        await buffer.add_episodes(
            [_episode(version=0, completion=5)], [m.Metric("rollout/stale", m.Sum(1.0))]
        )
        box["v"] = 1  # the v0 group is now stale (> 0) and will be dropped whole
        await buffer.add_episodes(
            [_episode(version=1, completion=5)], [m.Metric("rollout/fresh", m.Sum(1.0))]
        )
        _, ride_out = await _consume_one(buffer, batcher)
        keys = [metric.key for metric in ride_out]
        assert "rollout/fresh" in keys  # the trained group's metrics ride
        assert (
            "rollout/stale" not in keys
        )  # the dropped group's metrics are gone, not logged
        assert buffer._num_stale_episodes_dropped_total == 1

    asyncio.run(main())


def test_surplus_group_keeps_its_metrics_C3():
    # Codex blocker 3: when more than one batch is buffered, consuming the first batch must leave the
    # surplus group's metrics for the batch that consumes it — not clear them at take.
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)  # one 5-token episode per batch
        buffer, _ = _make_buffer(batcher, max_offpolicy_steps=5, max_buffered_batches=5)
        await buffer.add_episodes(
            [_episode(version=0, completion=5)], [m.Metric("rollout/first", m.Sum(1.0))]
        )
        await buffer.add_episodes(
            [_episode(version=0, completion=5)],
            [m.Metric("rollout/second", m.Sum(1.0))],
        )
        _, ride_out1 = await _consume_one(buffer, batcher)  # consumes the first group
        assert [
            k for k in (mm.key for mm in ride_out1) if k.startswith("rollout/")
        ] == ["rollout/first"]
        _, ride_out2 = await _consume_one(
            buffer, batcher
        )  # the surplus group, metrics intact
        assert [
            k for k in (mm.key for mm in ride_out2) if k.startswith("rollout/")
        ] == ["rollout/second"]

    asyncio.run(main())


def test_failed_round_metrics_drain_with_next_commit():
    # A round can fail and `put` zero episodes but real metrics (e.g. a group-level failure count).
    # Those metrics have no episodes to ride with, so they drain out with the next commit.
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, _ = _make_buffer(batcher, max_offpolicy_steps=5, max_buffered_batches=5)
        await buffer.add_episodes(
            [], [m.Metric("rollout/failures", m.Sum(2.0))]
        )  # no episodes
        await buffer.add_episodes(
            [_episode(version=0, completion=5)], [m.Metric("rollout/ok", m.Sum(1.0))]
        )
        _, ride_out = await _consume_one(buffer, batcher)
        keys = [metric.key for metric in ride_out]
        assert (
            "rollout/failures" in keys and "rollout/ok" in keys
        )  # both drain with this batch

    asyncio.run(main())


def test_commit_after_take_is_unconditional_safe_R6():
    # The pack loop may be cancelled between take and commit (the off-loop pack). The snapshot does not
    # pop, so a missed commit leaves the buffer consistent (no half-popped state).
    async def main():
        batcher = _FakeBatcher(target=5, seq_len=5)
        buffer, _ = _make_buffer(batcher, max_offpolicy_steps=5, max_buffered_batches=5)
        await buffer.add_episodes([_episode(version=0, completion=5)], [])
        episodes = await buffer.wait_for_batch_episodes()
        assert episodes is not None and _num_buffered(buffer) == 1  # take did NOT pop
        # Simulate cancellation before commit: the episode is still buffered and consumable.
        packed, _ = await _consume_one(buffer, batcher)
        assert packed.num_episodes_consumed == 1 and _num_buffered(buffer) == 0

    asyncio.run(main())
