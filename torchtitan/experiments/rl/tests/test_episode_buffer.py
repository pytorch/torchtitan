# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.experiments.rl.episode_buffer.EpisodeBuffer`."""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.episode_buffer import earliest_version, EpisodeBuffer
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Episode


class _FakeBatcher:
    """Stub Batcher: a fixed token target + seq_len for the row-budget peel, and a trivial pack
    that reports the episode count. The default huge `seq_len` makes every episode share one row
    (so `_peel_one_batch` takes the whole buffer); tests that exercise the packed-row peel pass a
    small `seq_len` to force episodes into separate rows."""

    def __init__(self, *, target: int, seq_len: int = 1_000_000) -> None:
        self._target = target
        self.seq_len = seq_len
        self._per_sample_pad_multiple = None

    def num_tokens_target(self, dp_degree: int) -> int:
        return self._target

    def batch(self, episodes, *, dp_degree):
        return (
            ["microbatch"],
            len(episodes),
            [],
        )  # (microbatches, num_valid, packing_metrics)


def _episode(*, version: int, prompt: int = 1, completion: int = 4) -> Episode:
    """A packed episode of `prompt + completion` tokens (so the buffer counts
    `len(token_ids) - 1 = prompt + completion - 1` unpadded tokens), prompt untrained."""
    return Episode(
        policy_version=version,
        sample_id="s",
        token_ids=list(range(prompt + completion)),
        loss_mask=[False] * prompt + [True] * completion,
        logprobs=[0.0] * prompt + [-0.1] * completion,
        advantage=[0.0] * (prompt + completion),
        version_intervals=[(prompt, version)],
    )


def _metric_value(metrics, key, reducer_cls) -> float | None:
    """Read back the value of `key`'s `reducer_cls` payload from a get_batch metrics list."""
    for metric in metrics:
        if metric.key == key and isinstance(metric.value, reducer_cls):
            return metric.value.value
    return None


def test_earliest_version_uses_oldest_interval():
    assert earliest_version(_episode(version=5)) == 5
    multi = _episode(version=5)
    multi.version_intervals = [(0, 5), (100, 6)]
    assert earliest_version(multi) == 5  # oldest, conservative


def test_earliest_version_falls_back_to_policy_version():
    episode = _episode(version=7)
    episode.version_intervals = []
    assert earliest_version(episode) == 7


def test_rejects_invalid_config():
    with pytest.raises(ValueError):
        EpisodeBuffer(
            batcher=_FakeBatcher(target=5),
            dp_degree=1,
            max_offpolicy_steps=-1,
            max_buffered_batches=2,
        )
    with pytest.raises(ValueError):
        EpisodeBuffer(
            batcher=_FakeBatcher(target=5),
            dp_degree=1,
            max_offpolicy_steps=1,
            max_buffered_batches=0,
        )


def test_get_batch_blocks_until_full_then_packs():
    async def main():
        # target=10 tokens; one episode contributes prompt+completion-1 tokens.
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=10),
            dp_degree=1,
            max_offpolicy_steps=5,
            max_buffered_batches=2,
        )
        get_task = asyncio.create_task(buffer.get_batch(train_version=0))
        await asyncio.sleep(0.01)
        assert not get_task.done()  # blocked: buffer empty

        # 6 + 5 - 1 = 10 tokens -> reaches the target.
        await buffer.put(
            [_episode(version=0, prompt=6, completion=5)], [], train_version=0
        )
        batch = await get_task
        assert (
            batch.microbatches == ["microbatch"] and batch.num_global_valid_tokens == 1
        )

    asyncio.run(main())


def test_drops_episodes_older_than_max_offpolicy_at_admission():
    async def main():
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=4),
            dp_degree=1,
            max_offpolicy_steps=1,
            max_buffered_batches=3,
        )
        # train_version=3: v2 is 1 stale (kept), v1 is 2 stale (dropped at admission). The v2
        # episode alone is 1 + 5 - 1 = 5 tokens >= target 4, so the batch is ready.
        await buffer.put(
            [_episode(version=2, completion=5), _episode(version=1, completion=5)],
            [],
            train_version=3,
        )
        batch = await buffer.get_batch(train_version=3)
        assert batch.num_global_valid_tokens == 1  # only the v2 episode survived
        assert buffer._num_dropped_stale == 1

    asyncio.run(main())


def test_get_batch_peels_one_batch_and_leaves_remainder():
    async def main():
        # target=5; two 5-token episodes = two batches buffered at once.
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=5, seq_len=5),
            dp_degree=1,
            max_offpolicy_steps=5,
            max_buffered_batches=5,
        )
        await buffer.put(
            [
                _episode(version=0, completion=5),
                _episode(version=0, completion=5),
            ],
            [],
            train_version=0,
        )
        batch1 = await buffer.get_batch(train_version=0)
        assert batch1.num_global_valid_tokens == 1  # exactly ONE batch, not both
        batch2 = await buffer.get_batch(train_version=0)
        assert (
            batch2.num_global_valid_tokens == 1
        )  # the remainder was left buffered for the next step

    asyncio.run(main())


def test_producer_running_ahead_makes_staleness_exceed_one():
    # The whole point of the depth bound: with the producer banked several batches ahead (all
    # sampled at v0), batches age step by step as the trainer advances, so staleness reaches 2 —
    # impossible with the old depth-1 buffer, which pinned staleness at 1.
    async def main():
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=5, seq_len=5),
            dp_degree=1,
            max_offpolicy_steps=5,
            max_buffered_batches=5,
        )
        for _ in range(3):  # bank 3 batches ahead, all at v0
            await buffer.put([_episode(version=0, completion=5)], [], train_version=0)
        # Consume them as the trainer advances v0 -> v1 -> v2.
        b0 = await buffer.get_batch(train_version=0)
        b1 = await buffer.get_batch(train_version=1)
        b2 = await buffer.get_batch(train_version=2)
        assert _metric_value(b0.metrics, "buffer/staleness", m.Max) == 0
        assert _metric_value(b1.metrics, "buffer/staleness", m.Max) == 1
        assert _metric_value(b2.metrics, "buffer/staleness", m.Max) == 2  # > 1

    asyncio.run(main())


def test_batch_aged_past_bound_is_dropped_at_consume():
    # A batch banked ahead can age past max_offpolicy_steps before it's consumed; the staleness
    # drop then fires (it never did in the depth-1 design, dropped_stale was always 0).
    async def main():
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=5, seq_len=5),
            dp_degree=1,
            max_offpolicy_steps=1,
            max_buffered_batches=5,
        )
        for _ in range(2):
            await buffer.put([_episode(version=0, completion=5)], [], train_version=0)
        await buffer.get_batch(train_version=0)  # consume one fresh batch
        # The remaining v0 batch is now 2 versions stale (> 1) -> re-dropped, consumer re-waits.
        get_task = asyncio.create_task(buffer.get_batch(train_version=2))
        await asyncio.sleep(0.01)
        assert not get_task.done()
        assert buffer._num_dropped_stale == 1
        # A fresh v2 batch lets the consumer proceed; dropped_stale rides out in the metrics.
        await buffer.put([_episode(version=2, completion=5)], [], train_version=2)
        batch = await get_task
        assert batch.num_global_valid_tokens == 1
        assert _metric_value(batch.metrics, "buffer/dropped_stale", m.NoReduce) == 1.0

    asyncio.run(main())


def test_close_returns_none_when_drained():
    async def main():
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=100),
            dp_degree=1,
            max_offpolicy_steps=5,
            max_buffered_batches=2,
        )
        await buffer.close()
        assert await buffer.get_batch(train_version=0) is None  # closed + empty

    asyncio.run(main())


def test_close_with_partial_batch_returns_none():
    # A partial (under-full) batch left when the buffer closes must NOT be peeled — peeling it
    # would under-fill the trainer's grad-accum microbatches. get_batch returns None instead.
    async def main():
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=100),  # one 5-token episode is far below target
            dp_degree=1,
            max_offpolicy_steps=5,
            max_buffered_batches=2,
        )
        await buffer.put([_episode(version=0, completion=5)], [], train_version=0)
        await buffer.close()
        assert (
            await buffer.get_batch(train_version=0) is None
        )  # partial remainder discarded

    asyncio.run(main())


def test_producer_consumer_run_without_deadlock():
    # A producer fills batches under backpressure while a consumer drains them; both finish.
    async def main():
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=10),
            dp_degree=1,
            max_offpolicy_steps=5,
            max_buffered_batches=2,
        )
        consumed = 0

        async def producer():
            # Far more rounds than the consumer drains, so the producer is still parked on
            # backpressure when cancelled (the CancelledError assertion below checks that).
            for _ in range(100):
                await buffer.put(
                    [_episode(version=0, prompt=6, completion=5)], [], train_version=0
                )

        async def consumer():
            nonlocal consumed
            for _ in range(3):
                batch = await buffer.get_batch(train_version=0)
                assert batch is not None
                consumed += 1

        producer_task = asyncio.create_task(producer())
        await asyncio.wait_for(consumer(), timeout=2.0)  # deadlock -> TimeoutError
        await buffer.close()
        producer_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer_task
        assert consumed == 3

    asyncio.run(main())


def test_all_stale_redrop_keeps_waiting_not_none():
    # The strict-on-policy regression (max_offpolicy_steps=0): a full batch admitted at v0 goes
    # stale the instant the trainer swaps to v1. get_batch must re-drop it and KEEP WAITING for a
    # fresh round — never return None (which the consumer reads as "closed", ending training early).
    async def main():
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=5),
            dp_degree=1,
            max_offpolicy_steps=0,
            max_buffered_batches=2,
        )
        # Consumer trains at v1; a full batch sampled at v0 is already buffered.
        get_task = asyncio.create_task(buffer.get_batch(train_version=1))
        await buffer.put(
            [_episode(version=0, prompt=1, completion=5)], [], train_version=0
        )
        await asyncio.sleep(0.01)
        assert not get_task.done()  # all-stale re-drop re-waits, does NOT return None
        # A fresh v1 round arrives -> the consumer finally gets a real batch.
        await buffer.put(
            [_episode(version=1, prompt=1, completion=5)], [], train_version=1
        )
        batch = await get_task
        assert batch.num_global_valid_tokens == 1  # the fresh v1 episode

    asyncio.run(main())


def test_partial_redrop_keeps_survivors_and_waits_for_topup():
    # When a re-drop removes only SOME episodes, the fresh survivors are kept (not wasted) and the
    # consumer waits for the producer to top the batch back up to the token target — it never
    # returns an underfilled batch.
    async def main():
        # target=10; each episode is 1 + 5 - 1 = 5 tokens, so two are needed for a full batch.
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=10),
            dp_degree=1,
            max_offpolicy_steps=1,
            max_buffered_batches=3,
        )
        get_task = asyncio.create_task(buffer.get_batch(train_version=2))
        # Admitted at v1 (both within 1 of v1). At consume-time v2, the v0 episode is 2 stale
        # (dropped) but the v1 episode is 1 stale (kept) -> batch now half-full.
        await buffer.put(
            [_episode(version=1, completion=5), _episode(version=0, completion=5)],
            [],
            train_version=1,
        )
        await asyncio.sleep(0.01)
        assert (
            not get_task.done()
        )  # half-full after the drop -> wait for a top-up, not return
        # A fresh v2 episode tops the batch back up to full.
        await buffer.put([_episode(version=2, completion=5)], [], train_version=2)
        batch = await get_task
        assert (
            batch.num_global_valid_tokens == 2
        )  # the kept v1 survivor + the new v2 episode

    asyncio.run(main())


def test_producer_crash_closes_buffer_and_unblocks_consumer():
    # Mirrors train()'s done-callback wiring: a producer that raises must close the buffer so the
    # consumer's get_batch returns (None) instead of hanging forever, and the error still surfaces.
    async def main():
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=100),
            dp_degree=1,
            max_offpolicy_steps=1,
            max_buffered_batches=2,
        )

        async def crashing_producer():
            raise RuntimeError("dataset blew up")

        producer = asyncio.create_task(crashing_producer())
        close_tasks: list[asyncio.Task] = []

        def _close_if_crashed(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception() is not None:
                close_tasks.append(asyncio.create_task(buffer.close()))

        producer.add_done_callback(_close_if_crashed)
        # Without the callback this would block forever (the buffer never fills or closes).
        batch = await asyncio.wait_for(buffer.get_batch(train_version=0), timeout=2.0)
        assert batch is None
        with pytest.raises(RuntimeError, match="dataset blew up"):
            await producer

    asyncio.run(main())


def test_drop_stale_helper_preserves_admission_and_consume_sites():
    # The staleness rule has one predicate but two drop SITES with different scopes: `put` filters
    # only the INCOMING episodes (leaving an aged backlog in place), and `get_batch` re-drops the
    # WHOLE buffer at consume. This pins that split so the helper refactor can't move it.
    async def main():
        buffer = EpisodeBuffer(
            batcher=_FakeBatcher(target=5, seq_len=5),
            dp_degree=1,
            max_offpolicy_steps=0,  # strict: an episode is stale one version later
            max_buffered_batches=10,  # high, so put() never backpressures here
        )
        # Admit v0 at train_version=0 (fresh). Then admit v1 at train_version=1 (fresh); v0 is now
        # 1 version stale, but put() must leave the already-buffered backlog alone.
        await buffer.put([_episode(version=0, completion=5)], [], train_version=0)
        await buffer.put([_episode(version=1, completion=5)], [], train_version=1)
        assert (
            len(buffer._episodes) == 2
        )  # put kept the aged v0; it filters incoming only
        assert buffer._num_dropped_stale == 0  # no backlog re-drop during put

        # Consume at v1: the whole-buffer re-check drops the stale v0, keeps v1.
        batch = await buffer.get_batch(train_version=1)
        assert batch.num_global_valid_tokens == 1  # only v1 survived
        assert buffer._num_dropped_stale == 1  # the consume-time re-drop fired
        # 1 dropped / (1 dropped + 1 consumed) since the last batch.
        assert _metric_value(batch.metrics, "buffer/stale_drop_rate", m.NoReduce) == 0.5

    asyncio.run(main())


def test_peel_one_batch_respects_per_sample_pad_multiple():
    # With a per-sample pad multiple, each episode's packed length rounds up, so fewer fit per row.
    # The peel must mirror that padded packing (two 5-token episodes would share one row unpadded,
    # but padded to 8 they need two rows — and the row budget here is 1).
    async def main():
        batcher = _FakeBatcher(target=10, seq_len=10)
        batcher._per_sample_pad_multiple = 8
        buffer = EpisodeBuffer(
            batcher=batcher, dp_degree=1, max_offpolicy_steps=5, max_buffered_batches=5
        )
        await buffer.put(
            [_episode(version=0, completion=5), _episode(version=0, completion=5)],
            [],
            train_version=0,
        )
        batch = await buffer.get_batch(train_version=0)
        assert (
            batch.num_global_valid_tokens == 1
        )  # only one padded episode fits the single row
        assert len(buffer._episodes) == 1  # the remainder stays buffered

    asyncio.run(main())
