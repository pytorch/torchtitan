# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer-loop tests: the overlapped weight-sync order, the divergence gate, the producer-crash
supervisor, and the pure metric/reduce helpers. Lifecycle is driven through stub actors; no GPUs."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.batcher import PackedBatch
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.tests.test_shutdown import _make_stub_rl_trainer
from torchtitan.experiments.rl.trainer import (
    _build_step_metrics,
    _generation_metrics,
    _reduce_microbatches,
)


class _ValueMesh:
    """Stand-in for a Monarch ValueMesh; `_get_rank_0_value` reads `.get(0)`."""

    def __init__(self, value):
        self._value = value

    def get(self, _rank):
        return self._value


class _Awaitable:
    """Awaitable that is NOT a coroutine, so the fire-and-forget `sync_log_step.call` won't warn."""

    def __init__(self, value):
        self._value = value

    def __await__(self):
        yield from ()
        return self._value


class _Endpoint:
    """Records each `.call()` by name and returns an awaitable `ValueMesh(result)`."""

    def __init__(self, name, calls, result=None):
        self._name = name
        self._calls = calls
        self._result = result

    def call(self, *args, **kwargs):
        self._calls.append(self._name)
        return _Awaitable(_ValueMesh(self._result))


_BATCH = PackedBatch(
    microbatches=[["mb"]],
    num_global_valid_tokens=10,
    num_episodes_consumed=4,
    metrics=[],
)
_WEIGHT_SYNC_ORDER = ["forward_backward", "optim_step", "push", "pull"]


def _ready_with(batch) -> asyncio.Queue:
    """A ready queue that serves one batch, then the None sentinel (closed + drained)."""
    ready: asyncio.Queue = asyncio.Queue()
    ready.put_nowait(batch)
    ready.put_nowait(None)
    return ready


def _stub_trainer_for_step(calls, *, loss):
    rl = _make_stub_rl_trainer()
    rl.trainer = SimpleNamespace(
        sync_log_step=_Endpoint("sync_log_step/trainer", calls),
        forward_backward=_Endpoint("forward_backward", calls, {"loss/mean": loss}),
        optim_step=_Endpoint(
            "optim_step", calls, SimpleNamespace(policy_version=7, metrics={})
        ),
        push_model_state_dict=_Endpoint("push", calls),
    )

    async def _fanout(method_name, *args, **kwargs):
        calls.append(f"router_fanout/{method_name}")
        return [None]

    async def _pull(*, policy_version):
        calls.append("pull")

    rl.generator_router = SimpleNamespace(fanout=_fanout, pull_model_state_dict=_pull)
    rl.metrics_processor = SimpleNamespace(log=lambda **kwargs: None)
    rl._train_version = 0
    return rl


def test_overlapped_weight_sync_advances_on_completion():
    # Must-preserve: fwd/bwd -> optim -> push -> (fire) pull. The pull is fired as a task and awaited
    # before the next publish / at the final drain; _train_version advances to the version that pull
    # made the generator adopt (advance ON completion, §5.2).
    async def main():
        calls: list[str] = []
        rl = _stub_trainer_for_step(calls, loss=0.5)
        await rl._consume_and_train(_ready_with(_BATCH), num_steps=5)
        assert [c for c in calls if c in _WEIGHT_SYNC_ORDER] == _WEIGHT_SYNC_ORDER
        assert rl._train_version == 7  # advanced to the pulled version, on completion

    asyncio.run(main())


def test_nan_loss_skips_optim_and_sync():
    # Must-preserve: a NaN/Inf loss breaks BEFORE optim/push/pull, so a bad step never publishes
    # weights and never advances the version.
    async def main():
        calls: list[str] = []
        rl = _stub_trainer_for_step(calls, loss=math.nan)
        await rl._consume_and_train(_ready_with(_BATCH), num_steps=5)
        assert "forward_backward" in calls
        assert "optim_step" not in calls
        assert "push" not in calls and "pull" not in calls
        assert rl._train_version == 0  # never advanced

    asyncio.run(main())


def test_run_rollout_producer_closes_buffer_on_crash_and_reraises():
    # Must-preserve: the producer supervisor closes the buffer on the way out (so the pack loop's
    # take_full_batch returns None and the trainer unblocks) and re-raises the real exception.
    async def main():
        rl = _make_stub_rl_trainer()
        closed: list[bool] = []

        class _Buffer:
            async def close(self):
                closed.append(True)

        async def _boom(_buffer):
            raise RuntimeError("producer blew up")

        rl._produce_rollouts = _boom
        with pytest.raises(RuntimeError, match="producer blew up"):
            await rl._run_rollout_producer(_Buffer())
        assert closed == [True]  # buffer closed in finally despite the crash

    asyncio.run(main())


def test_build_step_metrics_derives_perf_ratios():
    metrics = _build_step_metrics(
        batch=PackedBatch(
            microbatches=[["mb"]],
            num_global_valid_tokens=100,
            num_episodes_consumed=4,
            metrics=[m.Metric("perf/buffer_depth_batches", m.NoReduce(2.0))],
        ),
        fwd_bwd={"loss/mean": 0.5},
        optimizer={"train/lr": 1e-4},
        wait_s=8.0,
        train_s=1.0,
        push_s=0.2,
        pull_wait_s=0.3,
        sync_s=0.5,
        step_s=10.0,
    )
    by_key = {
        metric.key: metric.value.value
        for metric in metrics
        if isinstance(metric.value, m.NoReduce)
    }
    assert by_key["perf/trainer_idle_ratio"] == pytest.approx(0.8)  # 8 / 10
    assert by_key["perf/weight_sync_overhead_ratio"] == pytest.approx(0.05)  # 0.5 / 10
    assert by_key["perf/goodput_tokens_per_second"] == pytest.approx(10.0)  # 100 / 10
    # active throughput divides by TRAIN time, not the whole step (which would just be goodput):
    assert by_key["perf/active_tokens_per_second"] == pytest.approx(100.0)
    assert by_key["perf/buffer_depth_batches"] == pytest.approx(
        2.0
    )  # rode in batch.metrics


def test_reduce_microbatches_sums_real_grpo_loss_keys():
    # The REAL GRPO loss keys end in `_mean`/`_frac` (not `/mean`/`/frac`); the reduction must still sum
    # them (they are pre-normalized) or they report ~1/grad_accum of the truth. `/max` takes the max.
    reduced = _reduce_microbatches(
        [
            {
                "loss/mean": 0.4,
                "loss/ratio_mean": 1.0,
                "loss/ratio_clipped_frac": 0.1,
                "grad/x/max": 2.0,
            },
            {
                "loss/mean": 0.1,
                "loss/ratio_mean": 1.1,
                "loss/ratio_clipped_frac": 0.2,
                "grad/x/max": 5.0,
            },
        ]
    )
    assert reduced["loss/mean"] == pytest.approx(0.5)  # /mean -> summed
    assert reduced["loss/ratio_mean"] == pytest.approx(
        2.1
    )  # _mean -> summed (the bug Opus caught)
    assert reduced["loss/ratio_clipped_frac"] == pytest.approx(0.3)  # _frac -> summed
    assert reduced["grad/x/max"] == pytest.approx(5.0)  # /max -> maxed


def test_setup_async_rejects_rollouts_that_can_exceed_batch_seq_len():
    # The seq-len fail-fast guard: a rollout that can produce an episode longer than the batcher's
    # seq_len would silently under-fill packing and crash the trainer, so setup_async raises first.
    async def main():
        rl = _make_stub_rl_trainer()
        rl.config = SimpleNamespace(
            num_groups_per_rollout_batch=1,
            group_size=1,
            num_validation_samples=1,
            batcher=SimpleNamespace(batch=SimpleNamespace(seq_len=2048)),
            model_spec=SimpleNamespace(model=SimpleNamespace(max_seq_len=4096)),
            rollouter=SimpleNamespace(
                token_env=SimpleNamespace(max_rollout_tokens=4096)
            ),
        )
        with pytest.raises(ValueError, match="smaller than the longest episode"):
            await rl.setup_async(trainer_mesh=object(), generator_meshes=[object()])

    asyncio.run(main())


def test_pack_loop_sentinel_does_not_deadlock_when_ready_is_full():
    # The pack loop's finally must put the None sentinel NON-blocking. At a clean stop the trainer has
    # stopped consuming and `ready` may hold an unconsumed pre-packed batch (full); a blocking put there
    # deadlocks the cancel (the single cancel was already consumed). Regression test for that hang.
    async def main():
        from torchtitan.experiments.rl.episode_buffer import EpisodeBuffer
        from torchtitan.experiments.rl.tests.test_episode_buffer import (
            _episode,
            _FakeBatcher,
        )

        rl = _make_stub_rl_trainer()
        rl.batcher = _FakeBatcher(target=5, seq_len=5)  # one 5-token episode per batch
        rl.trainer_dp_degree = 1
        rl._train_version = 0
        buffer = EpisodeBuffer(
            batcher=rl.batcher,
            dp_degree=1,
            max_offpolicy_steps=5,
            max_buffered_batches=5,
            train_version=lambda: rl._train_version,
        )
        ready: asyncio.Queue = asyncio.Queue(maxsize=1)
        for _ in range(
            4
        ):  # enough for several batches, so the pack loop fills `ready` and blocks
            await buffer.put([_episode(version=0, completion=5)], [])

        pack_task = asyncio.create_task(rl._run_pack_loop(buffer, ready))
        await asyncio.sleep(
            0.05
        )  # let it pre-pack one batch into `ready` and block on the next put
        assert ready.full()  # a pre-packed batch is waiting, unconsumed

        pack_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(
                pack_task, timeout=2.0
            )  # must finish, not hang on the sentinel put

    asyncio.run(main())


def test_generation_metrics_flattens_turn_metrics_across_groups():
    group1 = SimpleNamespace(
        rollouts=[SimpleNamespace(turns=[SimpleNamespace(metrics=["a", "b"])])]
    )
    group2 = SimpleNamespace(
        rollouts=[SimpleNamespace(turns=[SimpleNamespace(metrics=["c"])])]
    )
    assert _generation_metrics([group1, group2]) == ["a", "b", "c"]
