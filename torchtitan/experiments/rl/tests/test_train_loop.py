# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer-loop tests: weight-sync ordering, the divergence gate, the producer-crash supervisor,
and the pure metric/flatten helpers. Lifecycle is driven through stub actors; no GPUs."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.episode_buffer import PackedEpisodeBatch
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.tests.test_shutdown import _make_stub_rl_trainer
from torchtitan.experiments.rl.trainer import (
    _build_train_step_metrics,
    _generation_metrics,
    _TrainStepTimings,
    _WeightSyncTimings,
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


class _OneBatchThenNone:
    """Buffer stub: serves one `PackedEpisodeBatch`, then `None` (closed + drained)."""

    def __init__(self, batch):
        self._batch = batch
        self._served = False

    async def get_batch(self, *, train_version):
        if self._served:
            return None
        self._served = True
        return self._batch


_BATCH = PackedEpisodeBatch(
    microbatches=[["mb"]], num_global_valid_tokens=10, metrics=[]
)
_WEIGHT_SYNC_ORDER = ["forward_backward", "optim_step", "push", "pull"]


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


def test_consume_and_train_syncs_weights_before_advancing_version():
    # Must-preserve: fwd/bwd -> optim -> push -> pull, THEN advance the version to what optim
    # published. Pins the order so a refactor can't pull before push or advance early.
    async def main():
        calls: list[str] = []
        rl = _stub_trainer_for_step(calls, loss=0.5)
        await rl._consume_and_train(_OneBatchThenNone(_BATCH), num_steps=5)
        assert [c for c in calls if c in _WEIGHT_SYNC_ORDER] == _WEIGHT_SYNC_ORDER
        assert (
            rl._train_version == 7
        )  # advanced to optim_output.policy_version, after the pull

    asyncio.run(main())


def test_consume_and_train_does_not_publish_weights_on_nan_loss():
    # Must-preserve: a NaN/Inf loss breaks BEFORE optim/push/pull, so a bad step never publishes
    # weights and never advances the version.
    async def main():
        calls: list[str] = []
        rl = _stub_trainer_for_step(calls, loss=math.nan)
        await rl._consume_and_train(_OneBatchThenNone(_BATCH), num_steps=5)
        assert "forward_backward" in calls
        assert "optim_step" not in calls
        assert "push" not in calls and "pull" not in calls
        assert rl._train_version == 0  # never advanced

    asyncio.run(main())


def test_run_rollout_producer_closes_buffer_on_crash_and_reraises():
    # Must-preserve: the producer supervisor closes the buffer on the way out (so a parked consumer
    # unblocks) and re-raises the real producer exception.
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


def test_build_train_step_metrics_derives_ratios_and_active_throughput():
    timings = _TrainStepTimings(
        step_s=10.0,
        get_batch_s=8.0,
        train_s=1.0,
        weight_sync=_WeightSyncTimings(push_s=0.4, total_s=1.0),
    )
    metrics = _build_train_step_metrics(
        buffer_metrics=[m.Metric("buffer/depth_batches", m.Max(2.0))],
        fwd_bwd_metrics={"loss/mean": 0.5},
        optimizer_metrics={"train/lr": 1e-4},
        num_global_valid_tokens=100,
        timings=timings,
    )
    by_key = {
        metric.key: metric.value.value
        for metric in metrics
        if isinstance(metric.value, m.NoReduce)
    }
    assert by_key["controller/trainer_idle_time_ratio"] == pytest.approx(0.8)  # 8 / 10
    assert by_key["timing/weight_sync_overhead_ratio"] == pytest.approx(0.1)  # 1 / 10
    assert by_key["timing/weight_sync/pull"] == pytest.approx(0.6)  # total - push
    assert by_key["perf/tokens_per_second"] == pytest.approx(
        10.0
    )  # goodput: tokens / step
    # active throughput divides by TRAIN time, not the whole step (which would just be goodput):
    assert by_key["perf/trainer_active_tokens_per_second"] == pytest.approx(100.0)


def test_setup_async_rejects_rollouts_that_can_exceed_batch_seq_len():
    # The seq-len fail-fast guard: if a rollout can produce an episode longer than the batcher's
    # seq_len, packing would silently drop it and crash the trainer mid-run, so setup_async raises
    # before spawning anything. (Pins the guard kept in setup_async.)
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


def test_generation_metrics_flattens_turn_metrics_across_groups():
    group1 = SimpleNamespace(
        rollouts=[SimpleNamespace(turns=[SimpleNamespace(metrics=["a", "b"])])]
    )
    group2 = SimpleNamespace(
        rollouts=[SimpleNamespace(turns=[SimpleNamespace(metrics=["c"])])]
    )
    assert _generation_metrics([group1, group2]) == ["a", "b", "c"]
