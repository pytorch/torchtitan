# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `Controller._trainer_loop`'s model state dict push and pull.

The push and pull of each step's policy overlap the next step's forward/backward.
These tests run the loop with fake actors (no GPU / Monarch / TorchStore) and check
the ordering that makes the overlap safe.
"""

import asyncio
import functools
import logging
from types import SimpleNamespace

import pytest

from torchtitan.rl.actors.system import TrainerHandle
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.observability import metrics as m

TRAINER_PUSH_KEY = "timing/weight_sync/trainer_push_model_state_dict"
GENERATOR_PULL_KEY = "timing/weight_sync/generator_pull_model_state_dict"


class _RankValues:
    """Stands in for a Monarch ValueMesh."""

    def __init__(self, value):
        self._value = value

    def get(self, rank):
        return self._value

    def values(self):
        return [self._value]


class _Endpoint:
    """Stands in for a Monarch endpoint; `on_call` runs on each call. Like Monarch's, `call` and
    `call_one` return a Future, not a coroutine: `TaskGroup.create_task` rejects one."""

    def __init__(self, on_call=None):
        self._on_call = on_call

    def call(self, *args, **kwargs) -> asyncio.Future:
        return asyncio.ensure_future(self._run(*args, **kwargs))

    def call_one(self, *args, **kwargs) -> asyncio.Future:
        return asyncio.ensure_future(self._run(*args, **kwargs))

    async def _run(self, *args, **kwargs):
        if self._on_call is not None:
            return await self._on_call(*args, **kwargs)


def _controller(
    events: list[str],
    *,
    on_forward_backward=None,
    on_push=None,
    on_pull=None,
    loss: float = 1.0,
    read_process_metrics: bool = False,
) -> Controller:
    """A controller with only the state `_trainer_loop` reads; each actor call appends to `events`.
    Without ``read_process_metrics``, each step's process-metrics read returns nothing at once, so
    ordering tests count event-loop turns, not the read's thread hop."""
    version = 0

    async def forward_backward(*args):
        events.append(f"fwd_bwd {version + 1}")
        if on_forward_backward is not None:
            await on_forward_backward()
        return _RankValues({"loss/mean": loss})

    async def optim_step(**kwargs):
        nonlocal version
        version += 1
        events.append(f"optim {version}")
        return _RankValues(SimpleNamespace(policy_version=version, metrics={}))

    async def push():
        events.append("push start")
        if on_push is not None:
            await on_push()
        events.append("push end")

    async def pull(policy_version):
        events.append(f"pull {policy_version}")
        if on_pull is not None:
            await on_pull()

    async def release_active_groups(count, *, reason):
        events.append(f"release {count} {reason}")

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            target_offpolicy_steps=1,
            max_offpolicy_steps=None,
            num_prompts_per_train_step=5,
        )
    )
    controller.start_step = 0
    controller._trainer_policy_version = 0
    trainer = SimpleNamespace(
        sync_log_step=_Endpoint(),
        forward_backward=_Endpoint(forward_backward),
        optim_step=_Endpoint(optim_step),
        push_model_state_dict=_Endpoint(push),
    )
    # DP degree 1: every trainer rank is at DP rank 0.
    controller.trainer = TrainerHandle(
        all_ranks=trainer, rank0=trainer, by_dp_rank=[trainer]
    )

    async def router_process_metrics():
        return [m.Metric("perf/router/cpu_seconds", m.NoReduce(1.0))]

    controller.generator_router = SimpleNamespace(
        sync_log_step=_Endpoint(),
        pull_model_state_dict=_Endpoint(pull),
        process_metrics=_Endpoint(router_process_metrics),
    )
    controller.batcher = SimpleNamespace(
        sync_log_step=SimpleNamespace(broadcast=lambda step: None),
        maybe_release_oldest_batch=SimpleNamespace(broadcast=lambda: None),
    )

    async def sync_log_step(step):
        pass

    async def worker_process_metrics():
        return [m.Metric("perf/rollout_worker/cpu_seconds", m.Sum(2.0))]

    controller._rollouter = SimpleNamespace(
        sync_log_step=sync_log_step,
        process_metrics=worker_process_metrics,
        commit_training_sample_ids=lambda group_ids: None,
        state_dict=lambda: {},
    )
    controller._process_stats = SimpleNamespace(
        metrics=lambda: [m.Metric("perf/controller/cpu_seconds", m.Sum(0.5))]
    )
    controller._group_buffer = SimpleNamespace(
        metrics=lambda: [], release_active_groups=release_active_groups
    )
    controller.logged_metrics = []
    if not read_process_metrics:
        controller._read_process_metrics = functools.partial(
            asyncio.sleep, 0, result=[]
        )
    controller.metrics_processor = SimpleNamespace(
        log=lambda *, step, is_validation, metrics: controller.logged_metrics.append(
            {metric.key: metric for metric in metrics}
        )
    )
    return controller


def _batch_queue(num_batches: int) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(num_batches):
        queue.put_nowait(
            SimpleNamespace(
                microbatch_groups=[],
                global_loss_token_counts=[1],
                global_routing_token_counts=None,
                group_ids=[],
                min_policy_versions=[0],
                metrics=[],
            )
        )
    return queue


def _run(controller: Controller, *, num_training_steps: int) -> None:
    asyncio.run(
        controller._trainer_loop(
            _batch_queue(num_training_steps), num_training_steps=num_training_steps
        )
    )


def test_push_then_pull_then_release_after_each_optimizer_step() -> None:
    events: list[str] = []
    _run(_controller(events), num_training_steps=2)

    assert events == [
        "fwd_bwd 1",
        "optim 1",
        # Step 1's push -> pull -> release starts in the background. The per-DP-rank
        # forward_backward calls run as tasks, so with these instant fakes it finishes first ...
        "push start",
        "push end",
        "pull 1",
        "release 5 trained",
        "fwd_bwd 2",
        # ... and lands before step 2's optimizer step.
        "optim 2",
        "push start",
        "push end",
        "pull 2",
        "release 5 trained",
    ]


def test_optimizer_step_waits_for_the_previous_push() -> None:
    async def push_until_fwd_bwd_2():
        # Ends 20 loop turns after forward/backward 2 starts, or after 1,000 turns if it never does.
        for _ in range(1000):
            if "fwd_bwd 2" in events:
                break
            await asyncio.sleep(0)
        for _ in range(20):
            await asyncio.sleep(0)

    events: list[str] = []
    _run(_controller(events, on_push=push_until_fwd_bwd_2), num_training_steps=2)

    # Forward/backward 2 overlaps push 1; optimizer step 2 waits for it to finish.
    assert events.index("fwd_bwd 2") < events.index("push end")
    assert events.index("push end") < events.index("optim 2")


def test_next_push_waits_for_the_previous_pull() -> None:
    async def slow_pull():
        for _ in range(5):
            await asyncio.sleep(0)

    events: list[str] = []
    _run(_controller(events, on_pull=slow_pull), num_training_steps=2)

    # Push 2 must not overwrite the TorchStore key while pull 1 reads it.
    second_push = [i for i, event in enumerate(events) if event == "push start"][1]
    assert events.index("release 5 trained") < second_push


def test_last_push_and_pull_finish_before_the_loop_returns() -> None:
    events: list[str] = []
    _run(_controller(events), num_training_steps=1)

    assert events[-3:] == ["push end", "pull 1", "release 5 trained"]


def test_push_and_pull_times_are_logged_one_step_late() -> None:
    controller = _controller([])
    _run(controller, num_training_steps=2)

    first, second = controller.logged_metrics
    # Step 1 has no previous push or pull.
    assert first[TRAINER_PUSH_KEY].value.value == 0.0
    assert first[GENERATOR_PULL_KEY].value.value == 0.0
    # Step 2 logs step 1's push and pull, which have run by then.
    assert second[TRAINER_PUSH_KEY].value.value > 0.0
    assert second[GENERATOR_PULL_KEY].value.value > 0.0


def test_each_step_logs_the_workers_router_and_controller_process_metrics() -> None:
    controller = _controller([], read_process_metrics=True)
    _run(controller, num_training_steps=2)

    assert len(controller.logged_metrics) == 2
    for logged in controller.logged_metrics:
        assert logged["perf/rollout_worker/cpu_seconds"].value.value == 2.0
        assert logged["perf/router/cpu_seconds"].value.value == 1.0
        assert logged["perf/controller/cpu_seconds"].value.value == 0.5
        assert "timing/step/blocking_read_process_metrics" in logged


def _run_and_check_the_read_ended(
    controller: Controller, reads: list[str], *, raises: str | None
) -> None:
    """Run the loop for 2 steps. When it returns (or raises ``raises``), the step's read has
    already ended, its cleanup included: nothing outlives the step."""

    async def run_loop():
        if raises is None:
            await controller._trainer_loop(_batch_queue(2), num_training_steps=2)
        else:
            with pytest.raises(RuntimeError, match=raises):
                await controller._trainer_loop(_batch_queue(2), num_training_steps=2)
        # Checked before any other turn of the event loop.
        assert reads[-1] == "cleaned up"

    asyncio.run(run_loop())


def _read_from_call(first_call: int, reads: list[str], *, fails: bool = False):
    """A router read that returns at once before call ``first_call``. From that call on, it
    raises at once (``fails``), or never ends and takes a few event-loop turns to clean up
    once cancelled."""

    async def read():
        reads.append("start")
        if len(reads) < first_call:
            return []
        if fails:
            raise RuntimeError("read failed")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            reads.append("cleaned up")
            raise

    return _Endpoint(read)


async def _failing_push():
    raise RuntimeError("push failed")


@pytest.mark.parametrize("early_exit", ["nan_loss", "failed_push"])
def test_an_early_exit_cancels_the_steps_process_metrics_read_and_waits_for_it(
    early_exit,
) -> None:
    """A NaN loss returns at step 1. A failed push raises at step 2: step 1's push fails in the
    background, and step 2 raises when it awaits it. Either way the read still running is
    cancelled, and its cleanup has finished when the loop returns or raises."""
    reads: list[str] = []
    if early_exit == "nan_loss":
        controller = _controller([], loss=float("nan"), read_process_metrics=True)
        first_never_ending_read, raises = 1, None
    else:
        controller = _controller([], on_push=_failing_push, read_process_metrics=True)
        first_never_ending_read, raises = 2, "push failed"
    controller.generator_router.process_metrics = _read_from_call(
        first_never_ending_read, reads
    )
    _run_and_check_the_read_ended(controller, reads, raises=raises)
    assert reads == ["start"] * first_never_ending_read + ["cleaned up"]
    assert len(controller.logged_metrics) == first_never_ending_read - 1


@pytest.mark.parametrize("early_exit", ["nan_loss", "failed_push"])
def test_a_read_that_failed_before_an_early_exit_is_logged_not_raised(
    early_exit, caplog
) -> None:
    """The step ends as it would without the read: a NaN loss returns, a failed push raises its
    own error. The read's error is logged."""
    # A slow forward/backward, so the read has failed before the step ends.
    slow = functools.partial(asyncio.sleep, 0.01)
    if early_exit == "nan_loss":
        controller = _controller(
            [], on_forward_backward=slow, loss=float("nan"), read_process_metrics=True
        )
        first_failing_read, raises = 1, None
    else:
        controller = _controller(
            [],
            on_forward_backward=slow,
            on_push=_failing_push,
            read_process_metrics=True,
        )
        first_failing_read, raises = 2, "push failed"
    controller.generator_router.process_metrics = _read_from_call(
        first_failing_read, [], fails=True
    )

    async def run_loop():
        await controller._trainer_loop(_batch_queue(2), num_training_steps=2)

    with caplog.at_level(logging.WARNING):
        if raises is None:
            asyncio.run(run_loop())
        else:
            with pytest.raises(RuntimeError, match=raises):
                asyncio.run(run_loop())
    assert "Reading process metrics failed" in caplog.text
    assert "RuntimeError('read failed')" in caplog.text


def test_push_failure_fails_the_loop() -> None:
    controller = _controller([], on_push=_failing_push)
    with pytest.raises(RuntimeError, match="push failed"):
        _run(controller, num_training_steps=2)
