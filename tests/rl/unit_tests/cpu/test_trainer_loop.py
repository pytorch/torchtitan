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
import time
from types import SimpleNamespace

import pytest

from torchtitan.rl.actors.system import TrainerHandle
from torchtitan.rl.controller import controller as controller_module
from torchtitan.rl.controller.controller import Controller, ValidationConfig

TRAINER_PUSH_KEY = "timing/weight_sync/trainer_push_model_state_dict"
GENERATOR_PULL_KEY = "timing/weight_sync/generator_pull_model_state_dict"


@pytest.fixture(autouse=True)
def _no_host_memory_scan(monkeypatch):
    # Awaiting the real scan yields to the event loop, so the fake push finishes before the next forward/backward.
    monkeypatch.setattr(controller_module, "compute_host_memory_metrics", lambda: [])


class _RankValues:
    """Stands in for a Monarch ValueMesh."""

    def __init__(self, value):
        self._value = value

    def get(self, rank):
        return self._value

    def values(self):
        return [self._value]


class _Endpoint:
    """Stands in for a Monarch endpoint; `on_call` runs on each call."""

    def __init__(self, on_call=None):
        self._on_call = on_call

    async def call(self, *args, **kwargs):
        if self._on_call is not None:
            return await self._on_call(*args, **kwargs)

    async def call_one(self, *args, **kwargs):
        return await self.call(*args, **kwargs)


def _controller(events: list[str], *, on_push=None, on_pull=None) -> Controller:
    """A controller with only the state `_trainer_loop` reads; each actor call appends to `events`."""
    version = 0

    async def forward_backward(*args):
        events.append(f"fwd_bwd {version + 1}")
        return _RankValues({"loss/mean": 1.0})

    async def optim_step(**kwargs):
        nonlocal version
        version += 1
        events.append(f"optim {version}")
        return _RankValues(SimpleNamespace(policy_version=version, metrics={}))

    async def save_checkpoint(*, last_step):
        events.append(f"save {version} last_step={last_step}")

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
            validation=ValidationConfig(),
            extra_validations=[],
        )
    )
    controller._extra_validation_tasks = {}
    controller._extra_validation_requested = set()
    controller.start_step = 0
    controller._trainer_policy_version = 0
    trainer = SimpleNamespace(
        sync_log_step=_Endpoint(),
        forward_backward=_Endpoint(forward_backward),
        optim_step=_Endpoint(optim_step),
        save_checkpoint=_Endpoint(save_checkpoint),
        push_model_state_dict=_Endpoint(push),
    )
    # DP degree 1: every trainer rank is at DP rank 0.
    controller.trainer = TrainerHandle(
        all_ranks=trainer, rank0=trainer, by_dp_rank=[trainer]
    )
    controller.generator_router = SimpleNamespace(
        sync_log_step=_Endpoint(), pull_model_state_dict=_Endpoint(pull)
    )
    controller.batcher = SimpleNamespace(
        sync_log_step=SimpleNamespace(broadcast=lambda step: None),
        maybe_release_oldest_batch=SimpleNamespace(broadcast=lambda: None),
    )

    async def sync_log_step(step):
        pass

    controller._rollouter = SimpleNamespace(
        sync_log_step=sync_log_step,
        acknowledge_training_sample_ids=lambda group_ids: None,
        state_dict=lambda: {},
        training_data_metrics=list,
    )

    async def record_step_start(*, trainer_policy_version):
        pass

    controller._group_buffer = SimpleNamespace(
        metrics=lambda: [],
        release_active_groups=release_active_groups,
        record_step_start=record_step_start,
        pop_dropped_group_ids=list,
    )
    controller.logged_metrics = []
    controller.metrics_processor = SimpleNamespace(
        log=lambda *, step, is_validation, metrics: controller.logged_metrics.append(
            {metric.key: metric for metric in metrics}
        ),
        commit=lambda: None,
    )
    return controller


def _batch_queue(num_batches: int) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(num_batches):
        queue.put_nowait(
            SimpleNamespace(
                microbatch_groups=[],
                global_loss_token_counts=[1],
                global_routing_token_counts=[1],
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
        "save 1 last_step=False",
        # Step 1's push -> pull -> release starts in the background. The per-DP-rank
        # forward_backward calls run as tasks, so with these instant fakes it finishes first ...
        "push start",
        "push end",
        "pull 1",
        "release 5 trained",
        "fwd_bwd 2",
        # ... and lands before step 2's optimizer step.
        "optim 2",
        "save 2 last_step=True",
        "push start",
        "push end",
        "pull 2",
        "release 5 trained",
    ]


def test_optimizer_step_waits_for_the_previous_push() -> None:
    events: list[str] = []

    async def slow_push():
        # Event-loop ticks alone can finish before forward/backward 2: the loop also awaits
        # worker threads (asyncio.to_thread). Hold push 1 until forward/backward 2 starts.
        for _ in range(1000):
            if "fwd_bwd 2" in events:
                return
            await asyncio.sleep(0.001)

    _run(_controller(events, on_push=slow_push), num_training_steps=2)

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


def test_step_total_covers_the_save_and_the_previous_metrics_log(monkeypatch) -> None:
    now = 0.0
    monkeypatch.setattr(time, "perf_counter", lambda: now)

    def takes(seconds):
        async def on_call(*args, **kwargs):
            nonlocal now
            now += seconds

        return on_call

    controller = _controller([])
    controller.trainer.all_ranks.sync_log_step = _Endpoint(takes(1.0))
    controller.trainer.all_ranks.save_checkpoint = _Endpoint(takes(30.0))
    log = controller.metrics_processor.log

    def slow_log(**kwargs):
        nonlocal now
        now += 2.0
        log(**kwargs)

    controller.metrics_processor.log = slow_log
    _run(controller, num_training_steps=2)

    first, second = (
        {key: metric.value.value for key, metric in logged.items()}
        for logged in controller.logged_metrics
    )
    # The save is its own phase, not part of the optimizer's.
    assert first["timing/step/save_checkpoint"] == 30.0
    assert first["timing/step/optimizer"] == 0.0
    # Step 1: sync_log_step + save. No earlier metrics log, so no `unaccounted`.
    assert first["timing/step/total"] == 31.0
    assert "perf/trainer/step_time_ratio/unaccounted" not in first
    # Step 2 starts at step 1's metrics log: 2 + 1 + 30, all inside a phase.
    assert second["timing/step/metrics_log"] == 2.0
    assert second["timing/step/total"] == 33.0
    assert second["perf/trainer/step_time_ratio/unaccounted"] == pytest.approx(0.0)


def test_push_failure_fails_the_loop() -> None:
    async def failing_push():
        raise RuntimeError("push failed")

    controller = _controller([], on_push=failing_push)
    with pytest.raises(RuntimeError, match="push failed"):
        _run(controller, num_training_steps=2)
