# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for WeightSyncManager.

It overlaps the trainer->generator weight handoff with the next training step:
`start_async_push_pull` fires push -> pull -> buffer-slot release in the background,
and the loop joins each leg with `wait_prev_*`. These tests use fakes for the
trainer actor, generator router, and group buffer (no GPU / Monarch / TorchStore).
The last test runs `Controller._trainer_loop` on these fakes to check when the loop awaits the push.
"""

import asyncio
import contextlib
import time
from types import SimpleNamespace

import torch

from torchtitan.rl.controller import Controller
from torchtitan.rl.distributed.weight_sync import WeightSyncManager
from torchtitan.rl.types import OptimizerStepOutput, TrainerStepBatch

TRAINER_PUSH_KEY = "timing/weight_sync/push_wall"
GENERATOR_PULL_KEY = "timing/weight_sync/pull_wall"


class _Endpoint:
    """Stands in for a Monarch endpoint: `await endpoint.call(...)` or `endpoint.call_one(...)`."""

    def __init__(self, on_call):
        self.call = self.call_one = on_call


class _FakeTrainer:
    def __init__(self, on_push):
        self.push_model_state_dict = _Endpoint(on_push)


class _FakeRouter:
    def __init__(self, on_pull):
        self._on_pull = on_pull
        self.pulled_versions: list[int] = []
        self.pull_model_state_dict = _PullEndpoint(self)


class _PullEndpoint:
    def __init__(self, router):
        self._router = router

    async def call_one(self, policy_version):
        self._router.pulled_versions.append(policy_version)
        await self._router._on_pull()


class _FakeBuffer:
    def __init__(self, events=None):
        self.releases: list[tuple[int, str]] = []
        self._events = events

    async def release_active_groups(self, count, *, reason):
        if self._events is not None:
            self._events.append("release")
        self.releases.append((count, reason))


async def _noop():
    return None


def _manager(*, trainer, router, buffer, num_prompts_per_train_step=8):
    return WeightSyncManager(
        trainer=trainer,
        generator_router=router,
        group_buffer=buffer,
        num_prompts_per_train_step=num_prompts_per_train_step,
    )


def test_push_then_pull_then_buffer_release_in_order() -> None:
    async def run() -> None:
        events: list[str] = []

        async def on_push():
            events.append("push")

        async def on_pull():
            events.append("pull")

        wsm = _manager(
            trainer=_FakeTrainer(on_push),
            router=_FakeRouter(on_pull),
            buffer=_FakeBuffer(events),
        )

        wsm.start_async_push_pull(version=7)
        push_metrics = await wsm.wait_prev_push()
        pull_metrics = await wsm.wait_prev_pull()

        # The pull reads what the push wrote, and the buffer-slot release rides on the pull.
        assert events == ["push", "pull", "release"]
        assert [metric.key for metric in push_metrics] == [TRAINER_PUSH_KEY]
        assert [metric.key for metric in pull_metrics] == [GENERATOR_PULL_KEY]

    asyncio.run(run())


def test_start_async_push_pull_returns_before_work_runs() -> None:
    async def run() -> None:
        gate = asyncio.Event()
        events: list[str] = []

        async def on_push():
            await gate.wait()
            events.append("push")

        async def on_pull():
            events.append("pull")

        wsm = _manager(
            trainer=_FakeTrainer(on_push),
            router=_FakeRouter(on_pull),
            buffer=_FakeBuffer(events),
        )

        wsm.start_async_push_pull(version=1)
        await asyncio.sleep(0)  # give the background tasks a turn
        assert events == []  # push is gated -> nothing ran; start did not block

        gate.set()
        await wsm.wait_prev_push()
        await wsm.wait_prev_pull()
        assert events == ["push", "pull", "release"]

    asyncio.run(run())


def test_buffer_release_uses_num_prompts_per_train_step_and_trained_reason() -> None:
    async def run() -> None:
        buffer = _FakeBuffer()
        wsm = _manager(
            trainer=_FakeTrainer(_noop),
            router=_FakeRouter(_noop),
            buffer=buffer,
            num_prompts_per_train_step=5,
        )
        wsm.start_async_push_pull(version=3)
        await wsm.wait_prev_pull()
        assert buffer.releases == [(5, "trained")]

    asyncio.run(run())


def test_pull_threads_the_started_version() -> None:
    async def run() -> None:
        router = _FakeRouter(_noop)
        wsm = _manager(trainer=_FakeTrainer(_noop), router=router, buffer=_FakeBuffer())
        wsm.start_async_push_pull(version=42)
        await wsm.wait_prev_pull()
        assert router.pulled_versions == [42]

    asyncio.run(run())


def test_wait_before_first_start_returns_zero_metrics() -> None:
    async def run() -> None:
        wsm = _manager(
            trainer=_FakeTrainer(_noop), router=_FakeRouter(_noop), buffer=_FakeBuffer()
        )
        push_metrics = await wsm.wait_prev_push()
        pull_metrics = await wsm.wait_prev_pull()
        assert push_metrics[0].key == TRAINER_PUSH_KEY
        assert push_metrics[0].value.value == 0.0
        assert pull_metrics[0].value.value == 0.0

    asyncio.run(run())


def test_pull_waits_for_its_own_push_not_a_later_one() -> None:
    # White-box: the pull must await the push task captured when it was started, not
    # whatever the shared push-task handle points at after a later start_async_push_pull.
    async def run() -> None:
        gate_first_push = asyncio.Event()
        push_calls = [0]

        async def on_push():
            push_calls[0] += 1
            if push_calls[0] == 1:
                await gate_first_push.wait()  # gate ONLY the first push

        wsm = _manager(
            trainer=_FakeTrainer(on_push),
            router=_FakeRouter(_noop),
            buffer=_FakeBuffer(),
        )

        wsm.start_async_push_pull(version=1)
        pull1 = wsm._generator_pull_task  # cycle-1 pull
        wsm.start_async_push_pull(version=2)  # reassigns the shared push-task handle

        for _ in range(5):
            await asyncio.sleep(0)
        # The cycle-1 pull is still blocked on cycle-1's (gated) push, even though a
        # newer ungated push exists -> it is bound to its own push, not the handle.
        assert not pull1.done()

        gate_first_push.set()
        await pull1
        assert pull1.done()

    asyncio.run(run())


def test_push_exception_propagates_through_wait() -> None:
    async def run() -> None:
        async def boom():
            raise RuntimeError("push failed")

        wsm = _manager(
            trainer=_FakeTrainer(boom), router=_FakeRouter(_noop), buffer=_FakeBuffer()
        )
        wsm.start_async_push_pull(version=1)

        raised = False
        try:
            await wsm.wait_prev_push()
        except RuntimeError:
            raised = True
        # The pull task also fails (it awaits the failed push); retrieve it so it is
        # not flagged as an unretrieved task exception.
        with contextlib.suppress(RuntimeError):
            await wsm._generator_pull_task
        assert raised

    asyncio.run(run())


FORWARD_BACKWARD_S = 0.3
PUSH_S = 0.05


async def _rpc(*args, **kwargs):
    await asyncio.sleep(0)  # an endpoint call always yields to the event loop


class _FakeMetricsProcessor:
    def __init__(self):
        self.values_by_step: dict[int, dict[str, float]] = {}

    def log(self, *, step, is_validation, metrics):
        self.values_by_step[step] = {
            metric.key: metric.value.value for metric in metrics
        }


async def _run_trainer_loop(*, num_training_steps):
    """Run `Controller._trainer_loop` on fakes; return each step's logged values and the event order.

    One asyncio loop stands in for the trainer actor's loop. Forward/backward blocks it with time.sleep,
    as `Trainer.forward_backward` blocks the actor's loop, so a push awaiting its RPC cannot resume.
    """

    events: list[str] = []

    async def push_model_state_dict():
        events.append("push_start")
        await asyncio.sleep(PUSH_S)  # torchstore RPCs
        events.append("push_end")

    async def pull_model_state_dict():
        await asyncio.sleep(0.01)  # the generators' pull

    async def forward_backward(*args):
        events.append("forward_backward_start")
        time.sleep(FORWARD_BACKWARD_S)
        events.append("forward_backward_end")
        return {"loss/mean": 1.0}

    async def optim_step(*, controller_state, last_step):
        return OptimizerStepOutput(
            policy_version=controller._trainer_policy_version + 1, metrics={}
        )

    trainer = SimpleNamespace(
        sync_log_step=_Endpoint(_rpc),
        forward_backward=_Endpoint(forward_backward),
        optim_step=_Endpoint(optim_step),
        push_model_state_dict=_Endpoint(push_model_state_dict),
    )
    controller = SimpleNamespace(
        start_step=0,
        trainer=trainer,
        generator_router=SimpleNamespace(sync_log_step=_Endpoint(_rpc)),
        _rollouter=SimpleNamespace(
            sync_log_step=_rpc,
            acknowledge_training_sample_ids=lambda sample_ids: None,
            state_dict=dict,
        ),
        _trainer_policy_version=0,
        config=SimpleNamespace(
            async_loop=SimpleNamespace(
                target_offpolicy_steps=1, max_offpolicy_steps=None
            )
        ),
        _get_rank_0_value=lambda result: result,
        _weight_sync=_manager(
            trainer=trainer,
            router=_FakeRouter(pull_model_state_dict),
            buffer=_FakeBuffer(events),  # records "release"
        ),
        _group_buffer=SimpleNamespace(metrics=lambda: []),
        metrics_processor=_FakeMetricsProcessor(),
    )
    training_batch_queue = asyncio.Queue()
    for _ in range(num_training_steps):
        training_batch_queue.put_nowait(
            TrainerStepBatch(
                microbatches=[],
                global_loss_token_counts=torch.tensor([10]),
                global_routing_token_counts=torch.tensor([10]),
                metrics=[],
                group_ids=[0],
                min_policy_versions=[0],
            )
        )
    await Controller._trainer_loop(
        controller, training_batch_queue, num_training_steps=num_training_steps
    )
    return controller.metrics_processor.values_by_step, events


def test_trainer_loop_finishes_push_before_forward_backward() -> None:
    values_by_step, events = asyncio.run(_run_trainer_loop(num_training_steps=2))
    # A push still waiting on its RPC when forward/backward starts would resume only after it ends.
    # The pull and slot release are not awaited before forward/backward.
    assert events == [
        *["forward_backward_start", "forward_backward_end"],  # step 1, no push yet
        *["push_start", "push_end"],  # step 1's push
        *["forward_backward_start", "forward_backward_end"],  # step 2
        "release",  # step 1's pull and slot release, not awaited before step 2's forward/backward
        *["push_start", "push_end", "release"],  # step 2's sync, awaited after the loop
    ]
    step_2 = values_by_step[2]
    # The push takes its own time, and the loop waits for it before forward/backward.
    assert step_2[TRAINER_PUSH_KEY] < FORWARD_BACKWARD_S / 2
    assert PUSH_S / 2 < step_2["timing/step/wait_for_push"] < FORWARD_BACKWARD_S / 2
