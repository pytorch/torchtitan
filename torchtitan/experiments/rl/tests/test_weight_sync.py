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
"""

import asyncio
import contextlib

from torchtitan.experiments.rl.components.weight_sync import WeightSyncManager

TRAINER_PUSH_KEY = "timing/weight_sync/trainer_push_model_state_dict"
GENERATOR_PULL_KEY = "timing/weight_sync/generator_pull_model_state_dict"


class _Endpoint:
    """Stands in for a Monarch endpoint, i.e. `trainer.push_model_state_dict.call()`."""

    def __init__(self, on_call):
        self._on_call = on_call

    async def call(self):
        await self._on_call()


class _FakeTrainer:
    def __init__(self, on_push):
        self.push_model_state_dict = _Endpoint(on_push)


class _FakeRouter:
    def __init__(self, on_pull):
        self._on_pull = on_pull
        self.pulled_versions: list[int] = []

    async def pull_model_state_dict(self, *, policy_version):
        self.pulled_versions.append(policy_version)
        await self._on_pull()


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


def _manager(*, trainer, router, buffer, groups_per_train_step=8):
    return WeightSyncManager(
        trainer=trainer,
        generator_router=router,
        group_buffer=buffer,
        groups_per_train_step=groups_per_train_step,
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
        push_metrics = await wsm.wait_prev_trainer_weight_push()
        pull_metrics = await wsm.wait_prev_generator_weight_pull()

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
        await wsm.wait_prev_trainer_weight_push()
        await wsm.wait_prev_generator_weight_pull()
        assert events == ["push", "pull", "release"]

    asyncio.run(run())


def test_buffer_release_uses_groups_per_train_step_and_trained_reason() -> None:
    async def run() -> None:
        buffer = _FakeBuffer()
        wsm = _manager(
            trainer=_FakeTrainer(_noop),
            router=_FakeRouter(_noop),
            buffer=buffer,
            groups_per_train_step=5,
        )
        wsm.start_async_push_pull(version=3)
        await wsm.wait_prev_generator_weight_pull()
        assert buffer.releases == [(5, "trained")]

    asyncio.run(run())


def test_pull_threads_the_started_version() -> None:
    async def run() -> None:
        router = _FakeRouter(_noop)
        wsm = _manager(trainer=_FakeTrainer(_noop), router=router, buffer=_FakeBuffer())
        wsm.start_async_push_pull(version=42)
        await wsm.wait_prev_generator_weight_pull()
        assert router.pulled_versions == [42]

    asyncio.run(run())


def test_wait_before_first_start_returns_zero_metrics() -> None:
    async def run() -> None:
        wsm = _manager(
            trainer=_FakeTrainer(_noop), router=_FakeRouter(_noop), buffer=_FakeBuffer()
        )
        push_metrics = await wsm.wait_prev_trainer_weight_push()
        pull_metrics = await wsm.wait_prev_generator_weight_pull()
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
            await wsm.wait_prev_trainer_weight_push()
        except RuntimeError:
            raised = True
        # The pull task also fails (it awaits the failed push); retrieve it so it is
        # not flagged as an unretrieved task exception.
        with contextlib.suppress(RuntimeError):
            await wsm._generator_pull_task
        assert raised

    asyncio.run(run())
