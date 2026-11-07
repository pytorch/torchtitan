# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Rollouter's controller-side worker dispatch."""

import asyncio
import functools
import gc
import pathlib
import sys
from types import SimpleNamespace

import cloudpickle
import pytest
from monarch.actor import ActorError, this_host

from torchtitan.rl.actors.monarch import rollout_worker as rollout_worker_module
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor

from torchtitan.rl.rollout import rollouter as rollouter_module, RolloutGroup
from torchtitan.rl.rollout.rollouter import Rollouter


class _RunGroupEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.error: Exception | None = None

    def call_one(self, **kwargs) -> asyncio.Future[RolloutGroup]:
        self.calls.append(kwargs)
        self.started.set()
        return asyncio.create_task(self._execute(kwargs))

    async def _execute(self, kwargs) -> RolloutGroup:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.error is not None:
            raise self.error
        return RolloutGroup(group_id=kwargs["group_id"], rollouts=[])


def test_rollout_worker_actor_builds_its_worker_and_inits_its_logger(
    monkeypatch,
) -> None:
    async def run() -> None:
        worker = object()
        worker_config = SimpleNamespace(build=lambda: worker)
        init_calls = []
        monkeypatch.setattr(
            rollout_worker_module, "current_rank", lambda: SimpleNamespace(rank=2)
        )
        actor = object.__new__(RolloutWorkerActor)

        RolloutWorkerActor.__init__(
            actor,
            worker_config=worker_config,
            num_threads=1,
            # Bound like the controller's; the worker overrides source and rank.
            structured_logger_init_fn=functools.partial(
                lambda **kwargs: init_calls.append(kwargs), source="rl", rank=0
            ),
        )

        assert actor._worker is worker
        assert init_calls == [{"source": "rl_rollout_worker", "rank": 2}]

    asyncio.run(run())


class _CallOneEndpoint:
    """Like a Monarch endpoint, `call_one` returns a Future, not a coroutine."""

    def __init__(self) -> None:
        self.calls: list = []

    def call_one(self, *args, **kwargs) -> asyncio.Future:
        self.calls.append(kwargs or args)
        return asyncio.ensure_future(asyncio.sleep(0))


class _CancelGroupEndpoint:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.called, self.release = asyncio.Event(), asyncio.Event()

    def call_one(self, group_id: int) -> asyncio.Future:
        self.calls.append(group_id)
        self.called.set()
        return asyncio.ensure_future(self.release.wait())


class _Worker:
    """A single-actor handle to one rollout worker."""

    def __init__(self) -> None:
        self.setup_async = _CallOneEndpoint()
        self.run_group = _RunGroupEndpoint()
        self.cancel_group = _CancelGroupEndpoint()
        self.wait_for_generate_calls = _CallOneEndpoint()
        self.sync_log_step = _CallOneEndpoint()


class _ActorSystem:
    def __init__(self, workers: list[_Worker]) -> None:
        self.workers = workers
        self.spawn_kwargs = None
        self.stopped = False

    async def spawn_rollout_workers(self, config, **kwargs) -> list[_Worker]:
        self.spawn_kwargs = {"config": config, **kwargs}
        return self.workers

    async def stop_rollout_workers(self) -> None:
        self.stopped = True


def _rollouter_without_datasets() -> Rollouter:
    rollouter = object.__new__(Rollouter)
    rollouter._config = SimpleNamespace(
        worker="worker_config",
        worker_pool_size=3,
        num_threads_per_worker=2,
    )
    rollouter._workers = []
    rollouter._actor_system = None
    rollouter._inflight_run_group_calls = set()
    return rollouter


async def _setup(rollouter: Rollouter, num_workers: int = 1) -> _ActorSystem:
    actor_system = _ActorSystem([_Worker() for _ in range(num_workers)])
    await rollouter.setup_async(
        actor_system=actor_system,
        tokenizer_config="tokenizer_config",
        renderer_config="renderer_config",
        hf_assets_path="hf_assets_path",
    )
    return actor_system


def _dispatch(rollouter: Rollouter, group_id: int) -> asyncio.Task:
    return asyncio.create_task(
        rollouter.run_group_rollouts(
            generate_fn="generate_fn",
            release_session_fn="release_session_fn",
            sample="sample",
            group_id=group_id,
            group_size=2,
            sampling="sampling",
        )
    )


def test_setup_spawns_and_sets_up_every_worker_and_close_stops_them() -> None:
    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        actor_system = await _setup(rollouter, num_workers=2)

        assert actor_system.spawn_kwargs == {
            "config": "worker_config",
            "num_workers": 3,
            "num_threads": 2,
        }
        for worker in actor_system.workers:
            assert worker.setup_async.calls == [
                {
                    "tokenizer_config": "tokenizer_config",
                    "renderer_config": "renderer_config",
                    "hf_assets_path": "hf_assets_path",
                }
            ]
        await rollouter.close()
        assert actor_system.stopped

    asyncio.run(run())


def test_dispatch_sends_each_group_to_one_worker_and_uses_every_worker() -> None:
    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        workers = (await _setup(rollouter, num_workers=4)).workers

        dispatches = [_dispatch(rollouter, group_id) for group_id in range(200)]
        await asyncio.sleep(0)
        group_ids = [[c["group_id"] for c in w.run_group.calls] for w in workers]
        assert sorted(sum(group_ids, [])) == list(range(200))
        # Random picks: a worker left idle by 200 groups has odds 4 * (3/4) ** 200.
        assert all(group_ids)

        for worker in workers:
            worker.run_group.release.set()
        await asyncio.gather(*dispatches)
        await rollouter.close()

    asyncio.run(run())


async def _dispatch_then_cancel_caller(rollouter: Rollouter, worker: _Worker) -> None:
    """Start one `run_group` call, then cancel its caller as `Controller.run` does at the end."""
    dispatch = _dispatch(rollouter, group_id=1)
    await worker.run_group.started.wait()
    # Let the worker's cancel_group return at once.
    worker.cancel_group.release.set()
    dispatch.cancel()
    await asyncio.gather(dispatch, return_exceptions=True)


def test_close_waits_for_run_group_of_cancelled_caller() -> None:
    async def run() -> None:
        loop_errors: list[str] = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: loop_errors.append(context["message"])
        )
        rollouter = _rollouter_without_datasets()
        actor_system = await _setup(rollouter)
        worker = actor_system.workers[0]
        await _dispatch_then_cancel_caller(rollouter, worker)

        close = asyncio.create_task(rollouter.close())
        await asyncio.sleep(0)
        # The worker still runs the group, so the pool stays up.
        assert not actor_system.stopped

        # Each rollout catches its own errors, but scoring can still make run_group raise.
        worker.run_group.error = RuntimeError("scoring failed")
        worker.run_group.release.set()
        await close
        assert actor_system.stopped
        assert not worker.run_group.cancelled.is_set()
        # The cancelled caller never reads the error; the done callback must, or asyncio logs it here.
        gc.collect()
        assert loop_errors == []

    asyncio.run(run())


def test_close_stops_waiting_after_drain_timeout(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setattr(rollouter_module, "_SHUTDOWN_DRAIN_TIMEOUT_S", 0.0)
        rollouter = _rollouter_without_datasets()
        actor_system = await _setup(rollouter)
        await _dispatch_then_cancel_caller(rollouter, actor_system.workers[0])

        await rollouter.close()

        assert actor_system.stopped
        assert not rollouter._inflight_run_group_calls

    asyncio.run(run())


def test_cancelled_dispatch_cancels_the_group_on_its_worker() -> None:
    """A cancelled caller stops the group on the worker that runs it, and only there, before the
    cancel propagates."""

    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        workers = (await _setup(rollouter, num_workers=2)).workers
        dispatch = _dispatch(rollouter, group_id=5)
        await asyncio.sleep(0)
        (running,) = [w for w in workers if w.run_group.calls]
        (other,) = [w for w in workers if w is not running]
        dispatch.cancel()
        await running.cancel_group.called.wait()
        # The cancel waits until the worker stopped the group.
        assert not dispatch.done()
        running.cancel_group.release.set()
        with pytest.raises(asyncio.CancelledError):
            await dispatch
        assert running.cancel_group.calls == [5]
        assert other.cancel_group.calls == []
        # The shielded run_group call still runs; finish it so close() doesn't wait for it.
        running.run_group.release.set()
        await rollouter.close()

    asyncio.run(run())


def test_cancelled_dispatch_stops_waiting_for_a_cancel_group_that_hangs(
    monkeypatch, caplog
) -> None:
    async def run() -> None:
        monkeypatch.setattr(rollouter_module, "_SHUTDOWN_DRAIN_TIMEOUT_S", 0.05)
        rollouter = _rollouter_without_datasets()
        worker = (await _setup(rollouter)).workers[0]  # its cancel_group never returns

        dispatch = _dispatch(rollouter, group_id=5)
        await worker.run_group.started.wait()
        dispatch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(dispatch, timeout=5)
        assert "cancel_group(5) failed" in caplog.text
        worker.run_group.release.set()
        await rollouter.close()

    asyncio.run(run())


class _GeneratingWorker:
    """A rollout worker whose group waits on one slow generate call. On a cancel it cleans up
    slowly (e.g. env.close()), then marks the group cancelled in a file."""

    def __init__(self, marker_dir: str) -> None:
        self.marker_dir = marker_dir

    async def run_group(self, *, generate_fn, group_id: int, **kwargs) -> RolloutGroup:
        try:
            await generate_fn([1], request_id="r", group_id=group_id)
        except asyncio.CancelledError:
            await asyncio.sleep(0.5)
            pathlib.Path(self.marker_dir, f"{group_id}.cancelled").touch()
            raise
        return RolloutGroup(group_id=group_id, rollouts=[])

    async def close(self) -> None:
        pathlib.Path(self.marker_dir, "closed").touch()


class _GeneratingWorkerConfig:
    def __init__(self, marker_dir: str) -> None:
        self.marker_dir = marker_dir

    def build(self) -> _GeneratingWorker:
        return _GeneratingWorker(self.marker_dir)


async def _slow_generate(marker_dir: str, prompt_token_ids, **kwargs) -> None:
    # A router call still running when its rollout is cancelled; the marker is its reply.
    await asyncio.sleep(1.5)
    pathlib.Path(marker_dir, "replied").touch()


def test_cancel_group_and_wait_for_generate_calls_on_real_worker_processes(
    tmp_path,
) -> None:
    """Sent right after `run_group`, `cancel_group` finds the group (Monarch starts one sender's
    concurrent calls in order) and returns once it stopped, while its generate call still runs.
    `wait_for_generate_calls` returns only after that call's reply, so `Rollouter.close` never stops
    a worker the router still owes. Stopping the process closes the worker."""

    async def run() -> None:
        mesh = this_host().spawn_procs(per_host={"cpus": 1})
        # The worker processes can't import this test module, so ship the fakes by value.
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
        try:
            workers = mesh.spawn(
                "rollout_worker",
                RolloutWorkerActor,
                worker_config=_GeneratingWorkerConfig(str(tmp_path)),
                num_threads=1,
                structured_logger_init_fn=None,
            )
            group = workers.run_group.choose(
                generate_fn=functools.partial(_slow_generate, str(tmp_path)),
                release_session_fn=None,
                sample=None,
                group_id=7,
                group_size=1,
                sampling=None,
            )
            await workers.cancel_group.call(7)
            assert (tmp_path / "7.cancelled").exists()
            with pytest.raises(ActorError):
                await group
            assert not (tmp_path / "replied").exists()
            await workers.wait_for_generate_calls.call()
            assert (tmp_path / "replied").exists()
            # The worker keeps serving after a cancel.
            await workers.sync_log_step.call(1)
        finally:
            cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
            await mesh.stop()
        assert (tmp_path / "closed").exists()

    asyncio.run(run())
