# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Rollouter's controller-side worker dispatch."""

import asyncio
import functools
import gc
from types import SimpleNamespace

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch import (
    provisioning,
    rollout_worker as rollout_worker_module,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.system import MonarchActorSystem

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
            first_worker_index=4,
        )

        assert actor._worker is worker
        # Rank 2 in the proc mesh of a host whose first worker is the pool's worker 4.
        assert init_calls == [{"source": "rl_rollout_worker", "rank": 6}]

    asyncio.run(run())


class _CallOneEndpoint:
    """Like a Monarch endpoint, `call_one` returns a Future, not a coroutine."""

    def __init__(self) -> None:
        self.calls: list = []

    def call_one(self, *args, **kwargs) -> asyncio.Future:
        self.calls.append(kwargs or args)
        return asyncio.ensure_future(asyncio.sleep(0))


class _Worker:
    """A single-actor handle to one rollout worker."""

    def __init__(self) -> None:
        self.setup_async = _CallOneEndpoint()
        self.run_group = _RunGroupEndpoint()
        self.sync_log_step = _CallOneEndpoint()
        self.hostname = _CallOneEndpoint()


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


class _WorkerActorMesh:
    """What Monarch's `spawn` returns: `flatten("rank").slice(rank=r)` is worker r's handle."""

    def __init__(self, workers: list[_Worker]) -> None:
        self.workers = workers
        self.stopped = False

    def flatten(self, dim: str) -> "_WorkerActorMesh":
        return self

    def slice(self, *, rank: int) -> _Worker:
        return self.workers[rank]

    async def stop(self) -> None:
        self.stopped = True


class _WorkerMesh:
    def __init__(self, actor_mesh: _WorkerActorMesh) -> None:
        self.actor_mesh = actor_mesh
        self.spawn_args = None
        self.stopped = False

    def spawn(self, *args, **kwargs) -> _WorkerActorMesh:
        self.spawn_args = (args, kwargs)
        return self.actor_mesh

    async def stop(self) -> None:
        await self.actor_mesh.stop()
        self.stopped = True


class _ControllerHost:
    def __init__(self, worker_mesh: _WorkerMesh) -> None:
        self.worker_mesh = worker_mesh
        self.spawn_kwargs = None

    def spawn_procs(self, **kwargs) -> _WorkerMesh:
        self.spawn_kwargs = kwargs
        return self.worker_mesh


def _rollouter_without_datasets() -> Rollouter:
    rollouter = object.__new__(Rollouter)
    rollouter._config = SimpleNamespace(
        worker="worker_config",
        worker_pool_size=3,
        num_threads_per_worker=2,
        workers_on_controller_host=False,
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


def test_monarch_spawns_rollout_workers_on_controller_host(monkeypatch) -> None:
    async def run() -> None:
        workers = [_Worker() for _ in range(3)]
        worker_mesh = _WorkerMesh(_WorkerActorMesh(workers))
        host = _ControllerHost(worker_mesh)
        monkeypatch.setattr(provisioning, "this_host", lambda: host)
        structured_logger_init_fn = object()
        monkeypatch.setattr(
            sl,
            "get_structured_logger_subprocess_init_fn",
            lambda: structured_logger_init_fn,
        )
        actor_system = MonarchActorSystem(MonarchActorSystem.Config())

        handles = await actor_system.spawn_rollout_workers(
            "worker_config", num_workers=3, num_threads=2, on_controller_host=True
        )

        assert handles == workers
        assert host.spawn_kwargs["per_host"] == {"cpus": 3}
        args, kwargs = worker_mesh.spawn_args
        assert args[0] == "rollout_worker"
        assert args[1] is RolloutWorkerActor
        assert kwargs == {
            "worker_config": "worker_config",
            "num_threads": 2,
            "structured_logger_init_fn": structured_logger_init_fn,
            "first_worker_index": 0,
        }
        await actor_system.stop_rollout_workers()
        assert worker_mesh.actor_mesh.stopped
        assert worker_mesh.stopped

    asyncio.run(run())


def test_setup_spawns_and_sets_up_every_worker_and_close_stops_them() -> None:
    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        actor_system = await _setup(rollouter, num_workers=2)

        assert actor_system.spawn_kwargs == {
            "config": "worker_config",
            "num_workers": 3,
            "num_threads": 2,
            "on_controller_host": False,
        }
        for worker in actor_system.workers:
            assert worker.setup_async.calls == [
                {
                    "tokenizer_config": "tokenizer_config",
                    "renderer_config": "renderer_config",
                    "hf_assets_path": "hf_assets_path",
                    "vision_processor_config": None,
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
        group_ids_per_worker = [
            [call["group_id"] for call in worker.run_group.calls] for worker in workers
        ]
        all_group_ids = [group_id for ids in group_ids_per_worker for group_id in ids]
        assert sorted(all_group_ids) == list(range(200))
        # Random picks: a worker left idle by 200 groups has odds 4 * (3/4) ** 200.
        assert all(group_ids_per_worker)
        # Every call carries the group's arguments; the group ids are checked above.
        first_call = dict(workers[0].run_group.calls[0])
        first_call.pop("group_id")
        assert first_call == {
            "generate_fn": "generate_fn",
            "release_session_fn": "release_session_fn",
            "sample": "sample",
            "group_size": 2,
            "sampling": "sampling",
        }

        for worker in workers:
            worker.run_group.release.set()
        await asyncio.gather(*dispatches)
        await rollouter.close()

    asyncio.run(run())


async def _dispatch_then_cancel_caller(rollouter: Rollouter, worker: _Worker) -> None:
    """Start one `run_group` call, then cancel its caller as `Controller.run` does at the end."""
    dispatch = _dispatch(rollouter, group_id=1)
    await worker.run_group.started.wait()
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
        monkeypatch.setattr(rollouter_module, "_RUN_GROUP_DRAIN_TIMEOUT_S", 0.0)
        rollouter = _rollouter_without_datasets()
        actor_system = await _setup(rollouter)
        await _dispatch_then_cancel_caller(rollouter, actor_system.workers[0])

        await rollouter.close()

        assert actor_system.stopped
        assert not rollouter._inflight_run_group_calls

    asyncio.run(run())
