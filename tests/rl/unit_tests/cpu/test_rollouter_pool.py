# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Rollouter's controller-side worker dispatch."""

import asyncio
import gc
from types import SimpleNamespace

from torchtitan.rl.actors.monarch import provisioning
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.system import MonarchActorSystem

from torchtitan.rl.rollout import rollouter as rollouter_module, RolloutGroup
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker


class _ChooseRunGroupEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.error: Exception | None = None

    def choose(self, **kwargs) -> asyncio.Future[RolloutGroup]:
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


def test_rollout_worker_actor_builds_configured_worker() -> None:
    async def run() -> None:
        worker = object()
        worker_config = SimpleNamespace(build=lambda: worker)
        actor = object.__new__(RolloutWorkerActor)

        RolloutWorkerActor.__init__(
            actor,
            worker_config=worker_config,
            num_threads=1,
        )

        assert actor._worker is worker

    asyncio.run(run())


class _SetupEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def call(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _WorkerActorMesh:
    def __init__(self) -> None:
        self.setup_async = _SetupEndpoint()
        self.run_group = _ChooseRunGroupEndpoint()
        self.stopped = False

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
    class WorkerConfig:
        _owner = RolloutWorker

    worker_config = WorkerConfig()
    rollouter = object.__new__(Rollouter)
    rollouter._config = SimpleNamespace(
        worker=worker_config,
        worker_pool_size=3,
        num_threads_per_worker=2,
    )
    rollouter._worker_actors = None
    rollouter._actor_system = None
    rollouter._inflight_run_group_calls = set()
    return rollouter


async def _setup(
    rollouter: Rollouter,
    actor_mesh: _WorkerActorMesh,
    monkeypatch,
) -> _WorkerMesh:
    worker_mesh = _WorkerMesh(actor_mesh)
    host = _ControllerHost(worker_mesh)
    monkeypatch.setattr(provisioning, "this_host", lambda: host)
    await rollouter.setup_async(
        actor_system=MonarchActorSystem(MonarchActorSystem.Config()),
        tokenizer_config="tokenizer_config",
        renderer_config="renderer_config",
        hf_assets_path="hf_assets_path",
    )
    return worker_mesh


def test_setup_spawns_worker_pool_on_controller_host(monkeypatch) -> None:
    async def run() -> None:
        worker_mesh = _WorkerMesh(_WorkerActorMesh())
        host = _ControllerHost(worker_mesh)
        monkeypatch.setattr(provisioning, "this_host", lambda: host)
        rollouter = _rollouter_without_datasets()

        await rollouter.setup_async(
            actor_system=MonarchActorSystem(MonarchActorSystem.Config()),
            tokenizer_config="tokenizer_config",
            renderer_config="renderer_config",
            hf_assets_path="hf_assets_path",
        )

        assert host.spawn_kwargs == {"per_host": {"cpus": 3}}
        args, kwargs = worker_mesh.spawn_args
        assert args[0] == "rollout_worker"
        assert args[1] is RolloutWorkerActor
        # The actor gets the worker's own config, not the whole Rollouter config.
        assert kwargs == {
            "worker_config": rollouter._config.worker,
            "num_threads": 2,
        }
        assert worker_mesh.actor_mesh.setup_async.calls == [
            {
                "tokenizer_config": "tokenizer_config",
                "renderer_config": "renderer_config",
                "hf_assets_path": "hf_assets_path",
            }
        ]
        await rollouter.close()
        assert worker_mesh.actor_mesh.stopped
        assert worker_mesh.stopped

    asyncio.run(run())


def test_rollout_group_dispatch_uses_choose(monkeypatch) -> None:
    async def run() -> None:
        actor_mesh = _WorkerActorMesh()
        rollouter = _rollouter_without_datasets()
        await _setup(rollouter, actor_mesh, monkeypatch)

        dispatch = asyncio.create_task(
            rollouter.run_group_rollouts(
                generate_fn="generate_fn",
                release_session_fn="release_session_fn",
                sample="sample",
                group_id=1,
                group_size=2,
                sampling="sampling",
            )
        )
        await actor_mesh.run_group.started.wait()
        assert actor_mesh.run_group.calls == [
            {
                "generate_fn": "generate_fn",
                "release_session_fn": "release_session_fn",
                "sample": "sample",
                "group_id": 1,
                "group_size": 2,
                "sampling": "sampling",
            }
        ]

        actor_mesh.run_group.release.set()
        assert (await dispatch).group_id == 1
        await rollouter.close()
        assert actor_mesh.stopped

    asyncio.run(run())


async def _dispatch_then_cancel_caller(
    rollouter: Rollouter, actor_mesh: _WorkerActorMesh
) -> None:
    """Start one `run_group` call, then cancel its caller as `Controller.run` does at the end."""
    dispatch = asyncio.create_task(
        rollouter.run_group_rollouts(
            generate_fn="generate_fn",
            release_session_fn="release_session_fn",
            sample="sample",
            group_id=1,
            group_size=2,
            sampling="sampling",
        )
    )
    await actor_mesh.run_group.started.wait()
    dispatch.cancel()
    await asyncio.gather(dispatch, return_exceptions=True)


def test_close_waits_for_run_group_of_cancelled_caller(monkeypatch) -> None:
    async def run() -> None:
        loop_errors: list[str] = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: loop_errors.append(context["message"])
        )
        actor_mesh = _WorkerActorMesh()
        rollouter = _rollouter_without_datasets()
        worker_mesh = await _setup(rollouter, actor_mesh, monkeypatch)
        await _dispatch_then_cancel_caller(rollouter, actor_mesh)

        close = asyncio.create_task(rollouter.close())
        await asyncio.sleep(0)
        # The worker still runs the group, so the mesh stays up.
        assert not worker_mesh.stopped

        # Each rollout catches its own errors, but scoring can still make run_group raise.
        actor_mesh.run_group.error = RuntimeError("scoring failed")
        actor_mesh.run_group.release.set()
        await close
        assert worker_mesh.stopped
        assert not actor_mesh.run_group.cancelled.is_set()
        # The cancelled caller never reads the error; the done callback must, or asyncio logs it here.
        gc.collect()
        assert loop_errors == []

    asyncio.run(run())


def test_close_stops_waiting_after_drain_timeout(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setattr(rollouter_module, "_RUN_GROUP_DRAIN_TIMEOUT_S", 0.0)
        actor_mesh = _WorkerActorMesh()
        rollouter = _rollouter_without_datasets()
        worker_mesh = await _setup(rollouter, actor_mesh, monkeypatch)
        await _dispatch_then_cancel_caller(rollouter, actor_mesh)

        await rollouter.close()

        assert worker_mesh.stopped
        assert not rollouter._inflight_run_group_calls

    asyncio.run(run())
