# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Rollouter's controller-side worker dispatch."""

import asyncio
from types import SimpleNamespace

from torchtitan.experiments.rl.rollout import (
    rollouter as rollouter_module,
    RolloutGroup,
)
from torchtitan.experiments.rl.rollout.rollouter import Rollouter


class _ChooseRunGroupEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def choose(self, **kwargs) -> asyncio.Future[RolloutGroup]:
        self.calls.append(kwargs)
        self.started.set()
        return asyncio.create_task(self._execute(kwargs))

    async def _execute(self, kwargs) -> RolloutGroup:
        await self.release.wait()
        return RolloutGroup(group_id=kwargs["group_id"], rollouts=[])


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
    rollouter = object.__new__(Rollouter)
    rollouter._config = SimpleNamespace(
        worker="worker_config",
        worker_pool_size=3,
        num_threads_per_worker=2,
    )
    rollouter._worker_actors = None
    rollouter._worker_mesh = None
    return rollouter


async def _setup(
    rollouter: Rollouter,
    actor_mesh: _WorkerActorMesh,
    monkeypatch,
) -> _WorkerMesh:
    worker_mesh = _WorkerMesh(actor_mesh)
    host = _ControllerHost(worker_mesh)
    monkeypatch.setattr(rollouter_module, "this_host", lambda: host)
    await rollouter.setup_async(
        tokenizer_config="tokenizer_config",
        renderer_config="renderer_config",
        hf_assets_path="hf_assets_path",
    )
    return worker_mesh


def test_setup_spawns_worker_pool_on_controller_host(monkeypatch) -> None:
    async def run() -> None:
        worker_mesh = _WorkerMesh(_WorkerActorMesh())
        host = _ControllerHost(worker_mesh)
        monkeypatch.setattr(rollouter_module, "this_host", lambda: host)
        rollouter = _rollouter_without_datasets()

        await rollouter.setup_async(
            tokenizer_config="tokenizer_config",
            renderer_config="renderer_config",
            hf_assets_path="hf_assets_path",
        )

        assert host.spawn_kwargs == {"per_host": {"cpus": 3}}
        args, kwargs = worker_mesh.spawn_args
        assert args[0] == "rollout_worker"
        assert args[1].__name__ == "RolloutWorkerActor"
        # The actor gets the worker's own config, not the whole Rollouter config.
        assert kwargs == {
            "worker_config": "worker_config",
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
