# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Rollouter's controller-side worker dispatch."""

import asyncio
import dataclasses
import functools
import gc
import os
from types import SimpleNamespace

import pytest

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch import (
    provisioning,
    rollout_worker as rollout_worker_module,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.system import MonarchActorSystem

from torchtitan.rl.rollout import rollouter as rollouter_module, RolloutGroup
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_fsdp2_gen_tp2_no_compile


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
        self.close = _CallOneEndpoint()
        self.hostname = _CallOneEndpoint()
        self.process_metrics = SimpleNamespace(
            call_one=lambda: asyncio.ensure_future(
                asyncio.sleep(0, result=[f"metric of worker {id(self)}"])
            )
        )


class _ActorSystem:
    def __init__(self, workers: list[_Worker]) -> None:
        self.workers = workers
        self.spawn_kwargs = None
        self.stopped = False
        # Each worker's number of close calls when the workers stopped.
        self.close_calls_at_stop: list[int] = []

    async def spawn_rollout_workers(self, config, **kwargs) -> list[_Worker]:
        self.spawn_kwargs = {"config": config, **kwargs}
        return self.workers

    async def stop_rollout_workers(self) -> None:
        self.stopped = True
        self.close_calls_at_stop = [len(worker.close.calls) for worker in self.workers]


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
        num_threads_per_worker=2,
        workers_on_controller_host=False,
    )
    rollouter._workers = []
    rollouter._num_rollouts_in_flight = []
    rollouter._next_worker_index = 0
    rollouter._actor_system = None
    rollouter._inflight_run_group_calls = set()
    return rollouter


async def _setup(rollouter: Rollouter, num_workers: int = 1) -> _ActorSystem:
    actor_system = _ActorSystem([_Worker() for _ in range(num_workers)])
    await rollouter.setup_async(
        actor_system=actor_system,
        num_workers=num_workers,
        generate_fn="generate_fn",
        release_session_fn="release_session_fn",
        tokenizer_config="tokenizer_config",
        renderer_config="renderer_config",
        hf_assets_path="hf_assets_path",
    )
    return actor_system


def _dispatch(rollouter: Rollouter, group_id: int, group_size: int = 2) -> asyncio.Task:
    return asyncio.create_task(
        rollouter.run_group_rollouts(
            sample="sample",
            group_id=group_id,
            group_size=group_size,
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


def test_setup_sets_up_every_worker_and_close_closes_each_before_the_stop() -> None:
    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        actor_system = await _setup(rollouter, num_workers=2)

        assert actor_system.spawn_kwargs == {
            "config": "worker_config",
            "num_workers": 2,
            "num_threads": 2,
            "on_controller_host": False,
        }
        for worker in actor_system.workers:
            assert worker.setup_async.calls == [
                {
                    "generate_fn": "generate_fn",
                    "release_session_fn": "release_session_fn",
                    "tokenizer_config": "tokenizer_config",
                    "renderer_config": "renderer_config",
                    "hf_assets_path": "hf_assets_path",
                    "vision_processor_config": None,
                }
            ]
        assert await rollouter.process_metrics() == [
            f"metric of worker {id(worker)}" for worker in actor_system.workers
        ]
        await rollouter.close()
        assert actor_system.close_calls_at_stop == [1, 1]

    asyncio.run(run())


def test_each_group_goes_to_the_worker_with_the_fewest_rollouts_in_flight() -> None:
    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        workers = (await _setup(rollouter, num_workers=3)).workers

        def group_ids_per_worker() -> list[list[int]]:
            return [
                [call["group_id"] for call in worker.run_group.calls]
                for worker in workers
            ]

        # Rollouts in flight per worker: [0, 0, 0] -> [4, 0, 0] -> [4, 2, 0] -> [4, 2, 2] -> [4, 3, 2]
        dispatches = [
            _dispatch(rollouter, group_id, group_size)
            for group_id, group_size in enumerate([4, 2, 2, 1])
        ]
        await asyncio.sleep(0)
        assert group_ids_per_worker() == [[0], [1, 3], [2]]
        # Worker 0's group ends: [0, 3, 2], so the next group goes to worker 0.
        workers[0].run_group.release.set()
        await dispatches[0]
        dispatches.append(_dispatch(rollouter, group_id=4))
        await asyncio.sleep(0)
        assert group_ids_per_worker() == [[0, 4], [1, 3], [2]]
        # [2, 3, 2]: workers 0 and 2 tie, and the last pick was worker 0, so worker 2 gets it.
        dispatches.append(_dispatch(rollouter, group_id=5, group_size=1))
        await asyncio.sleep(0)
        assert group_ids_per_worker() == [[0, 4], [1, 3], [2, 5]]
        assert workers[1].run_group.calls[0] == {
            "sample": "sample",
            "group_id": 1,
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


def test_each_worker_close_that_fails_or_hangs_is_logged_and_the_workers_still_stop(
    monkeypatch, caplog
) -> None:
    async def run() -> None:
        monkeypatch.setattr(rollouter_module, "_WORKER_CLOSE_TIMEOUT_S", 0.01)
        rollouter = _rollouter_without_datasets()
        actor_system = await _setup(rollouter, num_workers=3)
        loop = asyncio.get_running_loop()
        hung_close, failed_close = loop.create_future(), loop.create_future()
        failed_close.set_exception(RuntimeError("env stop failed"))
        actor_system.workers[0].close.call_one = lambda: hung_close
        actor_system.workers[2].close.call_one = lambda: failed_close

        await rollouter.close()

        assert actor_system.stopped
        assert hung_close.cancelled()

    asyncio.run(run())
    assert "Rollout worker 0 close still running" in caplog.text
    assert "Rollout worker 2 close failed" in caplog.text


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


def _config_with_groups(num_groups: int, group_size: int):
    """A CI recipe with `num_groups` training groups of `group_size` in flight."""
    config = rl_grpo_fsdp2_gen_tp2_no_compile()
    async_loop = dataclasses.replace(
        config.async_loop,
        target_offpolicy_steps=0,
        num_prompts_per_train_step=num_groups,
        num_samples_per_prompt=group_size,
    )
    return dataclasses.replace(config, async_loop=async_loop)


def test_worker_count_is_rollouts_in_flight_over_rollouts_per_worker_rounded_up() -> None:
    # 20 groups of 8: 160 rollouts in flight / 64 per worker = 2.5
    assert _config_with_groups(20, 8).num_rollout_workers == 3


def test_a_group_larger_than_rollouts_per_worker_raises() -> None:
    # A whole group runs on one worker, and 128 is over 64 per worker.
    with pytest.raises(ValueError, match="num_samples_per_prompt=128 is over"):
        _config_with_groups(16, 128)


def test_a_run_group_that_fails_to_send_leaves_its_worker_count_unchanged() -> None:
    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        worker = (await _setup(rollouter)).workers[0]

        def raise_while_sending(**kwargs):
            # Monarch pickles the arguments before it returns a Future.
            raise TypeError("cannot pickle '_thread.lock' object")

        worker.run_group.call_one = raise_while_sending
        with pytest.raises(TypeError, match="cannot pickle"):
            await _dispatch(rollouter, group_id=1)

        assert rollouter._num_rollouts_in_flight == [0]
        await rollouter.close()

    asyncio.run(run())


def test_the_other_hosts_stop_when_one_worker_mesh_fails_to_stop(caplog) -> None:
    class _Mesh:
        def __init__(self, error: Exception | None = None) -> None:
            self.error = error
            self.stopped = False

        async def stop(self) -> None:
            await asyncio.sleep(0)
            if self.error is not None:
                raise self.error
            self.stopped = True

    actor_system = MonarchActorSystem(MonarchActorSystem.Config())
    meshes = [_Mesh(), _Mesh(RuntimeError("host unreachable")), _Mesh()]
    actor_system._rollout_worker_meshes = list(meshes)

    asyncio.run(actor_system.stop_rollout_workers())

    assert [mesh.stopped for mesh in meshes] == [True, False, True]
    assert "Rollout worker mesh 1 stop failed" in caplog.text


def test_too_many_spread_workers_for_one_host_raise_before_any_worker_spawns(
    monkeypatch,
) -> None:
    """Spread workers on a one-node run all go to this host. The controller's check skips them
    (it can't see the hosts), so the spawn checks: 24 workers over 8 CPUs raise, and no worker
    spawns."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: range(8))
    host = _ControllerHost(worker_mesh=None)
    monkeypatch.setattr(provisioning, "this_host", lambda: host)
    actor_system = MonarchActorSystem(MonarchActorSystem.Config())

    with pytest.raises(ValueError, match="24 of 24 rollout workers go to one host"):
        asyncio.run(
            actor_system.spawn_rollout_workers(
                "worker_config", num_workers=24, num_threads=2, on_controller_host=False
            )
        )
    assert host.spawn_kwargs is None
