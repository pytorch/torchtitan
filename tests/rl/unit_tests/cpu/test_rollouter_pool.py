# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the rollout workers: where they spawn, and how the Rollouter dispatches to them."""

import asyncio
import functools
import gc
import os
import pathlib
import re
import socket
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import cloudpickle
import pytest
from monarch.actor import Actor, ActorError, endpoint, this_host
from monarch.job import ProcessJob

from torchtitan.observability import structured_logger as sl
from torchtitan.rl._env import apply_rollout_worker_env_defaults
from torchtitan.rl.actors.monarch import (
    provisioning,
    rollout_worker as rollout_worker_module,
    system as monarch_system,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.actors.system import RolloutWorkersHandle
from torchtitan.rl.examples.slake.data import SlakeDataset
from torchtitan.rl.observability import metrics as m

from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.rollout.data import RLDataset
from torchtitan.rl.rollout.rollouter import (
    BaseRolloutWorker,
    MessageEnvRolloutWorker,
    Rollouter,
    RolloutWorkerRoles,
)


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


def test_rollout_worker_actor_inits_its_logger_and_builds_its_worker_at_setup(
    monkeypatch,
) -> None:
    async def run() -> None:
        worker = _LoopBlockingWorker()
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
            first_worker_index=3,
        )

        # Rank 2 in the proc mesh of a host whose first worker is worker 3.
        assert init_calls == [{"source": "rl_rollout_worker", "rank": 5}]
        # The worker is built in `setup_async`, so a build error reaches the caller as an endpoint
        # error instead of killing the actor.
        assert not hasattr(actor, "_worker")
        setup_async = RolloutWorkerActor.setup_async._method.__wrapped__
        await setup_async(
            actor,
            tokenizer_config=_NO_OP_CONFIG,
            renderer_config=_NO_OP_CONFIG,
            hf_assets_path="",
            paths_on_every_host=[],
        )
        assert actor._worker is worker
        worker._event_loop_lag.close()
        worker._thread_pool_lag.close()

    try:
        asyncio.run(run())
    finally:
        # Give the frozen objects back to the collector, for the tests that run after this one.
        gc.unfreeze()


# A tokenizer or renderer config for fake workers: the base `setup_async` builds None from it.
_NO_OP_CONFIG = SimpleNamespace(build=lambda **kwargs: None)


class _LoopBlockingWorker(BaseRolloutWorker):
    """Holds its event loop for 0.3 s in setup and for `sample` seconds in each group, as a slow
    tokenizer build or render would. Then holds its one thread for `sample` seconds, as a
    blocking sandbox removal would."""

    def __init__(self) -> None:
        # No rubric or advantage estimator: this worker never scores.
        pass

    async def setup_async(self, **kwargs) -> None:
        # Yield first: a lag tracker started before setup would run now, then record the 0.3 s hold.
        await asyncio.sleep(0.1)
        time.sleep(0.3)
        await super().setup_async(**kwargs)

    async def run_group(
        self, *, sample: float, group_id: int, **kwargs
    ) -> RolloutGroup:
        time.sleep(sample)
        await asyncio.to_thread(time.sleep, sample)
        # Yield, so the lag task wakes up late and records the delay before the group returns.
        await asyncio.sleep(0.1)
        return RolloutGroup(group_id=group_id, rollouts=[])


def test_rollout_worker_reports_its_metrics() -> None:
    """A training group gets the worker's metrics since its previous training group. A validation
    group gets none:
        setup holds the loop for 0.3 s                    not counted
        training group                                    lag and thread wait < 250 ms
        validation group holds the loop, then the thread,
            for 0.3 s each                                no metrics
        training group                                    lag and thread wait >= 250 ms
    """

    async def run() -> list[RolloutGroup]:
        actor = object.__new__(RolloutWorkerActor)
        RolloutWorkerActor.__init__(
            actor,
            worker_config=SimpleNamespace(build=_LoopBlockingWorker),
            num_threads=1,
            structured_logger_init_fn=None,
            first_worker_index=0,
        )
        # Let the loop run between the spawn and the setup call, as in a real actor. Then a lag task
        # started in `__init__` would record setup's 0.3 s hold, so the "not counted" row can fail.
        await asyncio.sleep(0)
        # Call each endpoint's own body, which `concurrent_endpoint` keeps as `__wrapped__`.
        # The wrapper needs a Monarch response port, and this test has none.
        setup_async = RolloutWorkerActor.setup_async._method.__wrapped__
        run_group = RolloutWorkerActor.run_group._method.__wrapped__
        freeze_count_before_setup = gc.get_freeze_count()
        await setup_async(
            actor,
            tokenizer_config=_NO_OP_CONFIG,
            renderer_config=_NO_OP_CONFIG,
            hf_assets_path="",
            paths_on_every_host=[],
        )
        # Setup moved the objects built so far out of garbage collection.
        assert gc.get_freeze_count() > freeze_count_before_setup
        kwargs = dict(
            generate_fn=None, release_session_fn=None, group_size=1, sampling=None
        )
        groups = [
            await run_group(actor, sample=0.0, group_id=0, **kwargs),
            await run_group(actor, sample=0.3, group_id=-1, **kwargs),
            await run_group(actor, sample=0.0, group_id=1, **kwargs),
        ]
        actor._worker._event_loop_lag.close()
        actor._worker._thread_pool_lag.close()
        return groups

    try:
        first_training, validation, second_training = asyncio.run(run())
    finally:
        # Give the frozen objects back to the collector, for the tests that run after this one.
        gc.unfreeze()
    assert validation.metrics == []
    first_metrics = m.MetricsProcessor._aggregate_metrics(first_training.metrics)
    second_metrics = m.MetricsProcessor._aggregate_metrics(second_training.metrics)
    lag_mean = next(
        metric.value
        for metric in second_training.metrics
        if metric.key == "rollout_worker/event_loop_lag_ms_per_s"
    )
    # The lag task checks every 0.05 s, so the 0.3 s hold reads at least 0.25 s.
    assert first_metrics["rollout_worker/event_loop_lag_ms/max"] < 250.0
    assert second_metrics["rollout_worker/event_loop_lag_ms/max"] >= 250.0
    # `Mean.value` is the total ms late. The 0.3 s hold adds at least 250 ms.
    assert lag_mean.value >= 250.0
    assert first_metrics["rollout_worker/thread_pool_lag_ms/max"] < 250.0
    # The lag task checks every 0.05 s, so the thread's 0.3 s hold reads at least 0.25 s.
    assert second_metrics["rollout_worker/thread_pool_lag_ms/max"] >= 250.0
    assert second_metrics["rollout_worker/ram_gib/max"] > 0.0
    assert second_metrics["rollout_worker/peak_ram_gib/max"] > 0.0


class _CallEndpoint:
    """Records each `.call(...)`."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def call(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _Worker:
    """One worker's handle: `run_group.call_one` reaches this worker only."""

    def __init__(self) -> None:
        self.run_group = _RunGroupEndpoint()


class _WorkerActorMesh:
    """One host's workers, as `spawn` returns them: `flatten("rank").slice(rank=r)` is worker r."""

    def __init__(self, num_workers: int) -> None:
        self.setup_async = _CallEndpoint()
        self.sync_log_step = _CallEndpoint()
        self.close = _CallEndpoint()
        self.workers = [_Worker() for _ in range(num_workers)]
        self.stopped = False
        # How many times `close` ran when the workers stopped.
        self.close_calls_at_stop: int | None = None

    def __len__(self) -> int:
        return len(self.workers)

    def flatten(self, name: str) -> "_WorkerActorMesh":
        return self

    def slice(self, *, rank: int) -> _Worker:
        return self.workers[rank]

    async def stop(self) -> None:
        self.stopped = True
        self.close_calls_at_stop = len(self.close.calls)


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
        _owner = MessageEnvRolloutWorker

    worker_config = WorkerConfig()
    rollouter = object.__new__(Rollouter)
    rollouter._config = SimpleNamespace(
        worker=worker_config,
        rollouts_per_worker=7,
        rollouts_per_thread=4,
        run_group_drain_timeout_s=30.0,
        worker_close_timeout_s=60.0,
        worker_roles=RolloutWorkerRoles(),
        training_dataloader=SimpleNamespace(dataset=RLDataset.Config()),
    )
    # 20 training rollouts in flight / 7 per worker -> 3 workers
    rollouter._max_rollouts_in_flight = 20
    rollouter._workers = None
    rollouter._actor_system = None
    rollouter._inflight_run_group_calls = set()
    return rollouter


async def _setup(
    rollouter: Rollouter,
    actor_mesh: _WorkerActorMesh,
    monkeypatch,
    actor_system: MonarchActorSystem | None = None,
) -> _WorkerMesh:
    worker_mesh = _WorkerMesh(actor_mesh)
    host = _ControllerHost(worker_mesh)
    monkeypatch.setattr(provisioning, "this_host", lambda: host)
    # One worker per actor of the fake mesh.
    rollouter._max_rollouts_in_flight = rollouter._config.rollouts_per_worker * len(
        actor_mesh
    )
    await rollouter.setup_async(
        actor_system=actor_system or MonarchActorSystem(MonarchActorSystem.Config()),
        tokenizer_config="tokenizer_config",
        renderer_config="renderer_config",
        hf_assets_path="hf_assets_path",
    )
    return worker_mesh


def test_setup_raises_when_one_host_would_run_too_many_rollout_workers(
    monkeypatch,
) -> None:
    actor_system = MonarchActorSystem(
        MonarchActorSystem.Config(max_rollout_workers_per_host=1)
    )
    # No launcher, so all 3 workers would run on the controller's host.
    with pytest.raises(ValueError, match="max_rollout_workers_per_host"):
        asyncio.run(
            _setup(
                _rollouter_without_datasets(),
                _WorkerActorMesh(num_workers=3),
                monkeypatch,
                actor_system=actor_system,
            )
        )


def test_setup_spawns_worker_pool_on_controller_host(monkeypatch) -> None:
    async def run() -> None:
        worker_mesh = _WorkerMesh(_WorkerActorMesh(num_workers=3))
        host = _ControllerHost(worker_mesh)
        monkeypatch.setattr(provisioning, "this_host", lambda: host)
        structured_logger_init_fn = object()
        monkeypatch.setattr(
            sl,
            "get_structured_logger_subprocess_init_fn",
            lambda: structured_logger_init_fn,
        )
        rollouter = _rollouter_without_datasets()
        rollouter._config.worker_roles = RolloutWorkerRoles(
            generator=None, controller=True
        )
        # A launcher's hosts. Its controller runs outside them, so `controller=True` alone puts
        # every worker on `this_host()`.
        launcher_hosts = provisioning.HostMeshes(
            trainer=_FakeHostMesh(num_hosts=1, name="t"),
            generators=[_FakeHostMesh(num_hosts=1, name="g")],
            gpus_per_node=4,
            controller_on_first_trainer_host=False,
        )

        await rollouter.setup_async(
            actor_system=MonarchActorSystem(
                MonarchActorSystem.Config(), host_meshes=launcher_hosts
            ),
            tokenizer_config="tokenizer_config",
            renderer_config="renderer_config",
            hf_assets_path="hf_assets_path",
        )

        assert host.spawn_kwargs == {
            "per_host": {"cpus": 3},
            "bootstrap": apply_rollout_worker_env_defaults,
        }
        args, kwargs = worker_mesh.spawn_args
        assert args[0] == "rollout_worker"
        assert args[1] is RolloutWorkerActor
        # The actor gets the worker's own config, not the whole Rollouter config.
        assert kwargs == {
            "worker_config": rollouter._config.worker,
            "num_threads": 2,
            "structured_logger_init_fn": structured_logger_init_fn,
            "first_worker_index": 0,
        }
        assert rollouter._workers.by_host == [worker_mesh.actor_mesh]
        assert rollouter._workers.by_worker == worker_mesh.actor_mesh.workers
        assert worker_mesh.actor_mesh.setup_async.calls == [
            {
                "tokenizer_config": "tokenizer_config",
                "renderer_config": "renderer_config",
                "hf_assets_path": "hf_assets_path",
                "paths_on_every_host": [],
                "vision_processor_config": None,
            }
        ]
        await rollouter.close()
        # The workers' close ran once, before they stopped.
        assert worker_mesh.actor_mesh.close_calls_at_stop == 1
        assert worker_mesh.stopped

    asyncio.run(run())


def test_setup_and_sync_log_step_reach_the_workers_on_every_host() -> None:
    host_meshes = [_WorkerActorMesh(num_workers=2), _WorkerActorMesh(num_workers=1)]

    class _ActorSystem:
        async def spawn_rollout_workers(self, config, **kwargs) -> RolloutWorkersHandle:
            self.spawn_kwargs = kwargs
            return RolloutWorkersHandle(
                by_host=host_meshes,
                by_worker=[worker for mesh in host_meshes for worker in mesh.workers],
            )

    async def run() -> None:
        actor_system = _ActorSystem()
        rollouter = _rollouter_without_datasets()
        await rollouter.setup_async(
            actor_system=actor_system,
            tokenizer_config="tokenizer_config",
            renderer_config="renderer_config",
            hf_assets_path="hf_assets_path",
        )
        await rollouter.sync_log_step(step=7)

        assert actor_system.spawn_kwargs["worker_roles"] == RolloutWorkerRoles()
        assert [len(mesh.setup_async.calls) for mesh in host_meshes] == [1, 1]
        assert [mesh.sync_log_step.calls for mesh in host_meshes] == [
            [{"step": 7}],
            [{"step": 7}],
        ]

    asyncio.run(run())


class _FakeHostMesh:
    """A launcher's host mesh of `num_hosts` hosts: host i of a mesh named "g" is "g{i}"."""

    def __init__(self, *, num_hosts: int, name: str) -> None:
        self.num_hosts = num_hosts
        self.name = name

    def __len__(self) -> int:
        return self.num_hosts

    def flatten(self, name: str) -> "_FakeHostMesh":
        return self

    def slice(self, *, hosts: int) -> str:
        return f"{self.name}{hosts}"


def _host_meshes(
    trainer, generators, controller_on_first_trainer_host
) -> provisioning.HostMeshes | None:
    """Mesh "g*2" is 2 hosts, g0 and g1. Generators that name the same mesh share it."""
    if trainer is None:
        return None
    meshes = {
        spec: _FakeHostMesh(num_hosts=int(spec.split("*")[1]), name=spec.split("*")[0])
        for spec in [trainer, *generators]
    }
    return provisioning.HostMeshes(
        trainer=meshes[trainer],
        generators=[meshes[spec] for spec in generators],
        gpus_per_node=4,
        controller_on_first_trainer_host=controller_on_first_trainer_host,
    )


def _worker_roles(roles: dict[str, bool]) -> RolloutWorkerRoles:
    """Example: {"trainer": True, "controller": False}
    -> RolloutWorkerRoles(trainer=True, generator=None, controller=False)."""
    return RolloutWorkerRoles(
        **{"trainer": None, "generator": None, "controller": None, **roles}
    )


@pytest.mark.parametrize(
    "roles, trainer, generators, controller_on_first_trainer_host, expected",
    [
        # No launcher: the job is this host alone.
        ({"trainer": True, "generator": True}, None, [], False, ["this_host"]),
        # One host for every role, which runs the controller: listed once.
        (
            {"trainer": True, "generator": True, "controller": True},
            "t*1",
            ["t*1"],
            True,
            ["this_host"],
        ),
        # A generator on the trainer's node, which runs the controller: listed once.
        ({"generator": True, "controller": True}, "t*1", ["t*1"], True, ["this_host"]),
        # 4 hosts: trainer t0, t1 and two generators that share mesh g. The controller on t0, so
        # t0 is listed as `this_host()`.
        # Generator hosts:
        ({"generator": True}, "t*2", ["g*2", "g*2"], True, ["g0", "g1"]),
        # Every host but the controller's:
        (
            {"trainer": True, "generator": True, "controller": False},
            "t*2",
            ["g*2", "g*2"],
            True,
            ["t1", "g0", "g1"],
        ),
        # Every host:
        (
            {"trainer": True, "generator": True},
            "t*2",
            ["g*2", "g*2"],
            True,
            ["this_host", "t1", "g0", "g1"],
        ),
        # The controller's host only:
        ({"controller": True}, "t*2", ["g*2", "g*2"], True, ["this_host"]),
        # The controller's host and another role's hosts: the controller's host counts once.
        (
            {"trainer": True, "controller": True},
            "t*2",
            ["g*2", "g*2"],
            True,
            ["this_host", "t1"],
        ),
        (
            {"generator": True, "controller": True},
            "t*2",
            ["g*2", "g*2"],
            True,
            ["this_host", "g0", "g1"],
        ),
        # The same hosts, the controller outside the job.
        ({"controller": True}, "t*2", ["g*2", "g*2"], False, ["this_host"]),
        (
            {"trainer": True, "controller": True},
            "t*2",
            ["g*2", "g*2"],
            False,
            ["t0", "t1", "this_host"],
        ),
        (
            {"trainer": True, "controller": False},
            "t*2",
            ["g*2", "g*2"],
            False,
            ["t0", "t1"],
        ),
        # t0 runs the trainer and a generator: generator=False vetoes it.
        (
            {"trainer": True, "generator": False, "controller": True},
            "t*1",
            ["t*1", "g*1"],
            False,
            ["this_host"],
        ),
    ],
)
def test_rollout_worker_hosts(
    monkeypatch, roles, trainer, generators, controller_on_first_trainer_host, expected
):
    monkeypatch.setattr(provisioning, "this_host", lambda: "this_host")
    hosts = provisioning.rollout_worker_hosts(
        host_meshes=_host_meshes(trainer, generators, controller_on_first_trainer_host),
        worker_roles=_worker_roles(roles),
    )
    assert hosts == expected


def test_rollout_worker_hosts_raises_when_every_host_is_vetoed(monkeypatch) -> None:
    # The real `this_host()`, called outside asyncio, leaves Monarch's context set in this thread.
    # Monarch's log filter then prefixes later log lines, and test_shutdown.py's log checks fail.
    monkeypatch.setattr(provisioning, "this_host", lambda: "this_host")
    # One host runs the trainer, the generator and the controller: generator=False vetoes it.
    with pytest.raises(ValueError, match="picks no host"):
        provisioning.rollout_worker_hosts(
            host_meshes=_host_meshes("t*1", ["t*1"], True),
            worker_roles=_worker_roles({"generator": False, "controller": True}),
        )


def test_rollout_worker_hosts_uses_the_default_of_an_unset_role(monkeypatch) -> None:
    # `generator` defaults to True, so `controller=True` alone adds the controller's host and keeps
    # the generator hosts.
    monkeypatch.setattr(provisioning, "this_host", lambda: "this_host")
    hosts = provisioning.rollout_worker_hosts(
        host_meshes=_host_meshes("t*2", ["g*2", "g*2"], True),
        worker_roles=RolloutWorkerRoles(controller=True),
    )
    assert hosts == ["this_host", "g0", "g1"]


def _simulated_host() -> str:
    """The host this process runs on, as `ProcessJob` names it, e.g. "trainer_1"."""
    # TODO: read a public Monarch value once one names ProcessJob hosts. This env var is private,
    #   and the hosts can't be told apart by hostname: they all run on this machine.
    # e.g. "anon-0 @ trainer_1"
    return os.environ["HYPERACTOR_PROCESS_NAME"].split("@")[-1].strip()


def _record_log_rank(marker_dir: str, *, source: str, rank: int) -> None:
    """Stands in for the structured logger's init: one file per worker, naming its log rank and
    its host."""
    pathlib.Path(marker_dir, f"{rank}-{_simulated_host()}").touch()


class _EchoActor(Actor):
    """Stands in for the generator router: an actor on the controller's host."""

    @endpoint
    async def times_ten(self, value: int) -> int:
        return value * 10

    @endpoint
    async def host(self) -> str:
        return _simulated_host()


async def _ask_echo(echo, *, group_id: int) -> int:
    return await echo.times_ten.call_one(group_id)


class _SpreadWorker(BaseRolloutWorker):
    """Runs a group by calling `generate_fn`, and leaves a file naming the group and its host."""

    def __init__(self, marker_dir: str) -> None:
        # No rubric or advantage estimator: this worker never scores.
        self.marker_dir = marker_dir

    async def run_group(self, *, generate_fn, group_id, **kwargs) -> RolloutGroup:
        pathlib.Path(self.marker_dir, f"ran-{group_id}-{_simulated_host()}").touch()
        return RolloutGroup(group_id=await generate_fn(group_id=group_id), rollouts=[])


@dataclass
class _SpreadWorkerConfig:
    marker_dir: str

    def build(self) -> _SpreadWorker:
        return _SpreadWorker(self.marker_dir)


def test_rollout_workers_spread_over_the_hosts_of_a_job(tmp_path, monkeypatch):
    """A 4-host job simulated on this machine, the controller on trainer host 0. With all three
    roles True, trainer host 0 counts once and spawns through `this_host()`, the controller's
    machine. 5 workers go 2, 1, 1, 1 to the 4 hosts in job order, with log ranks 0-4."""
    monkeypatch.setattr(
        monarch_system.sl,
        "get_structured_logger_subprocess_init_fn",
        lambda: functools.partial(_record_log_rank, str(tmp_path)),
    )

    async def run() -> tuple[list[int], str]:
        # ======== Start a 4-host job simulated on this machine ========
        # One host mesh per role, as `Layout.host_meshes` builds them.
        job = ProcessJob({"trainer": 2, "generator_0": 1, "generator_1": 1})
        state = job.state(cached_path=None)
        echo_procs = this_host().spawn_procs(per_host={"cpus": 1})
        system = MonarchActorSystem(
            MonarchActorSystem.Config(),
            host_meshes=provisioning.HostMeshes(
                trainer=state.trainer,
                generators=[state.generator_0, state.generator_1],
                gpus_per_node=4,
                controller_on_first_trainer_host=True,
            ),
        )
        # The worker processes can't import this test module, so ship the fakes by value.
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
        try:
            # ======== Spawn 5 workers and run one group on each ========
            echo = echo_procs.spawn("echo", _EchoActor)
            workers = await system.spawn_rollout_workers(
                _SpreadWorkerConfig(str(tmp_path)),
                num_workers=5,
                num_threads=1,
                worker_roles=RolloutWorkerRoles(
                    trainer=True, generator=True, controller=True
                ),
            )
            # The Rollouter's calls: `.call()` on each host's workers, `call_one` on one worker.
            await asyncio.gather(
                *(
                    host_workers.setup_async.call(
                        tokenizer_config=_NO_OP_CONFIG,
                        renderer_config=_NO_OP_CONFIG,
                        hf_assets_path="",
                        paths_on_every_host=[],
                    )
                    for host_workers in workers.by_host
                )
            )
            await asyncio.gather(
                *(
                    host_workers.sync_log_step.call(step=1)
                    for host_workers in workers.by_host
                )
            )
            groups = await asyncio.gather(
                *(
                    worker.run_group.call_one(
                        generate_fn=functools.partial(_ask_echo, echo),
                        release_session_fn=None,
                        sample=None,
                        group_id=group_id,
                        group_size=1,
                        sampling=None,
                    )
                    for group_id, worker in enumerate(workers.by_worker)
                )
            )
            await system.stop_rollout_workers()
            return [group.group_id for group in groups], await echo.host.call_one()
        finally:
            cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
            await echo_procs.stop()
            job.kill()

    group_ids, controller_host = asyncio.run(run())
    # Each group called the actor on this host: a cross-mesh call, the path a cross-host call takes.
    assert group_ids == [0, 10, 20, 30, 40]
    host_by_worker = [
        controller_host,
        controller_host,
        "trainer_1",
        "generator_0_0",
        "generator_1_0",
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [f"{rank}-{host}" for rank, host in enumerate(host_by_worker)]
        + [f"ran-{rank}-{host}" for rank, host in enumerate(host_by_worker)]
    )


def test_a_group_goes_to_one_worker(monkeypatch) -> None:
    async def run() -> None:
        actor_mesh = _WorkerActorMesh(num_workers=3)
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
        # `run_group_rollouts` sends the call before its first wait.
        await asyncio.sleep(0)
        calls = [worker.run_group.calls for worker in actor_mesh.workers]
        assert sorted(calls, key=len) == [
            [],
            [],
            [
                {
                    "generate_fn": "generate_fn",
                    "release_session_fn": "release_session_fn",
                    "sample": "sample",
                    "group_id": 1,
                    "group_size": 2,
                    "sampling": "sampling",
                }
            ],
        ]

        for worker in actor_mesh.workers:
            worker.run_group.release.set()
        assert (await dispatch).group_id == 1
        await rollouter.close()
        assert actor_mesh.stopped

    asyncio.run(run())


async def _dispatch_then_cancel_caller(rollouter: Rollouter, worker: _Worker) -> None:
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
    await worker.run_group.started.wait()
    dispatch.cancel()
    await asyncio.gather(dispatch, return_exceptions=True)


def test_close_waits_for_run_group_of_cancelled_caller(monkeypatch) -> None:
    async def run() -> None:
        loop_errors: list[str] = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: loop_errors.append(context["message"])
        )
        actor_mesh = _WorkerActorMesh(num_workers=1)
        (worker,) = actor_mesh.workers
        rollouter = _rollouter_without_datasets()
        worker_mesh = await _setup(rollouter, actor_mesh, monkeypatch)
        await _dispatch_then_cancel_caller(rollouter, worker)

        close = asyncio.create_task(rollouter.close())
        await asyncio.sleep(0)
        # The worker still runs the group, so the mesh stays up.
        assert not worker_mesh.stopped

        # Each rollout catches its own errors, but scoring can still make run_group raise.
        worker.run_group.error = RuntimeError("scoring failed")
        worker.run_group.release.set()
        await close
        assert worker_mesh.stopped
        assert not worker.run_group.cancelled.is_set()
        # The cancelled caller never reads the error; the done callback must, or asyncio logs it here.
        gc.collect()
        assert loop_errors == []

    asyncio.run(run())


def test_close_stops_waiting_after_drain_timeout(monkeypatch) -> None:
    async def run() -> None:
        actor_mesh = _WorkerActorMesh(num_workers=1)
        rollouter = _rollouter_without_datasets()
        rollouter._config.run_group_drain_timeout_s = 0.0
        worker_mesh = await _setup(rollouter, actor_mesh, monkeypatch)
        await _dispatch_then_cancel_caller(rollouter, actor_mesh.workers[0])

        await rollouter.close()

        assert worker_mesh.stopped
        assert not rollouter._inflight_run_group_calls

    asyncio.run(run())


class _HangingCallEndpoint(_CallEndpoint):
    """Records each `.call(...)`, which then never returns."""

    async def call(self, **kwargs) -> None:
        self.calls.append(kwargs)
        await asyncio.Event().wait()


@pytest.mark.parametrize("close_ends_with", ["return", "hang"])
def test_close_closes_the_workers_on_every_host_then_stops_them(
    close_ends_with, caplog
) -> None:
    """Both hosts' workers get `close`. If both hang, `close` waits one `worker_close_timeout_s`
    for all hosts together, not one per host, then stops the workers anyway."""
    host_meshes = [_WorkerActorMesh(num_workers=1), _WorkerActorMesh(num_workers=1)]
    if close_ends_with == "hang":
        for mesh in host_meshes:
            mesh.close = _HangingCallEndpoint()
    stopped_after_close_calls = []

    async def stop_rollout_workers() -> None:
        stopped_after_close_calls.append(
            [len(mesh.close.calls) for mesh in host_meshes]
        )

    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        rollouter._config.worker_close_timeout_s = 0.05
        rollouter._workers = RolloutWorkersHandle(by_host=host_meshes, by_worker=[])
        rollouter._actor_system = SimpleNamespace(
            stop_rollout_workers=stop_rollout_workers
        )
        await rollouter.close()

    asyncio.run(run())
    # The workers stop once, after every host's close was called.
    assert stopped_after_close_calls == [[1, 1]]
    # One limit for all hosts: one warning. One limit per host would warn once per host.
    expected_warnings = 1 if close_ends_with == "hang" else 0
    assert caplog.text.count("still closing after") == expected_warnings


class _RaisingCallEndpoint(_CallEndpoint):
    """Records each `.call(...)`, then raises, as a dead worker's endpoint does."""

    async def call(self, **kwargs) -> None:
        self.calls.append(kwargs)
        raise RuntimeError("worker died")


class _SlowCallEndpoint(_CallEndpoint):
    """Records each `.call(...)`, which returns after a short wait."""

    def __init__(self) -> None:
        super().__init__()
        self.finished = False

    async def call(self, **kwargs) -> None:
        self.calls.append(kwargs)
        await asyncio.sleep(0.05)
        self.finished = True


def test_one_host_failing_to_close_doesnt_cut_the_other_hosts_short(caplog) -> None:
    dead_host = _WorkerActorMesh(num_workers=1)
    slow_host = _WorkerActorMesh(num_workers=1)
    dead_host.close = _RaisingCallEndpoint()
    slow_host.close = _SlowCallEndpoint()
    slow_host_finished_at_stop = []

    async def stop_rollout_workers() -> None:
        slow_host_finished_at_stop.append(slow_host.close.finished)

    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        rollouter._workers = RolloutWorkersHandle(
            by_host=[dead_host, slow_host], by_worker=[]
        )
        rollouter._actor_system = SimpleNamespace(
            stop_rollout_workers=stop_rollout_workers
        )
        await rollouter.close()

    asyncio.run(run())
    assert slow_host_finished_at_stop == [True]
    assert "rollout workers on host 0 failed to close" in caplog.text


def test_close_stops_the_workers_of_a_spawn_that_failed_partway() -> None:
    """`spawn_rollout_workers` raised after starting some hosts: no handle, but those stop."""
    stop_calls = []

    async def stop_rollout_workers() -> None:
        stop_calls.append(True)

    async def run() -> None:
        rollouter = _rollouter_without_datasets()
        rollouter._actor_system = SimpleNamespace(
            stop_rollout_workers=stop_rollout_workers
        )
        await rollouter.close()

    asyncio.run(run())
    assert stop_calls == [True]


class _NoOpWorker:
    async def setup_async(self, **kwargs) -> None:
        pass

    async def close(self) -> None:
        pass


class _NoOpWorkerConfig:
    def build(self) -> _NoOpWorker:
        return _NoOpWorker()


def test_setup_fails_naming_a_missing_dataset_dir(tmp_path) -> None:
    """One host, SLAKE's data dir overridden to a dir this host lacks: setup fails naming the
    overridden dir and the host."""
    data_dir = tmp_path / "slake"
    dataset = SlakeDataset.Config(data_dir="./assets/datasets/slake")
    # What an override of the dataset config does.
    dataset.data_dir = str(data_dir)
    rollouter = _rollouter_without_datasets()
    rollouter._config.worker = _NoOpWorkerConfig()
    rollouter._config.training_dataloader = SimpleNamespace(dataset=dataset)
    # One worker.
    rollouter._max_rollouts_in_flight = rollouter._config.rollouts_per_worker

    async def run() -> None:
        # The worker processes can't import this test module, so ship the fakes by value.
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
        try:
            await rollouter.setup_async(
                actor_system=MonarchActorSystem(MonarchActorSystem.Config()),
                tokenizer_config=None,
                renderer_config=None,
                hf_assets_path="",
            )
        finally:
            await rollouter.close()
            cloudpickle.unregister_pickle_by_value(sys.modules[__name__])

    with pytest.raises(
        ActorError,
        match=re.escape(f"{data_dir} is missing on host {socket.gethostname()}."),
    ):
        asyncio.run(run())
