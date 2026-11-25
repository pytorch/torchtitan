# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Rollouter's controller-side worker dispatch."""

import asyncio
import functools
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch import (
    provisioning,
    rollout_worker as rollout_worker_module,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.observability import metrics as m

from torchtitan.rl.rollout import rollouter as rollouter_module, RolloutGroup
from torchtitan.rl.rollout.rollouter import (
    BaseRolloutWorker,
    MessageEnvRolloutWorker,
    Rollouter,
)


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
            worker_kwargs={},
            num_threads=1,
            # Bound like the controller's; the worker overrides source and rank.
            structured_logger_init_fn=functools.partial(
                lambda **kwargs: init_calls.append(kwargs), source="rl", rank=0
            ),
        )

        assert actor._worker is worker
        assert init_calls == [{"source": "rl_rollout_worker", "rank": 2}]

    asyncio.run(run())


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
        group = RolloutGroup(group_id=group_id, rollouts=[])
        group.metrics += self.flush_process_metrics(group_id)
        return group


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
        # One thread, so a group's 0.3 s call in a thread delays the thread-pool lag check.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1)
        )
        worker = _LoopBlockingWorker()
        freeze_count_before_setup = gc.get_freeze_count()
        await worker.setup_async(
            tokenizer_config=_NO_OP_CONFIG,
            renderer_config=_NO_OP_CONFIG,
            hf_assets_path="",
        )
        # Setup moved the objects built so far out of garbage collection.
        assert gc.get_freeze_count() > freeze_count_before_setup
        kwargs = dict(
            generate_fn=None, release_session_fn=None, group_size=1, sampling=None
        )
        groups = [
            await worker.run_group(sample=0.0, group_id=0, **kwargs),
            await worker.run_group(sample=0.3, group_id=-1, **kwargs),
            await worker.run_group(sample=0.0, group_id=1, **kwargs),
        ]
        worker._event_loop_lag.close()
        worker._thread_pool_lag.close()
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
        _owner = MessageEnvRolloutWorker

    worker_config = WorkerConfig()
    rollouter = object.__new__(Rollouter)
    rollouter._config = SimpleNamespace(
        worker=worker_config, rollouts_per_worker=7, rollouts_per_thread=4
    )
    # 20 training rollouts in flight / 7 per worker -> 3 workers
    rollouter._max_rollouts_in_flight = 20
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
        structured_logger_init_fn = object()
        monkeypatch.setattr(
            sl,
            "get_structured_logger_subprocess_init_fn",
            lambda: structured_logger_init_fn,
        )
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
            "worker_kwargs": {},
            "num_threads": 2,
            "structured_logger_init_fn": structured_logger_init_fn,
        }
        assert worker_mesh.actor_mesh.setup_async.calls == [
            {
                "tokenizer_config": "tokenizer_config",
                "renderer_config": "renderer_config",
                "hf_assets_path": "hf_assets_path",
                "vision_processor_config": None,
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
