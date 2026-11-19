# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Rollouter's controller-side worker dispatch."""

import asyncio
import functools
import gc
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from renderers import PlaceholderRange

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch import (
    provisioning,
    rollout_worker as rollout_worker_module,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.system import MonarchActorSystem

from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import rollouter as rollouter_module, RolloutGroup
from torchtitan.rl.rollout.rollouter import MessageEnvRolloutWorker, Rollouter
from torchtitan.rl.rollout.types import (
    PromptImage,
    Rollout,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)


class _ChooseRunGroupEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.error: Exception | None = None
        # The rollouts of the group the worker returns.
        self.rollouts: list[Rollout] = []

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
        return RolloutGroup(group_id=kwargs["group_id"], rollouts=self.rollouts)


def test_rollout_worker_actor_builds_its_worker_and_inits_its_logger(
    monkeypatch,
) -> None:
    async def run() -> None:
        worker = object()
        worker_config = SimpleNamespace(build=lambda: worker, rdma_min_image_bytes=None)
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
        worker=worker_config, num_threads_per_worker=2, rollouts_per_worker=7
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


@dataclass(kw_only=True, slots=True)
class _TurnWithTopkLogprobs(RolloutTurn):
    """A turn with one more per-token array, as score centering adds top-k logprobs."""

    completion_topk_logprobs: np.ndarray


def test_run_group_rollouts_logs_the_call_duration_and_the_payload(monkeypatch) -> None:
    async def run() -> dict[str, float]:
        actor_mesh = _WorkerActorMesh()
        rollouter = _rollouter_without_datasets()
        await _setup(rollouter, actor_mesh, monkeypatch)
        turn = _TurnWithTopkLogprobs(
            rollout_id=RolloutTurnID(group_id=1, rollout_id=0, turn_id=0),
            prompt_prefix_len=0,
            prompt_delta_token_ids=np.zeros(3, dtype=np.int32),  # 12 bytes
            prompt_delta_images=(
                PromptImage(
                    hash_value="h",
                    source_path="a.png",
                    grid_thw=(1, 2, 2),
                    data=b"x" * 100,
                ),
            ),
            prompt_delta_image_placeholders=(PlaceholderRange(offset=0, length=1),),
            completion_token_ids=np.zeros(2, dtype=np.int32),  # 8 bytes
            completion_logprobs=np.zeros(2, dtype=np.float32),  # 8 bytes
            completion_topk_logprobs=np.zeros((2, 4), dtype=np.float32),  # 32 bytes
        )
        actor_mesh.run_group.rollouts = [
            Rollout(
                group_id=1, rollout_id=0, turns=[turn], status=RolloutStatus.COMPLETED
            )
        ]
        dispatch = asyncio.create_task(
            rollouter.run_group_rollouts(
                generate_fn="generate_fn",
                release_session_fn="release_session_fn",
                sample="sample",
                group_id=1,
                group_size=1,
                sampling="sampling",
            )
        )
        await actor_mesh.run_group.started.wait()
        # The reply takes 50 ms to arrive.
        await asyncio.sleep(0.05)
        actor_mesh.run_group.release.set()
        group = await dispatch
        await rollouter.close()
        return m.MetricsProcessor._aggregate_metrics(group.metrics)

    metrics = asyncio.run(run())

    assert metrics["timing/hop/run_group/call_duration/mean"] >= 0.05
    assert metrics["timing/hop/run_group/call_duration/max"] == pytest.approx(
        metrics["timing/hop/run_group/call_duration/mean"]
    )
    # Every array field counts, including ones RolloutTurn itself does not declare.
    assert metrics["payload/rollout_group_array_bytes/mean"] == 12 + 100 + 8 + 8 + 32


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
