# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import contextlib
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call, create_autospec, MagicMock, patch, sentinel

import pytest
import torchstore as ts
from monarch.actor import this_proc

from torchtitan.config import OverrideConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.rl import train
from torchtitan.rl.actors.monarch import provisioning, system as monarch_system
from torchtitan.rl.actors.monarch.generator import GeneratorActor
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.actors.monarch.trainer import TrainerActor
from torchtitan.rl.actors.system import ServiceHandle, TrainerHandle
from torchtitan.rl.controller import controller as controller_module
from torchtitan.rl.controller.controller import AsyncLoopConfig
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout.recorder import RolloutSampleRecorder
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter

from tests.rl.unit_tests.cpu.virtual_clock import VirtualClockLoop


class _FakeController:
    instances = []

    def __init__(self, config=None):
        self.config = config
        self.events = []
        self.actor_system = None
        self.instances.append(self)

    async def setup_async(self):
        self.events.append("setup")
        if getattr(self.config, "fail_setup", False):
            raise RuntimeError("setup failed")

    async def run(self):
        self.events.append("train")
        if getattr(self.config, "fail_train", False):
            raise RuntimeError("train failed")
        if getattr(self.config, "cancel_train", False):
            raise asyncio.CancelledError()

    async def close(self):
        self.events.append("close")


class _FakeDebug:
    enable_structured_logging = False


class _FakeTrainerConfig:
    debug = _FakeDebug()


class _FakeConfig:
    """Fake config whose build() returns a _FakeController."""

    dump_folder = "/tmp/test_rl"
    trainer = _FakeTrainerConfig()
    # main() also reads config.generator.parallelism (same stubbing applies).
    generator = SimpleNamespace(parallelism=None)
    num_generators = 1

    @property
    def __class__(self):
        # main() does `assert isinstance(config, Controller.Config)`. Reporting
        # that type lets this lightweight stand-in pass the check (the
        # unittest.mock `spec` idiom) without constructing a real Config.
        return train.Controller.Config

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def build(self):
        return _FakeController(config=self)


class _FakeConfigLoader:
    config = _FakeConfig()

    def load(self):
        return self.config


def test_async_loop_config_derives_window_and_max_offpolicy_steps() -> None:
    default_loop = AsyncLoopConfig()
    assert default_loop.target_offpolicy_steps == 3
    assert default_loop.windowed_fifo_batches is None

    # P=8, S=3: windowed_fifo_batches=1 -> 8 ids, cap 4; 3 -> 24 ids, cap 6; None -> no window, no cap
    one_batch = AsyncLoopConfig(num_prompts_per_train_step=8, windowed_fifo_batches=1)
    assert (one_batch.window_size, one_batch.max_offpolicy_steps) == (8, 4)
    three_batches = AsyncLoopConfig(
        num_prompts_per_train_step=8, windowed_fifo_batches=3
    )
    assert (three_batches.window_size, three_batches.max_offpolicy_steps) == (24, 6)
    uncapped = AsyncLoopConfig(num_prompts_per_train_step=8, windowed_fifo_batches=None)
    assert (uncapped.window_size, uncapped.max_offpolicy_steps) == (None, None)

    # a window larger than the buffer is allowed: P=8, S=1 (16 slots), 5 batches -> 40 ids, cap 6
    wide = AsyncLoopConfig(
        num_prompts_per_train_step=8, target_offpolicy_steps=1, windowed_fifo_batches=5
    )
    assert (wide.window_size, wide.max_offpolicy_steps) == (40, 6)


def test_async_loop_config_rejects_bad_windowed_fifo_batches() -> None:
    with pytest.raises(ValueError, match="windowed_fifo_batches"):
        AsyncLoopConfig(windowed_fifo_batches=0)
    with pytest.raises(TypeError, match="window_fraction"):
        AsyncLoopConfig(window_fraction=0.3)  # type: ignore[call-arg]


def _make_stub_rl_trainer():
    """Create an Controller with a minimal stub config (no Generator validation)."""
    from torchtitan.rl.observability import metrics as m

    class _StubRollouter:
        async def close(self):
            pass

    class _StubConfig:
        async_loop = AsyncLoopConfig()
        metrics = m.MetricsProcessor.Config()
        dump_folder = "/tmp/test_rl"
        rollout_recorder = RolloutSampleRecorder.Config()
        hf_assets_path = "./tests/assets/tokenizer"
        # __init__ builds these too; stub them so construction does no real work.
        tokenizer = SimpleNamespace(
            build=lambda *, tokenizer_path: SimpleNamespace(eos_id=0)
        )
        renderer = SimpleNamespace(
            build=lambda *, tokenizer: SimpleNamespace(get_stop_token_ids=lambda: [])
        )
        # __init__ copies each role's model, reads generator.sampling (a dataclass,
        # for replace) + generator.debug.seed, and applies the (empty) override.
        trainer = SimpleNamespace(model=None)
        generator = SimpleNamespace(
            model=None, sampling=SamplingConfig(), debug=SimpleNamespace(seed=None)
        )
        override = OverrideConfig()
        rollouter = SimpleNamespace(build=lambda: _StubRollouter())

        def to_dict(self):
            return {}

        def maybe_log(self) -> None:
            pass

    return train.Controller(_StubConfig())


def test_main_shuts_down_after_success(monkeypatch):
    _FakeConfigLoader.config = _FakeConfig()
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigLoader", _FakeConfigLoader)

    asyncio.run(train.main())

    trainer = _FakeController.instances[0]
    assert trainer.events == ["setup", "train", "close"]


def test_main_shuts_down_after_train_failure(monkeypatch):
    _FakeConfigLoader.config = _FakeConfig(fail_train=True)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigLoader", _FakeConfigLoader)

    with pytest.raises(RuntimeError, match="train failed"):
        asyncio.run(train.main())

    assert _FakeController.instances[0].events == ["setup", "train", "close"]


def test_main_shuts_down_after_setup_failure(monkeypatch):
    _FakeConfigLoader.config = _FakeConfig(fail_setup=True)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigLoader", _FakeConfigLoader)

    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(train.main())

    assert _FakeController.instances[0].events == ["setup", "close"]


def test_rl_trainer_shutdown_is_noop_before_meshes_spawn():
    trainer = _make_stub_rl_trainer()

    asyncio.run(trainer.close())

    assert trainer.trainer is None
    assert trainer.generator_router is None
    assert trainer.actor_system is None


def test_main_reraises_cancellation_after_shutdown(monkeypatch):
    """Signal-driven cancellation surfaces as ``CancelledError`` from the
    running task. Monarch delivers unhandled actor faults the same way, so
    ``main`` runs ``close`` and re-raises to exit nonzero."""
    _FakeConfigLoader.config = _FakeConfig(cancel_train=True)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigLoader", _FakeConfigLoader)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(train.main())

    assert _FakeController.instances[0].events == ["setup", "train", "close"]


def test_vllm_generator_does_not_touch_cuda_from_finalizer():
    assert "__del__" not in Generator.__dict__


class _StubEndpoint:
    def __init__(self, name, events, raises=False):
        self._name = name
        self._events = events
        # True raises a RuntimeError, and an exception raises itself.
        self._raises = RuntimeError(f"{name} failed") if raises is True else raises

    async def call(self):
        self._events.append(self._name)
        if self._raises:
            raise self._raises


class _RouterCloseEndpoint:
    def __init__(self, router):
        self._router = router

    async def call_one(self):
        return await self._router._fanout("close", return_exceptions=True)


class _StubRouterHandle:
    """Stands in for the router's actor-mesh handle, whose endpoints take adverbs."""

    def __init__(self, router):
        self.close_generators = _RouterCloseEndpoint(router)


class _StubActor:
    def __init__(self, name, events, raises=False):
        self.close = _StubEndpoint(name, events, raises)


def _service(actor) -> ServiceHandle:
    """A single-rank service: its all-ranks and rank-0 handles are the same actor."""
    return ServiceHandle(all_ranks=actor, rank0=actor)


def _trainer(actor) -> TrainerHandle:
    """A single-rank trainer: every handle is the same actor."""
    return TrainerHandle(all_ranks=actor, rank0=actor, by_dp_rank=[actor])


def _actor_system(*mesh_names, events, batcher_mesh_name=None) -> MonarchActorSystem:
    """A Monarch actor system holding stub meshes: the router's, the trainer's, then
    one per generator; and the batcher's if named."""
    actor_system = MonarchActorSystem(MonarchActorSystem.Config())
    meshes = [_StubMesh(name, events) for name in mesh_names]
    actor_system._router_mesh, actor_system._trainer_mesh = meshes[0], meshes[1]
    actor_system._generator_meshes = meshes[2:]
    if batcher_mesh_name is not None:
        actor_system._batcher_mesh = _StubMesh(batcher_mesh_name, events)
    return actor_system


class _StubMesh:
    def __init__(self, name, events, error=None):
        self._name = name
        self._events = events
        self._error = error

    async def stop(self):
        # Suspends once, as ProcMesh.stop does, which is where a pending cancel lands.
        await asyncio.sleep(0)
        self._events.append(self._name)
        if self._error is not None:
            raise self._error


def _set_generator_router(rl_trainer, generators):
    rl_trainer.generator_router = _StubRouterHandle(
        InterGeneratorRouter(
            InterGeneratorRouter.Config(),
            generators=[_service(generator) for generator in generators],
        )
    )


def _with_rollout_worker_pool(rl_trainer, events) -> None:
    """Gives the actor system a rollout worker pool, which the rollouter's close stops,
    as Rollouter.close does."""
    rl_trainer.actor_system._rollout_worker_mesh = _StubMesh(
        "mesh.stop[rollout_workers]", events
    )
    rl_trainer._rollouter = SimpleNamespace(
        close=rl_trainer.actor_system.stop_rollout_workers
    )


def test_shutdown_calls_actor_close_before_mesh_stop():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _trainer(_StubActor("trainer.close", events))
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[router]",
        "mesh.stop[trainer]",
        "mesh.stop[generator]",
        events=events,
        batcher_mesh_name="mesh.stop[batcher]",
    )
    _with_rollout_worker_pool(rl_trainer, events)

    asyncio.run(rl_trainer.close())

    # The rollouter's close stopped the pool. So the actor system's stop does not.
    assert events == [
        "trainer.close",
        "mesh.stop[rollout_workers]",
        "generator.close",
        "mesh.stop[router]",
        "mesh.stop[batcher]",
        "mesh.stop[trainer]",
        "mesh.stop[generator]",
    ]
    # A second stop has nothing left to stop.
    asyncio.run(rl_trainer.actor_system.stop())
    assert len(events) == 7


def test_shutdown_closes_all_generators():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _trainer(_StubActor("trainer.close", events))
    _set_generator_router(
        rl_trainer,
        [
            _StubActor("generator[0].close", events),
            _StubActor("generator[1].close", events),
        ],
    )
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[0]", "mesh.stop[1]", events=events
    )

    asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "generator[0].close",
        "generator[1].close",
        "mesh.stop[0]",
        "mesh.stop[1]",
    ]


def test_shutdown_continues_after_actor_close_failure():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _trainer(_StubActor("trainer.close", events, raises=True))
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[0]", "mesh.stop[1]", events=events
    )

    asyncio.run(rl_trainer.close())

    # trainer.close raised, but every later step still ran.
    assert events == [
        "trainer.close",
        "generator.close",
        "mesh.stop[0]",
        "mesh.stop[1]",
    ]


def test_shutdown_continues_after_generator_close_failure():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _trainer(_StubActor("trainer.close", events))
    _set_generator_router(
        rl_trainer,
        [
            _StubActor("generator[0].close", events, raises=True),
            _StubActor("generator[1].close", events),
        ],
    )
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[0]", "mesh.stop[1]", events=events
    )

    asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "generator[0].close",
        "generator[1].close",
        "mesh.stop[0]",
        "mesh.stop[1]",
    ]


def _cancelled_error(coro: Coroutine[Any, Any, object]) -> asyncio.CancelledError:
    """Runs ``coro`` and returns the ``CancelledError`` it raises."""
    with pytest.raises(asyncio.CancelledError) as raised:
        asyncio.run(coro)
    return raised.value


def _record_metrics_close(rl_trainer, events) -> None:
    rl_trainer.metrics_processor = SimpleNamespace(
        close=lambda: events.append("metrics.close")
    )


def test_shutdown_stops_the_actor_system_when_close_is_cancelled():
    """``close`` still stops every mesh of the actor system once Monarch's fault hook
    cancels it because a rank froze, including a rollout worker pool whose stop the
    cancelled close never reached, and then closes the metric backends."""
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    cancel = asyncio.CancelledError()
    rl_trainer.trainer = _trainer(_StubActor("trainer.close", events, raises=cancel))
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[router]",
        "mesh.stop[trainer]",
        "mesh.stop[generator]",
        events=events,
        batcher_mesh_name="mesh.stop[batcher]",
    )
    _with_rollout_worker_pool(rl_trainer, events)
    _record_metrics_close(rl_trainer, events)

    assert _cancelled_error(rl_trainer.close()) is cancel

    assert events == [
        "trainer.close",
        "mesh.stop[router]",
        "mesh.stop[batcher]",
        "mesh.stop[trainer]",
        "mesh.stop[generator]",
        "mesh.stop[rollout_workers]",
        "metrics.close",
    ]


def test_shutdown_stops_the_rollout_worker_pool_once_when_its_stop_is_cancelled():
    """A cancel that lands in the pool's stop inside the rollouter's close leaves that
    stop started. So the actor system's stop leaves the pool alone."""
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _trainer(_StubActor("trainer.close", events))
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[router]",
        "mesh.stop[trainer]",
        "mesh.stop[generator]",
        events=events,
        batcher_mesh_name="mesh.stop[batcher]",
    )
    _with_rollout_worker_pool(rl_trainer, events)
    cancel = asyncio.CancelledError()
    rl_trainer.actor_system._rollout_worker_mesh = _StubMesh(
        "mesh.stop[rollout_workers]", events, cancel
    )
    _record_metrics_close(rl_trainer, events)

    assert _cancelled_error(rl_trainer.close()) is cancel

    assert events == [
        "trainer.close",
        "mesh.stop[rollout_workers]",
        "mesh.stop[router]",
        "mesh.stop[batcher]",
        "mesh.stop[trainer]",
        "mesh.stop[generator]",
        "metrics.close",
    ]


def test_shutdown_stops_the_later_meshes_when_a_mesh_stop_is_cancelled():
    """A cancel that lands in a mesh stop still lets the later meshes stop, and the
    metric backends then close."""
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _trainer(_StubActor("trainer.close", events))
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[router]", "mesh.stop[trainer]", "mesh.stop[generator]", events=events
    )
    cancel = asyncio.CancelledError()
    rl_trainer.actor_system._trainer_mesh = _StubMesh(
        "mesh.stop[trainer]", events, cancel
    )
    _record_metrics_close(rl_trainer, events)

    assert _cancelled_error(rl_trainer.close()) is cancel

    assert events == [
        "trainer.close",
        "generator.close",
        "mesh.stop[router]",
        "mesh.stop[trainer]",
        "mesh.stop[generator]",
        "metrics.close",
    ]


@pytest.mark.parametrize("cancelled", [False, True], ids=["closed", "cancelled"])
def test_shutdown_logs_a_failed_metrics_close(
    cancelled: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """A metrics close that raises after the meshes stop is logged with its error. Close
    then returns or raises the cancel that ended it."""
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    cancel = asyncio.CancelledError()
    rl_trainer.trainer = _trainer(
        _StubActor("trainer.close", events, raises=cancel if cancelled else False)
    )
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[router]", "mesh.stop[trainer]", events=events
    )
    error = RuntimeError("W&B finish failed")

    def close_metrics() -> None:
        events.append("metrics.close")
        raise error

    rl_trainer.metrics_processor = SimpleNamespace(close=close_metrics)

    if cancelled:
        assert _cancelled_error(rl_trainer.close()) is cancel
    else:
        asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "mesh.stop[router]",
        "mesh.stop[trainer]",
        "metrics.close",
    ]
    logged = [
        record.exc_info[1]
        for record in caplog.records
        if record.name == controller_module.logger.name and record.exc_info is not None
    ]
    assert logged == [error]


def test_shutdown_stops_every_mesh_when_a_cancel_lands_in_the_metrics_close():
    """A cancel that the fault hook requests while close runs the synchronous metrics
    close, such as W&B's finish, lands only once every mesh has stopped."""
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _trainer(_StubActor("trainer.close", events))
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[router]", "mesh.stop[trainer]", "mesh.stop[generator]", events=events
    )

    def close_metrics() -> None:
        events.append("metrics.close")
        # As asyncio.run's SIGINT handler does with the fault hook's interrupt.
        asyncio.current_task().cancel("metrics close")

    rl_trainer.metrics_processor = SimpleNamespace(close=close_metrics)

    with pytest.raises(asyncio.CancelledError, match="metrics close"):
        asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "generator.close",
        "mesh.stop[router]",
        "mesh.stop[trainer]",
        "mesh.stop[generator]",
        "metrics.close",
    ]


class _FakeHostMesh:
    """Records the CUDA_VISIBLE_DEVICES of each proc mesh spawned on it."""

    def __init__(self, num_hosts=1):
        self.num_hosts = num_hosts
        self.visible_devices = []

    def __len__(self):
        return self.num_hosts

    def spawn_procs(self, *, per_host, bootstrap, bootstrap_command):
        self.visible_devices.append(bootstrap_command["CUDA_VISIBLE_DEVICES"])


@pytest.fixture
def bootstrap_cmd_returns_env(monkeypatch):
    """`default_bootstrap_cmd().with_env(env)` returns `env`, so `_FakeHostMesh` can read it."""
    monkeypatch.setattr(
        provisioning,
        "default_bootstrap_cmd",
        lambda: SimpleNamespace(with_env=lambda env: env),
    )


def test_spawn_proc_mesh_gives_roles_sharing_a_host_mesh_disjoint_gpus(
    bootstrap_cmd_returns_env,
):
    """Roles on one host mesh get disjoint GPUs; another host mesh starts at GPU 0."""
    mesh_a, mesh_b = _FakeHostMesh(), _FakeHostMesh()

    provisioning.spawn_proc_mesh(
        trainer_world_size=2,
        per_generator_world_size=1,
        host_meshes=provisioning.HostMeshes(
            trainer=mesh_a,
            generators=[mesh_a, mesh_a, mesh_b, mesh_b, mesh_b, mesh_b],
            gpus_per_node=4,
        ),
        num_generators=6,
    )

    assert mesh_a.visible_devices == ["0,1", "2", "3"]
    assert mesh_b.visible_devices == ["0", "1", "2", "3"]


@pytest.mark.parametrize(
    "num_hosts, layout, match",
    [
        (1, "AAABBB", "need 5 GPUs per host, but gpus_per_node is 4"),
        (2, "AABBBB", r"generator0 world size \(1\) .* host count \(2\)"),
    ],
)
def test_spawn_proc_mesh_checks_gpus_per_host_before_spawning(
    bootstrap_cmd_returns_env, num_hosts, layout, match
):
    """Roles that don't fit their host mesh raise before any proc mesh is spawned."""
    meshes = {"A": _FakeHostMesh(num_hosts), "B": _FakeHostMesh(num_hosts)}

    with pytest.raises(ValueError, match=match):
        provisioning.spawn_proc_mesh(
            trainer_world_size=2,
            per_generator_world_size=1,
            host_meshes=provisioning.HostMeshes(
                trainer=meshes["A"],
                generators=[meshes[name] for name in layout],
                gpus_per_node=4,
            ),
            num_generators=len(layout),
        )

    assert all(mesh.visible_devices == [] for mesh in meshes.values())


def test_monarch_actor_system_provisions_every_role_before_spawning(monkeypatch):
    """`provision` spawns the GPU meshes on the launcher's hosts, then the router's and
    the batcher's procs, then sets up torch elastic on each GPU mesh, before any actor
    is spawned."""
    events: list[str] = []

    def spawn_proc_mesh(trainer_world_size, per_generator_world_size, **kwargs):
        events.append(
            f"gpu meshes {trainer_world_size}+"
            f"{kwargs['num_generators']}x{per_generator_world_size} "
            f"on {kwargs['host_meshes']} with {kwargs['generator_env']}"
        )
        return "trainer_mesh", ["generator_mesh_0", "generator_mesh_1"]

    async def setup_torch_elastic_env(mesh):
        events.append(f"torch elastic {mesh}")

    monkeypatch.setattr(monarch_system, "spawn_proc_mesh", spawn_proc_mesh)
    monkeypatch.setattr(
        monarch_system,
        "spawn_cpu_procs",
        lambda num_procs: events.append(f"cpu procs {num_procs}") or "cpu_mesh",
    )
    monkeypatch.setattr(
        monarch_system, "setup_torch_elastic_env", setup_torch_elastic_env
    )
    actor_system = MonarchActorSystem.Config().build(
        host_meshes="host_meshes", generator_env={"KEY": "value"}
    )

    asyncio.run(
        actor_system.provision(
            trainer_parallelism=ParallelismConfig(data_parallel_shard_degree=2),
            generator_parallelism=InferenceParallelismConfig(tensor_parallel_degree=2),
            num_generators=2,
        )
    )

    assert events == [
        "gpu meshes 2+2x2 on host_meshes with {'KEY': 'value'}",
        "cpu procs 1",
        "cpu procs 1",
        "torch elastic trainer_mesh",
        "torch elastic generator_mesh_0",
        "torch elastic generator_mesh_1",
    ]


class _FakeTrainerRanks:
    """A trainer actor mesh of ranks laid out as (pp, dp, cp_tp); ``get_dp_rank``
    reports ``reported_dp_ranks`` for those ranks."""

    def __init__(self, reported_dp_ranks: list[int], ranks: list[int] | None = None):
        self.reported_dp_ranks = reported_dp_ranks
        self.ranks = list(range(len(reported_dp_ranks))) if ranks is None else ranks
        self.get_dp_rank = SimpleNamespace(call=self._get_dp_rank)

    async def _get_dp_rank(self):
        return {rank: self.reported_dp_ranks[rank] for rank in self.ranks}

    def flatten(self, name):
        return self

    def split(self, *, rank, pp, dp):
        self.pp_degree = pp
        self.dp_degree = dp
        return self

    def slice(self, *, rank=None, dp=None):
        if rank is not None:
            return _FakeTrainerRanks(self.reported_dp_ranks, [self.ranks[rank]])
        num_ranks_per_stage = len(self.ranks) // self.pp_degree
        num_ranks_per_dp_rank = num_ranks_per_stage // self.dp_degree
        ranks_at_dp_rank = []
        for stage in range(self.pp_degree):
            start = stage * num_ranks_per_stage + dp * num_ranks_per_dp_rank
            ranks_at_dp_rank.extend(self.ranks[start : start + num_ranks_per_dp_rank])
        return _FakeTrainerRanks(self.reported_dp_ranks, ranks_at_dp_rank)


def _spawn_trainer(
    trainer_ranks: _FakeTrainerRanks, *, dp_degree: int, pp_degree: int = 1
):
    actor_system = MonarchActorSystem(MonarchActorSystem.Config())
    actor_system._trainer_mesh = SimpleNamespace(
        spawn=lambda name, actor_cls, config, **kwargs: trainer_ranks
    )
    trainer_config = SimpleNamespace(
        parallelism=SimpleNamespace(pipeline_parallel_degree=pp_degree)
    )
    return asyncio.run(actor_system.spawn_trainer(trainer_config, dp_degree=dp_degree))


def test_monarch_actor_system_groups_trainer_ranks_by_dp_rank():
    """DP degree 2, cp * tp = 2: ranks 0, 1 are DP rank 0 and ranks 2, 3 DP rank 1."""
    trainer = _spawn_trainer(_FakeTrainerRanks([0, 0, 1, 1]), dp_degree=2)

    assert [handle.ranks for handle in trainer.by_dp_rank] == [[0, 1], [2, 3]]
    assert trainer.rank0.ranks == [0]


def test_monarch_actor_system_spans_pipeline_stages_per_dp_rank():
    """PP degree 2, DP degree 2, cp * tp = 1: each pipeline stage holds one rank
    per DP rank, so DP rank 0 is ranks 0 and 2."""
    trainer = _spawn_trainer(_FakeTrainerRanks([0, 1, 0, 1]), dp_degree=2, pp_degree=2)

    assert [handle.ranks for handle in trainer.by_dp_rank] == [[0, 2], [1, 3]]


def test_monarch_actor_system_rejects_a_trainer_not_laid_out_as_dp_cp_tp():
    with pytest.raises(RuntimeError, match="DP rank 0 report DP ranks"):
        _spawn_trainer(_FakeTrainerRanks([0, 1, 0, 1]), dp_degree=2)


# A typo, wrong capitalization, a non-string, and a ControlPlane instance, which
# TorchStore would take as the caller's own plane. Each is built in the test so this
# module loads with a TorchStore that has no ControlPlane.
@pytest.mark.parametrize(
    "make_control_plane",
    [
        lambda: "monarc",
        lambda: "Monarch",
        lambda: None,
        lambda: MagicMock(spec=ts.ControlPlane),
    ],
    ids=["typo", "capitalized", "none", "control_plane_instance"],
)
# build() checks again since recipes may assign a config field after construction.
@pytest.mark.parametrize("checked_by", ["construction", "build"])
def test_monarch_actor_system_config_rejects_other_torchstore_control_planes(
    make_control_plane: Callable[[], object], checked_by: str
) -> None:
    control_plane = make_control_plane()
    error = re.escape(
        "torchstore_control_plane must be one of: ['monarch', 'service'] "
        f"(got {control_plane!r})"
    )
    if checked_by == "construction":
        check = lambda: MonarchActorSystem.Config(  # noqa: E731
            torchstore_control_plane=control_plane  # type: ignore[arg-type]
        )
    else:
        config = MonarchActorSystem.Config()
        config.torchstore_control_plane = control_plane  # type: ignore[assignment]
        check = config.build
    with pytest.raises(ValueError, match=error):
        check()


# Test IDs for the client setups _actor_system_with_clients returns in order.
_CLIENTS = ["trainer", "generator_0", "generator_1", "generator_2"]


def _actor_system_with_clients(
    config: MonarchActorSystem.Config | None = None,
) -> tuple[MonarchActorSystem, list[AsyncMock]]:
    """A Monarch actor system built from ``config`` with a trainer and 3 generators, and
    the mocks of their TorchStore client setup: the trainer's, then each generator's."""
    actor_system = MonarchActorSystem(config or MonarchActorSystem.Config())
    actor_system._trainer_mesh = sentinel.trainer_mesh
    actor_system._trainer = MagicMock()
    actor_system._generators = [
        ServiceHandle(all_ranks=MagicMock(), rank0=MagicMock()) for _ in range(3)
    ]
    actors = [actor_system._trainer, *(g.all_ranks for g in actor_system._generators)]
    for actor in actors:
        actor.initialize_torchstore_client.call = AsyncMock()
    return actor_system, [actor.initialize_torchstore_client.call for actor in actors]


def _patch_torchstore(address: object, shutdown: Callable[[], None]):
    """Patches the actor system's TorchStore. Its ``ts.initialize`` returns ``address``
    on "service" and None on "monarch" as TorchStore's does."""
    torchstore = create_autospec(ts)
    torchstore.initialize.side_effect = lambda **kwargs: (
        None if kwargs["control_plane"] == "monarch" else address
    )
    torchstore.shutdown.side_effect = shutdown
    return patch("torchtitan.rl.actors.monarch.system.ts", torchstore)


# A hard-coded control plane fails one of the two.
@pytest.mark.parametrize(
    "control_plane, address",
    [("monarch", None), ("service", sentinel.address)],
    ids=["monarch", "service"],
)
def test_monarch_actor_system_passes_its_control_plane_and_the_address(
    control_plane: str, address: object
) -> None:
    """``ts.initialize`` gets the configured control plane on the trainer mesh, and what
    it returns reaches the trainer's and every generator's client setup."""
    actor_system, clients = _actor_system_with_clients(
        MonarchActorSystem.Config(torchstore_control_plane=control_plane)
    )
    with patch("torchtitan.rl.actors.monarch.system.ts", autospec=True) as torchstore:
        torchstore.initialize.return_value = address
        asyncio.run(actor_system.initialize_torchstore())

    torchstore.initialize.assert_awaited_once_with(
        mesh=sentinel.trainer_mesh,
        strategy=torchstore.TorchStoreStrategy.return_value,
        client_type=torchstore.ClientType.ROUTING,
        control_plane=control_plane,
    )
    torchstore.TorchStoreStrategy.assert_called_once_with()
    clients[0].assert_awaited_once_with(address=address)
    for requester_index, client in enumerate(clients[1:]):
        client.assert_awaited_once_with(
            requester_index=requester_index, address=address
        )


@pytest.mark.parametrize("held", range(len(_CLIENTS)), ids=_CLIENTS)
def test_monarch_actor_system_waits_for_every_client_setup(held: int) -> None:
    """``initialize_torchstore`` returns only once every client setup has finished. The
    role-less state-dict APIs then use each rank's client."""
    actor_system, clients = _actor_system_with_clients()

    async def run() -> None:
        release = asyncio.Event()

        async def held_setup(**kwargs: object) -> None:
            await release.wait()

        clients[held].side_effect = held_setup
        setup = asyncio.create_task(actor_system.initialize_torchstore())
        # Every shorter timer fires first, each once no callback is left to run. So a
        # call that returns early, at once or after a shorter wait, has returned by now.
        await asyncio.sleep(3600)
        # An early failure raises here as itself rather than as the check below.
        if setup.done():
            setup.result()
        assert not setup.done()
        release.set()
        # A setup left with nothing to run fails here instead of spinning the loop.
        await asyncio.wait_for(setup, timeout=3600)

    with patch("torchtitan.rl.actors.monarch.system.ts", autospec=True):
        with asyncio.Runner(loop_factory=VirtualClockLoop) as runner:
            runner.run(run())
    for client in clients:
        client.assert_awaited_once()


def test_monarch_actor_system_sets_up_no_client_when_torchstore_fails() -> None:
    actor_system, clients = _actor_system_with_clients()
    with patch("torchtitan.rl.actors.monarch.system.ts", autospec=True) as torchstore:
        torchstore.initialize.side_effect = RuntimeError("ts.initialize failed")
        with pytest.raises(RuntimeError, match="ts.initialize failed"):
            asyncio.run(actor_system.initialize_torchstore())
    for client in clients:
        client.assert_not_called()


@pytest.mark.parametrize("failing", range(len(_CLIENTS)), ids=_CLIENTS)
def test_monarch_actor_system_raises_a_failed_client_setup(failing: int) -> None:
    """Its error raises, and ``stop`` still stops every mesh and shuts down the service
    it started."""
    calls: list[str] = []
    actor_system, clients = _actor_system_with_clients(
        MonarchActorSystem.Config(torchstore_control_plane="service")
    )
    clients[failing].side_effect = RuntimeError(f"{_CLIENTS[failing]} client failed")
    mesh_stops = _stub_meshes(actor_system, calls)
    with _patch_torchstore(sentinel.address, lambda: calls.append("ts.shutdown")):
        with pytest.raises(RuntimeError, match=f"{_CLIENTS[failing]} client failed"):
            asyncio.run(actor_system.initialize_torchstore())
        asyncio.run(actor_system.stop())
    assert calls == mesh_stops + ["ts.shutdown"]


class _StopError(Exception):
    """Neither a RuntimeError nor an OSError, as Monarch's ActorError is not."""


def _stub_meshes(
    actor_system: MonarchActorSystem,
    calls: list[str],
    errors: dict[str, BaseException] | None = None,
) -> list[str]:
    """Gives ``actor_system`` a stub mesh per role, a rollout worker pool that a
    cancelled close left included, each raising its entry of ``errors``. Returns the
    stops they record in ``calls``, in the order ``stop`` must reach them."""
    names = "router batcher trainer generator_0 generator_1 rollout_workers".split()
    router, batcher, trainer, *generators, pool = (
        _StubMesh(f"mesh.stop[{name}]", calls, (errors or {}).get(name))
        for name in names
    )
    actor_system._router_mesh, actor_system._batcher_mesh = router, batcher
    actor_system._trainer_mesh, actor_system._generator_meshes = trainer, generators
    actor_system._rollout_worker_mesh = pool
    return [f"mesh.stop[{name}]" for name in names]


@pytest.mark.parametrize(
    "control_plane, address",
    [("monarch", None), ("service", sentinel.address)],
    ids=["monarch", "service"],
)
def test_monarch_actor_system_stop_shuts_down_the_torchstore_service_it_started(
    control_plane: str, address: object
) -> None:
    calls: list[str] = []
    actor_system, _ = _actor_system_with_clients(
        MonarchActorSystem.Config(torchstore_control_plane=control_plane)
    )
    with _patch_torchstore(address, lambda: calls.append("ts.shutdown")):
        asyncio.run(actor_system.initialize_torchstore())
        mesh_stops = _stub_meshes(actor_system, calls)
        asyncio.run(actor_system.stop())
        # A second stop finds nothing left to stop.
        asyncio.run(actor_system.stop())
    # Monarch's control plane returns no address and has no service to shut down.
    assert calls == mesh_stops + ([] if address is None else ["ts.shutdown"])


def test_monarch_actor_system_stop_logs_each_failure_and_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []
    actor_system, _ = _actor_system_with_clients(
        MonarchActorSystem.Config(torchstore_control_plane="service")
    )

    def shutdown() -> None:
        calls.append("ts.shutdown")
        raise _StopError("boom")

    with _patch_torchstore(sentinel.address, shutdown):
        asyncio.run(actor_system.initialize_torchstore())
        mesh_stops = _stub_meshes(
            actor_system, calls, {"trainer": _StopError("trainer mesh")}
        )
        asyncio.run(actor_system.stop())
    assert calls == mesh_stops + ["ts.shutdown"]
    # Each failure is logged with its error, not only a message.
    logged = {
        record.getMessage(): str(record.exc_info[1])
        for record in caplog.records
        if record.exc_info is not None and record.name == monarch_system.logger.name
    }
    assert logged == {
        "mesh.stop[2] failed": "trainer mesh",
        "ts.shutdown failed": "boom",
    }


def _cancel_stop(
    actor_system: MonarchActorSystem, when_to_cancel: Callable[[], Awaitable[object]]
) -> bool:
    """Starts ``actor_system.stop()``, cancels it once ``when_to_cancel`` returns and
    reports whether that cancel raised from it, not a return or another cancel."""

    async def run() -> bool:
        stop = asyncio.create_task(actor_system.stop())
        # The clock jumps to each bound once nothing else can run. Unlike wait_for on
        # Python 3.11, asyncio.timeout adds no loop pass before the cancel.
        async with asyncio.timeout(3600):
            await when_to_cancel()
        stop.cancel("cancel_stop")
        await asyncio.wait([stop], timeout=3600)
        try:
            stop.result()
        except asyncio.CancelledError as error:
            return error.args == ("cancel_stop",)
        return False

    with asyncio.Runner(loop_factory=VirtualClockLoop) as runner:
        return runner.run(run())


def test_monarch_actor_system_stop_shuts_down_the_service_when_cancelled() -> None:
    """A cancel that lands while a mesh stop is suspended ends stop's wait for it and
    raises only once the later meshes have stopped and the service has shut down."""
    calls: list[str] = []
    actor_system, _ = _actor_system_with_clients(
        MonarchActorSystem.Config(torchstore_control_plane="service")
    )
    with _patch_torchstore(sentinel.address, lambda: calls.append("ts.shutdown")):
        asyncio.run(actor_system.initialize_torchstore())
        mesh_stops = _stub_meshes(actor_system, calls)
        # One pass of the loop takes stop to the router mesh's suspension.
        assert _cancel_stop(actor_system, lambda: asyncio.sleep(0))
    assert calls == mesh_stops[1:] + ["ts.shutdown"]


def test_monarch_actor_system_stop_raises_a_cancel_in_the_service_shutdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cancel that lands while ``ts.shutdown`` waits raises from ``stop`` instead of
    being logged as a failed shutdown."""
    calls: list[str] = []
    actor_system, _ = _actor_system_with_clients(
        MonarchActorSystem.Config(torchstore_control_plane="service")
    )
    shutting_down = asyncio.Event()

    async def shutdown() -> None:
        calls.append("ts.shutdown")
        shutting_down.set()
        await asyncio.Event().wait()

    with (
        _patch_torchstore(sentinel.address, shutdown),
        caplog.at_level(logging.ERROR, logger=monarch_system.logger.name),
    ):
        asyncio.run(actor_system.initialize_torchstore())
        mesh_stops = _stub_meshes(actor_system, calls)
        assert _cancel_stop(actor_system, shutting_down.wait)
    assert calls == mesh_stops + ["ts.shutdown"]
    assert [r for r in caplog.records if r.name == monarch_system.logger.name] == []


def test_monarch_actor_system_stop_ends_the_mesh_stops_at_a_second_cancel() -> None:
    """A second cancel ends the mesh stops and thus the wait on a stop that hangs. The
    service still shuts down."""
    calls: list[str] = []
    actor_system, _ = _actor_system_with_clients(
        MonarchActorSystem.Config(torchstore_control_plane="service")
    )
    with _patch_torchstore(sentinel.address, lambda: calls.append("ts.shutdown")):
        asyncio.run(actor_system.initialize_torchstore())
        second = asyncio.CancelledError()
        mesh_stops = _stub_meshes(
            actor_system,
            calls,
            {"batcher": asyncio.CancelledError(), "generator_0": second},
        )
        assert _cancelled_error(actor_system.stop()) is second
    assert calls == mesh_stops[:4] + ["ts.shutdown"]


class _BareTrainerActor(TrainerActor):
    """TrainerActor without the trainer's setup, for its stateless endpoints."""

    def __init__(self) -> None:
        pass


class _BareGeneratorActor(GeneratorActor):
    """GeneratorActor without the generator's setup, for its stateless endpoints."""

    def __init__(self) -> None:
        pass


# Bounds an actor exchange and each actor's stop. A hang that only awaits fails just its
# test, while one that blocks an actor's thread fails the later bare-actor tests that
# call an actor too.
_ACTOR_TIMEOUT_S = 60


@contextlib.asynccontextmanager
async def _bare_actors(*actor_classes: type) -> AsyncIterator[list[Any]]:
    """Spawns ``actor_classes`` in this process where patches reach them and on exit
    waits up to its bound for each actor's stop even after an earlier one failed or
    timed out. The body's failure or else the first stop failure raises, and later
    ones are logged."""
    actors: list[Any] = []
    body_failed = False
    try:
        for index, actor_class in enumerate(actor_classes):
            actors.append(this_proc().spawn(f"bare_{index}", actor_class))
        yield actors
    except BaseException:
        body_failed = True
        raise
    finally:
        stop_error: BaseException | None = None
        for actor in actors:
            try:
                await asyncio.wait_for(actor.stop(), timeout=_ACTOR_TIMEOUT_S)
            except BaseException as error:
                if body_failed or stop_error is not None:
                    logging.getLogger(__name__).exception("stopping %s failed", actor)
                else:
                    stop_error = error
        if stop_error is not None:
            raise stop_error


@pytest.mark.parametrize(
    "body_fails", [False, True], ids=["body_passed", "body_failed"]
)
def test_bare_actors_stops_each_actor_within_its_bound(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, body_fails: bool
) -> None:
    """A hung stop ends at the bound and a failed one leaves the next actor to stop.
    The body's failure or else the first stop failure raises, and the rest are logged."""
    stopped: list[str] = []
    stop_failure = RuntimeError("stop failed")
    body_failure = RuntimeError("body failed")

    async def hang() -> None:
        await asyncio.Event().wait()

    async def fail() -> None:
        raise stop_failure

    async def finish() -> None:
        stopped.append("finish")

    # Each stand-in actor's stop is the coroutine function it is spawned with.
    monkeypatch.setitem(
        globals(),
        "this_proc",
        lambda: SimpleNamespace(spawn=lambda name, stop: SimpleNamespace(stop=stop)),
    )

    async def run() -> tuple[BaseException, float]:
        with pytest.raises(Exception) as raised:
            # The clock jumps to each bound once nothing else can run.
            async with asyncio.timeout(3600), _bare_actors(hang, fail, finish):
                if body_fails:
                    raise body_failure
        return raised.value, asyncio.get_running_loop().time()

    with (
        caplog.at_level(logging.ERROR, logger=__name__),
        asyncio.Runner(loop_factory=VirtualClockLoop) as runner,
    ):
        error, elapsed = runner.run(run())
    assert elapsed == _ACTOR_TIMEOUT_S
    assert stopped == ["finish"]
    logged = [
        r.exc_info[1] for r in caplog.records if r.name == __name__ and r.exc_info
    ]
    if body_fails:
        assert error is body_failure
        assert [type(e) for e in logged] == [TimeoutError, RuntimeError]
    else:
        assert type(error) is TimeoutError
        assert [type(e) for e in logged] == [RuntimeError]
    assert logged[-1] is stop_failure


def test_monarch_actor_system_gives_every_client_the_torchstore_address() -> None:
    """What ``ts.initialize`` returns reaches ``ts.client`` through the actor system's
    endpoint calls in the trainer actor and in each generator actor under that
    generator's index."""

    async def run() -> None:
        async with _bare_actors(
            _BareTrainerActor, _BareGeneratorActor, _BareGeneratorActor
        ) as (trainer, *generators):
            actor_system = MonarchActorSystem(MonarchActorSystem.Config())
            actor_system._trainer = trainer
            actor_system._generators = [
                ServiceHandle(all_ranks=generator, rank0=generator)
                for generator in generators
            ]
            await actor_system.initialize_torchstore()

    with (
        patch("torchtitan.rl.actors.monarch.system.ts", autospec=True) as system_ts,
        patch("torchtitan.rl.trainer.trainer.ts", autospec=True) as trainer_ts,
        patch("torchtitan.rl.generator.generator.ts", autospec=True) as generator_ts,
    ):
        system_ts.initialize.return_value = sentinel.address
        asyncio.run(asyncio.wait_for(run(), timeout=_ACTOR_TIMEOUT_S))

    trainer_ts.client.assert_awaited_once_with(
        role=ts.RankRole.PUBLISHER, address=sentinel.address
    )
    assert sorted(
        generator_ts.client.await_args_list, key=lambda c: c.kwargs["group"]
    ) == [
        call(role=ts.RankRole.REQUESTER, group=group, address=sentinel.address)
        for group in range(2)
    ]


_TRAINER_MODULE = "torchtitan.rl.trainer.trainer"
_GENERATOR_MODULE = "torchtitan.rl.generator.generator"


@pytest.mark.parametrize(
    "actor_class, role_module, args, kwargs, error",
    [
        (_BareTrainerActor, _TRAINER_MODULE, (), {}, "address"),
        (_BareTrainerActor, _TRAINER_MODULE, (None,), {}, "positional"),
        (_BareGeneratorActor, _GENERATOR_MODULE, (), {"requester_index": 3}, "address"),
        (_BareGeneratorActor, _GENERATOR_MODULE, (3, None), {}, "positional"),
    ],
    ids=["trainer", "trainer_positional", "generator", "generator_positional"],
)
def test_monarch_actor_endpoints_take_the_torchstore_address_by_keyword(
    actor_class: type,
    role_module: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    error: str,
) -> None:
    """An endpoint call that leaves out ``address`` or passes it positionally fails at
    the caller, where Monarch binds an endpoint's arguments."""

    async def run() -> None:
        async with _bare_actors(actor_class) as (actor,):
            await actor.initialize_torchstore_client.call_one(*args, **kwargs)

    # TorchStore is patched so a regression fails on the missing TypeError instead of
    # in a real coordinator lookup.
    with patch(f"{role_module}.ts", autospec=True):
        with pytest.raises(TypeError, match=error):
            asyncio.run(asyncio.wait_for(run(), timeout=_ACTOR_TIMEOUT_S))
