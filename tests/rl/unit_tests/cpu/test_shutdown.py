# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from types import SimpleNamespace

import pytest

from torchtitan.config import OverrideConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.rl import train
from torchtitan.rl._env import RL_ROLE_ENV
from torchtitan.rl.actors.monarch import provisioning, system as monarch_system
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.actors.system import ServiceHandle, TrainerHandle
from torchtitan.rl.controller.controller import AsyncLoopConfig
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout.recorder import RolloutSampleRecorder
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter


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
        self._raises = raises

    async def call(self):
        self._events.append(self._name)
        if self._raises:
            raise RuntimeError(f"{self._name} failed")


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
    def __init__(self, name, events):
        self._name = name
        self._events = events

    async def stop(self):
        self._events.append(self._name)


def _set_generator_router(rl_trainer, generators):
    rl_trainer.generator_router = _StubRouterHandle(
        InterGeneratorRouter(
            InterGeneratorRouter.Config(),
            generators=[_service(generator) for generator in generators],
        )
    )


def test_shutdown_calls_actor_close_before_mesh_stop():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _trainer(_StubActor("trainer.close", events))

    async def close_rollouter():
        events.append("rollouter.close")

    rl_trainer._rollouter = SimpleNamespace(close=close_rollouter)
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer.actor_system = _actor_system(
        "mesh.stop[router]",
        "mesh.stop[trainer]",
        "mesh.stop[generator]",
        events=events,
        batcher_mesh_name="mesh.stop[batcher]",
    )

    asyncio.run(rl_trainer.close())

    # Generators close before the rollouter, so their failed replies reach live workers.
    assert events == [
        "trainer.close",
        "generator.close",
        "rollouter.close",
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


class _FakeHostMesh:
    """Records the CUDA_VISIBLE_DEVICES and role of each proc mesh spawned on it."""

    def __init__(self, num_hosts=1):
        self.num_hosts = num_hosts
        self.visible_devices = []
        self.roles = []

    def __len__(self):
        return self.num_hosts

    def spawn_procs(self, *, per_host, bootstrap, bootstrap_command):
        self.visible_devices.append(bootstrap_command["CUDA_VISIBLE_DEVICES"])
        self.roles.append(bootstrap_command[RL_ROLE_ENV])


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
    assert mesh_a.roles == ["trainer", "generator", "generator"]


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
        lambda num_procs, *, role: events.append(f"cpu procs {num_procs}")
        or "cpu_mesh",
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


def test_controller_defaults_to_the_monarch_actor_system():
    from torchtitan.rl.controller.controller import _default_actor_system_config

    assert isinstance(_default_actor_system_config(), MonarchActorSystem.Config)
