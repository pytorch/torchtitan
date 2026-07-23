# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl import train
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.controller import AsyncLoopConfig
from torchtitan.experiments.rl.rollout_recorder import RolloutSampleRecorder
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)


class _FakeController:
    instances = []

    def __init__(self, config=None):
        self.config = config
        self.events = []
        self.setup_trainer_mesh = None
        self.setup_generator_meshes = None
        self.instances.append(self)

    async def setup_async(self, *, trainer_mesh=None, generator_meshes=None):
        self.events.append("setup")
        self.setup_trainer_mesh = trainer_mesh
        self.setup_generator_meshes = generator_meshes
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
    # main() reads config.trainer.parallelism to size the trainer mesh; the
    # value is irrelevant here because stub_mesh_provisioning stubs
    # _compute_trainer_world_size, so a placeholder is enough to resolve the access.
    parallelism = None


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


class _FakeConfigManager:
    config = _FakeConfig()

    def parse_args(self):
        return self.config


def test_async_loop_config_derives_window_and_max_offpolicy_steps() -> None:
    strict_loop = AsyncLoopConfig(
        num_prompts_per_train_step=3,
        target_offpolicy_steps=2,
    )
    assert strict_loop.window_size == 1
    assert strict_loop.max_offpolicy_steps == 2

    async_loop = AsyncLoopConfig(
        num_prompts_per_train_step=3,
        target_offpolicy_steps=2,
        windowed_fifo_fraction=4 / 9,
    )
    assert async_loop.max_active_rollout_groups == 9
    assert async_loop.window_size == 4
    assert async_loop.max_offpolicy_steps == 3

    assert (
        AsyncLoopConfig(
            num_prompts_per_train_step=3,
            target_offpolicy_steps=2,
            windowed_fifo_fraction=1 / 9,
        ).window_size
        == 1
    )
    assert (
        AsyncLoopConfig(
            num_prompts_per_train_step=3,
            target_offpolicy_steps=2,
            windowed_fifo_fraction=1.0,
        ).window_size
        == 9
    )
    assert (
        AsyncLoopConfig(
            num_prompts_per_train_step=8,
            target_offpolicy_steps=3,
            windowed_fifo_fraction=1.0,
        ).window_size
        == 32
    )


def test_async_loop_config_handles_windowed_fifo_fraction_bounds() -> None:
    with pytest.raises(ValueError, match="windowed_fifo_fraction"):
        AsyncLoopConfig(windowed_fifo_fraction=0)
    with pytest.raises(ValueError, match="windowed_fifo_fraction"):
        AsyncLoopConfig(windowed_fifo_fraction=1.1)
    with pytest.warns(UserWarning, match="forcing window_size=1"):
        async_loop = AsyncLoopConfig(
            num_prompts_per_train_step=8,
            target_offpolicy_steps=0,
            windowed_fifo_fraction=0.01,
        )
    assert async_loop.window_size == 1


def _make_stub_rl_trainer():
    """Create an Controller with a minimal stub config (no VLLMGenerator validation)."""
    from torchtitan.experiments.rl.observability import metrics as m

    class _StubConfig:
        async_loop = AsyncLoopConfig()
        metrics = m.MetricsProcessor.Config()
        dump_folder = "/tmp/test_rl"
        rollout_recorder = RolloutSampleRecorder.Config()
        hf_assets_path = "./tests/assets/tokenizer"
        # __init__ builds these too; stub them so construction does no real work.
        renderer = SimpleNamespace(
            build=lambda *, tokenizer_path: SimpleNamespace(
                get_stop_token_ids=lambda: [],
                _tokenizer=SimpleNamespace(eos_token_id=0),
            )
        )
        # __init__ reads generator.sampling (a dataclass, for replace) + generator.debug.seed.
        generator = SimpleNamespace(
            sampling=SamplingConfig(), debug=SimpleNamespace(seed=None)
        )
        rollouter = SimpleNamespace(build=lambda: SimpleNamespace())

        def to_dict(self):
            return {}

    return train.Controller(_StubConfig())


@pytest.fixture
def stub_mesh_provisioning(monkeypatch):
    """Stub the Monarch mesh provisioning ``main()`` runs before ``setup_async``.

    ``spawn_proc_mesh`` spawns real GPU proc meshes and the
    ``_compute_*_world_size`` helpers read parallelism degrees; neither matters
    to shutdown behavior, so stub both and let the test intercept at the
    ``_FakeController`` boundary.
    """
    monkeypatch.setattr(train, "_compute_trainer_world_size", lambda p: 1)
    monkeypatch.setattr(train, "_compute_generator_world_size", lambda p: 1)

    def _spawn_proc_mesh(*args, num_generators=1, **kwargs):
        return "trainer_mesh", [
            f"generator_mesh_{idx}" for idx in range(num_generators)
        ]

    monkeypatch.setattr(train, "spawn_proc_mesh", _spawn_proc_mesh)


def test_main_shuts_down_after_success(monkeypatch, stub_mesh_provisioning):
    _FakeConfigManager.config = _FakeConfig()
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigManager", _FakeConfigManager)

    asyncio.run(train.main())

    trainer = _FakeController.instances[0]
    assert trainer.events == ["setup", "train", "close"]
    assert trainer.setup_trainer_mesh == "trainer_mesh"
    assert trainer.setup_generator_meshes == ["generator_mesh_0"]


def test_main_passes_configured_num_generators(monkeypatch, stub_mesh_provisioning):
    _FakeConfigManager.config = _FakeConfig(num_generators=2)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigManager", _FakeConfigManager)

    asyncio.run(train.main())

    trainer = _FakeController.instances[0]
    assert trainer.events == ["setup", "train", "close"]
    assert trainer.setup_generator_meshes == [
        "generator_mesh_0",
        "generator_mesh_1",
    ]


def test_main_shuts_down_after_train_failure(monkeypatch, stub_mesh_provisioning):
    _FakeConfigManager.config = _FakeConfig(fail_train=True)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigManager", _FakeConfigManager)

    with pytest.raises(RuntimeError, match="train failed"):
        asyncio.run(train.main())

    assert _FakeController.instances[0].events == ["setup", "train", "close"]


def test_main_shuts_down_after_setup_failure(monkeypatch, stub_mesh_provisioning):
    _FakeConfigManager.config = _FakeConfig(fail_setup=True)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigManager", _FakeConfigManager)

    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(train.main())

    assert _FakeController.instances[0].events == ["setup", "close"]


def test_rl_trainer_shutdown_is_noop_before_meshes_spawn():
    trainer = _make_stub_rl_trainer()

    asyncio.run(trainer.close())

    assert trainer.trainer is None
    assert trainer.generator_router is None
    assert trainer._proc_meshes == []


def test_main_swallows_cancellation_after_shutdown(monkeypatch, stub_mesh_provisioning):
    """Signal-driven cancellation surfaces as ``CancelledError`` from the
    running task; ``main`` runs ``close`` in ``finally`` and the explicit
    ``except`` clause swallows the interrupt so the process exits 0
    without a traceback."""
    _FakeConfigManager.config = _FakeConfig(cancel_train=True)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigManager", _FakeConfigManager)

    # No exception escapes; close still ran.
    asyncio.run(train.main())

    assert _FakeController.instances[0].events == ["setup", "train", "close"]


def test_vllm_generator_does_not_touch_cuda_from_finalizer():
    assert "__del__" not in VLLMGenerator.__dict__


class _StubEndpoint:
    def __init__(self, name, events, raises=False):
        self._name = name
        self._events = events
        self._raises = raises

    async def call(self):
        self._events.append(self._name)
        if self._raises:
            raise RuntimeError(f"{self._name} failed")


class _StubActor:
    def __init__(self, name, events, raises=False):
        self.close = _StubEndpoint(name, events, raises)

    def flatten(self, *args, **kwargs):
        # The router builds a rank-0 handle via flatten("rank").slice(rank=0);
        # a single-rank stub collapses to itself for both.
        return self

    def slice(self, **kwargs):
        return self

    def __len__(self):
        return 1


class _StubMesh:
    def __init__(self, name, events):
        self._name = name
        self._events = events

    async def stop(self):
        self._events.append(self._name)


def _set_generator_router(rl_trainer, generators):
    rl_trainer.generator_router = InterGeneratorRouter(
        InterGeneratorRouter.Config(),
        generators=generators,
    )


def test_shutdown_calls_actor_close_before_mesh_stop():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _StubActor("trainer.close", events)
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer._proc_meshes = [
        _StubMesh("mesh.stop[0]", events),
        _StubMesh("mesh.stop[1]", events),
    ]

    asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "generator.close",
        "mesh.stop[0]",
        "mesh.stop[1]",
    ]
    assert rl_trainer._proc_meshes == []


def test_shutdown_closes_all_generators():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _StubActor("trainer.close", events)
    _set_generator_router(
        rl_trainer,
        [
            _StubActor("generator[0].close", events),
            _StubActor("generator[1].close", events),
        ],
    )
    rl_trainer._proc_meshes = [_StubMesh("mesh.stop[0]", events)]

    asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "generator[0].close",
        "generator[1].close",
        "mesh.stop[0]",
    ]


def test_shutdown_continues_after_actor_close_failure():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _StubActor("trainer.close", events, raises=True)
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer._proc_meshes = [_StubMesh("mesh.stop[0]", events)]

    asyncio.run(rl_trainer.close())

    # trainer.close raised, but every later step still ran.
    assert events == ["trainer.close", "generator.close", "mesh.stop[0]"]


def test_shutdown_continues_after_generator_close_failure():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _StubActor("trainer.close", events)
    _set_generator_router(
        rl_trainer,
        [
            _StubActor("generator[0].close", events, raises=True),
            _StubActor("generator[1].close", events),
        ],
    )
    rl_trainer._proc_meshes = [_StubMesh("mesh.stop[0]", events)]

    asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "generator[0].close",
        "generator[1].close",
        "mesh.stop[0]",
    ]
