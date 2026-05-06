# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from torchtitan.experiments.rl import grpo
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer


class _FakeConfigManager:
    config = object()

    def parse_args(self):
        return self.config


class _FakeRLTrainer:
    instances = []

    def __init__(self, config):
        self.config = config
        self.events = []
        self.instances.append(self)

    async def setup(self):
        self.events.append("setup")
        if getattr(self.config, "fail_setup", False):
            raise RuntimeError("setup failed")

    async def train(self):
        self.events.append("train")
        if getattr(self.config, "fail_train", False):
            raise RuntimeError("train failed")
        if getattr(self.config, "cancel_train", False):
            raise asyncio.CancelledError()

    async def cleanup(self):
        self.events.append("cleanup")


def test_main_cleans_up_after_success(monkeypatch):
    _FakeConfigManager.config = object()
    _FakeRLTrainer.instances = []
    monkeypatch.setattr(grpo, "ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(grpo, "RLTrainer", _FakeRLTrainer)

    asyncio.run(grpo.main())

    assert _FakeRLTrainer.instances[0].events == ["setup", "train", "cleanup"]


def test_main_cleans_up_after_train_failure(monkeypatch):
    class FailingConfig:
        fail_train = True

    _FakeConfigManager.config = FailingConfig()
    _FakeRLTrainer.instances = []
    monkeypatch.setattr(grpo, "ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(grpo, "RLTrainer", _FakeRLTrainer)

    with pytest.raises(RuntimeError, match="train failed"):
        asyncio.run(grpo.main())

    assert _FakeRLTrainer.instances[0].events == ["setup", "train", "cleanup"]


def test_main_cleans_up_after_setup_failure(monkeypatch):
    class FailingConfig:
        fail_setup = True

    _FakeConfigManager.config = FailingConfig()
    _FakeRLTrainer.instances = []
    monkeypatch.setattr(grpo, "ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(grpo, "RLTrainer", _FakeRLTrainer)

    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(grpo.main())

    assert _FakeRLTrainer.instances[0].events == ["setup", "cleanup"]


def test_rl_trainer_cleanup_is_noop_before_meshes_spawn():
    trainer = grpo.RLTrainer(object())

    asyncio.run(trainer.cleanup())

    assert trainer._proc_meshes == []


def test_main_cleans_up_after_cancellation(monkeypatch):
    """When the main task is cancelled (Ctrl-C path), ``main()`` swallows
    the ``CancelledError`` after running ``cleanup`` so the process exits
    cleanly with no spurious asyncio traceback."""

    class CancelledConfig:
        cancel_train = True

    _FakeConfigManager.config = CancelledConfig()
    _FakeRLTrainer.instances = []
    monkeypatch.setattr(grpo, "ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(grpo, "RLTrainer", _FakeRLTrainer)

    # No exception escapes; cleanup still ran.
    asyncio.run(grpo.main())

    assert _FakeRLTrainer.instances[0].events == ["setup", "train", "cleanup"]


def test_vllm_generator_does_not_touch_cuda_from_finalizer():
    assert "__del__" not in VLLMGenerator.__dict__


def test_actors_define_shutdown_endpoint():
    """Both actors must expose a ``shutdown`` endpoint that ``RLTrainer.cleanup``
    calls before stopping the proc mesh — naming matters because ``Actor.stop``
    is reserved by Monarch."""
    from monarch._src.actor.endpoint import EndpointProperty

    assert isinstance(VLLMGenerator.__dict__.get("shutdown"), EndpointProperty)
    assert isinstance(PolicyTrainer.__dict__.get("shutdown"), EndpointProperty)


class _RecordingActor:
    """Records ``shutdown`` calls. The ``shutdown`` attribute is a stand-in
    for a Monarch endpoint reference: ``actor.shutdown.call()`` is the real
    invocation pattern."""

    def __init__(self, name, *, raise_on_shutdown=False):
        self._name = name
        self._raise = raise_on_shutdown

    @property
    def shutdown(self):
        actor = self

        class _EP:
            async def call(self_inner):  # noqa: N805 - mimicking Monarch endpoint
                actor._record("shutdown")
                if actor._raise:
                    raise RuntimeError(f"{actor._name}.shutdown failed")

        return _EP()

    def _record(self, event):
        events.append((self._name, event))


class _RecordingMesh:
    def __init__(self, name):
        self._name = name

    async def stop(self):
        events.append((self._name, "mesh.stop"))


events: list = []


def _setup_recording_trainer(*, trainer_raises=False, generator_raises=False):
    rl_trainer = grpo.RLTrainer(object())
    rl_trainer.trainer = _RecordingActor("trainer", raise_on_shutdown=trainer_raises)
    rl_trainer.generator = _RecordingActor(
        "generator", raise_on_shutdown=generator_raises
    )
    rl_trainer._proc_meshes = [
        _RecordingMesh("trainer_mesh"),
        _RecordingMesh("generator_mesh"),
    ]
    return rl_trainer


def test_cleanup_calls_actor_shutdown_before_mesh_stop():
    events.clear()
    rl_trainer = _setup_recording_trainer()
    asyncio.run(rl_trainer.cleanup())
    assert events == [
        ("trainer", "shutdown"),
        ("generator", "shutdown"),
        ("trainer_mesh", "mesh.stop"),
        ("generator_mesh", "mesh.stop"),
    ]
    assert rl_trainer._proc_meshes == []


def test_cleanup_continues_after_actor_shutdown_failure():
    events.clear()
    rl_trainer = _setup_recording_trainer(trainer_raises=True)
    asyncio.run(rl_trainer.cleanup())
    # trainer.shutdown raised, but generator.shutdown and both mesh.stops
    # must still run; no exception escapes cleanup().
    assert events == [
        ("trainer", "shutdown"),
        ("generator", "shutdown"),
        ("trainer_mesh", "mesh.stop"),
        ("generator_mesh", "mesh.stop"),
    ]


def test_cleanup_skips_missing_actor_attributes():
    """If ``setup`` raised before ``self.trainer`` or ``self.generator`` were
    assigned, ``cleanup`` should not crash trying to call ``.shutdown`` on a
    missing attribute."""
    events.clear()
    rl_trainer = grpo.RLTrainer(object())
    # No self.trainer / self.generator; no proc meshes either.
    asyncio.run(rl_trainer.cleanup())
    assert events == []
    assert rl_trainer._proc_meshes == []
