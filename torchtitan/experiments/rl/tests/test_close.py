# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from torchtitan.experiments.rl import grpo
from torchtitan.experiments.rl.actors.generator import VLLMGenerator


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

    async def close(self):
        self.events.append("close")


def test_main_closes_after_success(monkeypatch):
    _FakeConfigManager.config = object()
    _FakeRLTrainer.instances = []
    monkeypatch.setattr(grpo, "ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(grpo, "RLTrainer", _FakeRLTrainer)

    asyncio.run(grpo.main())

    assert _FakeRLTrainer.instances[0].events == ["setup", "train", "close"]


def test_main_closes_after_train_failure(monkeypatch):
    class FailingConfig:
        fail_train = True

    _FakeConfigManager.config = FailingConfig()
    _FakeRLTrainer.instances = []
    monkeypatch.setattr(grpo, "ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(grpo, "RLTrainer", _FakeRLTrainer)

    with pytest.raises(RuntimeError, match="train failed"):
        asyncio.run(grpo.main())

    assert _FakeRLTrainer.instances[0].events == ["setup", "train", "close"]


def test_main_closes_after_setup_failure(monkeypatch):
    class FailingConfig:
        fail_setup = True

    _FakeConfigManager.config = FailingConfig()
    _FakeRLTrainer.instances = []
    monkeypatch.setattr(grpo, "ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(grpo, "RLTrainer", _FakeRLTrainer)

    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(grpo.main())

    assert _FakeRLTrainer.instances[0].events == ["setup", "close"]


def test_rl_trainer_close_is_noop_before_meshes_spawn():
    trainer = grpo.RLTrainer(object())

    asyncio.run(trainer.close())

    assert trainer.trainer is None
    assert trainer.generator is None
    assert trainer._proc_meshes == []


def test_main_swallows_cancellation_after_close(monkeypatch):
    """Signal-driven cancellation surfaces as ``CancelledError`` from the
    running task; ``main`` runs ``close`` in ``finally`` and the explicit
    ``except`` clause swallows the interrupt so the process exits 0
    without a traceback."""

    class CancelledConfig:
        cancel_train = True

    _FakeConfigManager.config = CancelledConfig()
    _FakeRLTrainer.instances = []
    monkeypatch.setattr(grpo, "ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(grpo, "RLTrainer", _FakeRLTrainer)

    # No exception escapes; close still ran.
    asyncio.run(grpo.main())

    assert _FakeRLTrainer.instances[0].events == ["setup", "train", "close"]


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


class _StubMesh:
    def __init__(self, name, events):
        self._name = name
        self._events = events

    async def stop(self):
        self._events.append(self._name)


def test_close_calls_actor_close_before_mesh_stop():
    events: list[str] = []
    rl_trainer = grpo.RLTrainer(object())
    rl_trainer.trainer = _StubActor("trainer.close", events)
    rl_trainer.generator = _StubActor("generator.close", events)
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


def test_close_continues_after_actor_close_failure():
    events: list[str] = []
    rl_trainer = grpo.RLTrainer(object())
    rl_trainer.trainer = _StubActor("trainer.close", events, raises=True)
    rl_trainer.generator = _StubActor("generator.close", events)
    rl_trainer._proc_meshes = [_StubMesh("mesh.stop[0]", events)]

    asyncio.run(rl_trainer.close())

    # trainer.close raised, but every later step still ran.
    assert events == ["trainer.close", "generator.close", "mesh.stop[0]"]
