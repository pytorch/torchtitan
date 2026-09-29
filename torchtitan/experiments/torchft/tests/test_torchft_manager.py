# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quorum mode configuration of the TorchFT manager.

torchft is optional and is not installed in this test environment, so the
torchft surface used by the manager is replaced with recording doubles.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from torchtitan.experiments.torchft import manager as manager_module


class RecordingManagedProcessGroup:
    def __init__(self, manager):
        self.manager = manager
        self.registered_names = []

    def register(self, name):
        self.registered_names.append(name)
        return name


@pytest.fixture
def torchft_doubles(monkeypatch):
    doubles = SimpleNamespace(
        Manager=mock.MagicMock(name="Manager"),
        ProcessGroupGloo=mock.MagicMock(name="ProcessGroupGloo"),
        process_group=SimpleNamespace(ManagedProcessGroup=RecordingManagedProcessGroup),
    )
    monkeypatch.setattr(manager_module, "torchft", doubles, raising=False)
    monkeypatch.setattr(manager_module, "has_torchft", True)
    return doubles


def build_manager(**config_overrides):
    return manager_module.TorchFTManager(
        manager_module.TorchFTManager.Config(enable=True, **config_overrides)
    )


def test_default_quorum_runs_asynchronously(torchft_doubles):
    ft_manager = build_manager()

    assert torchft_doubles.Manager.call_args.kwargs["use_async_quorum"] is True
    assert ft_manager.use_async_quorum is True


def test_synchronous_quorum_keeps_cross_replica_synchronization(torchft_doubles):
    """Turning the async quorum off must not turn off replica synchronization.

    Synchronous quorum is the escape hatch for the state export racing with the
    forward pass, so the replicate process group and the loss sync group it
    backs have to stay in place.
    """
    ft_manager = build_manager(use_async_quorum=False)

    assert torchft_doubles.Manager.call_args.kwargs["use_async_quorum"] is False
    assert ft_manager.replicate_pg.registered_names == ["dp_replicate"]
    assert ft_manager.loss_sync_pg is ft_manager.replicate_pg


@pytest.mark.parametrize("semi_sync_method", ["local_sgd", "diloco"])
def test_semi_sync_training_always_uses_synchronous_quorum(
    torchft_doubles, semi_sync_method
):
    """Semi-sync methods synchronize replicas themselves, so they never opt in."""
    ft_manager = build_manager(semi_sync_method=semi_sync_method)

    assert torchft_doubles.Manager.call_args.kwargs["use_async_quorum"] is False
    assert ft_manager.use_async_quorum is False
    assert not hasattr(ft_manager, "replicate_pg")
    assert ft_manager.loss_sync_pg is None
