# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from torchtitan.experiments.torchft import manager as manager_module
from torchtitan.experiments.torchft.manager import TorchFTManager


def _new_enabled_manager(*, group_size: int = 2) -> TorchFTManager:
    manager = object.__new__(TorchFTManager)
    manager._manager = object()
    manager.use_async_quorum = True
    manager.group_size = group_size
    manager.replicate_pg = mock.sentinel.replicate_pg
    return manager


def test_init_sync_is_forwarded_and_single_replica_skips_managed_group():
    process_group = mock.Mock()
    manager_cls = mock.Mock()
    managed_group_cls = mock.Mock()
    process_group_cls = mock.Mock(return_value=process_group)
    fake_torchft = SimpleNamespace(
        Manager=manager_cls,
        ProcessGroupGloo=process_group_cls,
        process_group=SimpleNamespace(ManagedProcessGroup=managed_group_cls),
    )
    config = TorchFTManager.Config(enable=True, group_size=1, init_sync=False)

    with (
        mock.patch.object(manager_module, "has_torchft", True),
        mock.patch.object(manager_module, "torchft", fake_torchft, create=True),
    ):
        manager = TorchFTManager(config)

    assert manager.manager is manager_cls.return_value
    assert manager_cls.call_args.kwargs["pg"] is process_group
    assert manager_cls.call_args.kwargs["init_sync"] is False
    managed_group_cls.assert_not_called()


def test_sync_hook_covers_replicate_and_fsdp_modules():
    class FakeFSDPModule(nn.Module):
        def __init__(self, num_param_groups: int) -> None:
            super().__init__()
            self.param_groups = [
                SimpleNamespace(_all_reduce_hook=None)
                for _ in range(num_param_groups)
            ]
            self.public_hook = None

        def _get_fsdp_state(self):
            return SimpleNamespace(_fsdp_param_groups=self.param_groups)

        def set_all_reduce_hook(self, hook):
            self.public_hook = hook

    class FakeReplicateModule(FakeFSDPModule):
        pass

    replicate = FakeReplicateModule(1)
    fsdp = FakeFSDPModule(2)
    model = nn.Sequential(replicate, fsdp)
    manager = _new_enabled_manager()

    with (
        mock.patch.object(manager_module, "FSDPModule", FakeFSDPModule),
        mock.patch.object(manager_module.dist, "all_reduce") as all_reduce,
    ):
        manager.maybe_set_all_reduce_hook([model])
        output = torch.ones(1)
        replicate.public_hook(output)
        for param_group in fsdp.param_groups:
            param_group._all_reduce_hook(output)

    assert all_reduce.call_count == 3
    all_reduce.assert_called_with(
        output, group=mock.sentinel.replicate_pg, op=manager_module.ReduceOp.AVG
    )


def test_single_fsdp_parameter_group_uses_public_hook_api() -> None:
    module = mock.MagicMock()
    module._get_fsdp_state.return_value = SimpleNamespace(
        _fsdp_param_groups=[SimpleNamespace(_all_reduce_hook=None)]
    )
    hook = mock.MagicMock()

    manager_module._set_fsdp_all_reduce_hook(module, hook)

    module.set_all_reduce_hook.assert_called_once_with(hook)


def test_multiple_fsdp_parameter_groups_receive_the_same_hook() -> None:
    param_groups = [
        SimpleNamespace(_all_reduce_hook=None),
        SimpleNamespace(_all_reduce_hook=None),
    ]
    module = mock.MagicMock()
    module._get_fsdp_state.return_value = SimpleNamespace(
        _fsdp_param_groups=param_groups
    )
    hook = mock.MagicMock()

    manager_module._set_fsdp_all_reduce_hook(module, hook)

    assert all(group._all_reduce_hook is hook for group in param_groups)
    module.set_all_reduce_hook.assert_not_called()


def test_incompatible_multi_group_fsdp_api_fails_explicitly() -> None:
    compatible_group = SimpleNamespace(_all_reduce_hook=None)
    module = mock.MagicMock()
    module._get_fsdp_state.return_value = SimpleNamespace(
        _fsdp_param_groups=[
            compatible_group,
            SimpleNamespace(),
        ]
    )

    with pytest.raises(RuntimeError, match="parameter-group API is incompatible"):
        manager_module._set_fsdp_all_reduce_hook(module, mock.MagicMock())

    assert compatible_group._all_reduce_hook is None


def test_sync_hook_rejects_model_without_data_parallel_wrapper():
    manager = _new_enabled_manager()

    with pytest.raises(RuntimeError, match="no supported data-parallel module"):
        manager.maybe_set_all_reduce_hook([nn.Linear(2, 2)])


def test_single_replica_bypasses_hooks_and_loss_sync():
    manager = _new_enabled_manager(group_size=1)

    manager.maybe_set_all_reduce_hook([nn.Linear(2, 2)])

    assert manager.loss_sync_pg is None


def test_multi_replica_uses_managed_group_for_loss_sync():
    manager = _new_enabled_manager()

    assert manager.loss_sync_pg is mock.sentinel.replicate_pg
