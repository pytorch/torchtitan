# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from datetime import timedelta
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan.experiments.torchft import (
    create_process_group,
    registered_process_group_names,
    register_process_group_factory,
)


def test_builtin_process_group_factories() -> None:
    timeout = timedelta(seconds=12)

    with (
        patch("torchft.ProcessGroupGloo", autospec=True) as gloo,
        patch("torchft.ProcessGroupNCCL", autospec=True) as nccl,
    ):
        assert create_process_group("GLOO", timeout) is gloo.return_value
        assert create_process_group("nccl", timeout) is nccl.return_value

    gloo.assert_called_once_with(timeout=timeout)
    nccl.assert_called_once_with(timeout=timeout)
    assert {"gloo", "nccl", "mccl"}.issubset(registered_process_group_names())


def test_mccl_process_group_factory_keeps_existing_construction() -> None:
    timeout = timedelta(seconds=5)
    torchcomms = ModuleType("torchcomms")
    new_comm = MagicMock()
    setattr(torchcomms, "new_comm", new_comm)
    torchft_torchcomms = ModuleType("torchft.torchcomms")
    process_group_cls = MagicMock()
    setattr(torchft_torchcomms, "ProcessGroupTorchComms", process_group_cls)

    with patch.dict(
        sys.modules,
        {
            "torchcomms": torchcomms,
            "torchft.torchcomms": torchft_torchcomms,
        },
    ):
        process_group = create_process_group("mccl", timeout)

    new_comm.assert_called_once_with(
        "mccl",
        device=torch.device("cuda"),
        name="mccl_ft",
        timeout=timeout,
        enable_reconfigure=True,
    )
    process_group_cls.assert_called_once_with(new_comm.return_value, timeout=timeout)
    assert process_group is process_group_cls.return_value


def test_external_process_group_factory_registration() -> None:
    timeout = timedelta(seconds=3)
    factory = MagicMock(return_value=object())

    register_process_group_factory("test_backend", factory)
    register_process_group_factory("TEST_BACKEND", factory)

    assert create_process_group("test_backend", timeout) is factory.return_value
    factory.assert_called_once_with(timeout)


def test_process_group_factory_conflict_is_rejected() -> None:
    register_process_group_factory("conflicting_backend", MagicMock())

    with pytest.raises(ValueError, match="already registered"):
        register_process_group_factory("conflicting_backend", MagicMock())


def test_invalid_process_group_factory_registration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        register_process_group_factory("  ", MagicMock())

    with pytest.raises(TypeError, match="must be callable"):
        register_process_group_factory("not_callable", object())


def test_unknown_process_group_lists_registered_backends() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported process group: unknown.*gloo.*mccl.*nccl",
    ):
        create_process_group("unknown", timedelta(seconds=1))
