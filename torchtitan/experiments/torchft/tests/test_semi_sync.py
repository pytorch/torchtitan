# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch.nn as nn

from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.manager import maybe_semi_sync_training


@pytest.mark.parametrize("offload_to_cpu", [False, True])
def test_local_sgd_cpu_offload_is_forwarded(offload_to_cpu):
    local_sgd = ModuleType("torchft.local_sgd")
    local_sgd_cls = mock.Mock(return_value=mock.sentinel.context)
    setattr(local_sgd, "LocalSGD", local_sgd_cls)
    torchft = ModuleType("torchft")
    setattr(torchft, "local_sgd", local_sgd)

    config = FaultTolerance(
        enable=True,
        semi_sync_method="local_sgd",
        sync_steps=7,
        local_sgd_offload_to_cpu=offload_to_cpu,
    )
    ft_manager = SimpleNamespace(_manager=mock.sentinel.manager)
    model = nn.Linear(2, 2)
    optimizer = mock.sentinel.optimizer

    with mock.patch.dict(sys.modules, {"torchft": torchft}):
        result = maybe_semi_sync_training(
            config,
            ft_manager,
            model,
            n_layers=1,
            optimizer=optimizer,
        )

    assert result is mock.sentinel.context
    local_sgd_cls.assert_called_once_with(
        manager=mock.sentinel.manager,
        model=model,
        optimizer=optimizer,
        sync_every=7,
        offload_averaged_parameters_to_cpu=offload_to_cpu,
    )
