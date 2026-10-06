# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from torchtitan.rl.train import PerHostProvisioner


def test_allocate_partitions_cuda_devices() -> None:
    with patch.dict(os.environ, {}, clear=True):
        provisioner = PerHostProvisioner(total_gpus=4)

        assert provisioner.allocate(2) == {"CUDA_VISIBLE_DEVICES": "0,1"}
        assert provisioner.allocate(2) == {"CUDA_VISIBLE_DEVICES": "2,3"}


def test_allocate_partitions_xpu_devices() -> None:
    with patch.dict(os.environ, {"ZE_AFFINITY_MASK": "0,1,2,3"}, clear=True):
        provisioner = PerHostProvisioner(total_gpus=4)

        assert provisioner.allocate(2) == {"ZE_AFFINITY_MASK": "0,1"}
        assert provisioner.allocate(2, extra_env={"EXTRA": "1"}) == {
            "ZE_AFFINITY_MASK": "2,3",
            "EXTRA": "1",
        }


def test_allocate_rejects_more_devices_than_available() -> None:
    with patch.dict(os.environ, {}, clear=True):
        provisioner = PerHostProvisioner(total_gpus=4)
        provisioner.allocate(3)

        with pytest.raises(RuntimeError, match="only 1 available"):
            provisioner.allocate(2)
