# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from tests.unit_tests.cpu.test_kimi_k3_vision_dep import _VisionDepChecks


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestKimiK3VisionDepNccl(_VisionDepChecks, DTensorTestBase):
    # The tower's kernels load for the first time while the step's transfers are posted.
    gelu = True
    exact = False
    lr = 1e-3

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_encodes_before_and_backwards_after_the_schedule_under_nccl(self):
        self._check(bubble=False, frozen_tower=False)

    @with_comms
    def test_encodes_and_backwards_in_idle_slots_under_nccl(self):
        self._check(bubble=True, frozen_tower=False)

    @with_comms
    def test_a_backward_waits_in_its_idle_run_for_its_gradient_under_nccl(self):
        self._check(
            bubble=True, frozen_tower=False, cost_ratio=0.25, backward_on=(6, 3)
        )


if __name__ == "__main__":
    unittest.main()
