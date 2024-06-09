# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed._tensor import (
    distribute_tensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed._tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)

from torchtitan.models.norms import fused_rms_norm_fn


class TestFusedRMSNorm(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_fused_rms_norm(self):
        mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        x = torch.randn(4, 4, 4, device=self.device_type)  # Shard(1)
        w = torch.randn(4, device=self.device_type, requires_grad=True)  # Replicate

        dx = distribute_tensor(x, mesh, [Shard(1)])
        dw = distribute_tensor(w, mesh, [Replicate()])

        comm_mode = CommDebugMode()
        # fused rmsnorm
        with comm_mode:
            out = fused_rms_norm_fn(dx, dw)

        self.assertEqual(comm_mode.get_total_counts(), 0)

        with comm_mode:
            grad_out = torch.ones_like(out)
            out.backward(grad_out)

        self.assertEqual(comm_mode.get_total_counts(), 0)


if __name__ == "__main__":
    run_tests()
