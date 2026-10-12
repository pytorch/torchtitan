# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.utils import clip_grad_norm_


class TestClipGradNormPipelineRankWithoutGradients(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_rank_without_gradients_joins_the_pp_reduction(self):
        pp_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("pp",)
        )
        params = []
        if self.rank == 1:
            param = torch.nn.Parameter(torch.ones(4, device=self.device_type))
            param.grad = torch.full((4,), 2.0, device=self.device_type)
            params.append(param)

        total_norm = clip_grad_norm_(params, max_norm=1.0, pp_mesh=pp_mesh)

        torch.testing.assert_close(
            total_norm, torch.tensor(4.0, device=self.device_type)
        )
