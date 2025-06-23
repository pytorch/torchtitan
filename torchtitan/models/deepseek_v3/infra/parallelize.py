# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims


def parallelize_deepseekv3(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    # TODO: Add support for parallelizing the model, this is a placeholder function for now
    return model
