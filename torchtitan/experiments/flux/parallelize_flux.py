# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.


import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims


def parallelize_flux(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    # TODO: Add model parallel strategy here
    return model
