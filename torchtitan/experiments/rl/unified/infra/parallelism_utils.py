# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization utilities for vLLM + TorchTitan models.

This module provides functions for setting up device mesh and applying
tensor parallelism to TorchTitan models in vLLM using TorchTitan's ParallelDims.
"""

from torchtitan.config.job_config import Comm
from torchtitan.distributed import utils as dist_utils

from torchtitan.distributed.parallel_dims import ParallelDims
from vllm.logger import init_logger


logger = init_logger(__name__)


def create_trainer_parallel_dims(ddp_size, tp_size) -> ParallelDims:
    """
    Create ParallelDims for trainer with specified DDP and TP sizes.

    This function initializes the distributed process group and creates a ParallelDims
    object configured for for trainer SPMD workers.

    Args:
        ddp_size: Data parallel (DDP) replicate size
        tp_size: Tensor parallel size

    Returns:
        ParallelDims object with trainer parallelism settings
    """
    world_size = dist_utils.init_distributed(
        Comm(),
    )
    return ParallelDims(
        dp_replicate=ddp_size,
        dp_shard=1,
        tp=tp_size,
        cp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )
