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

import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.logger import init_logger

from torchtitan.distributed.parallel_dims import ParallelDims


logger = init_logger(__name__)


def create_parallel_dims_from_vllm_config(vllm_config: VllmConfig) -> ParallelDims:
    """
    Create ParallelDims from vLLM config.

    Maps vLLM parallelism settings to TorchTitan's ParallelDims dataclass.

    Args:
        vllm_config: vLLM configuration object

    Returns:
        ParallelDims object with parallelism settings validated

    Note:
        vLLM doesn't use FSDP sharding (dp_shard=1) or expert parallelism (ep=1, etp=1)
        in inference. These are set to default values.
    """
    world_size = dist.get_world_size()

    # Map vLLM config to TorchTitan ParallelDims
    parallel_dims = ParallelDims(
        dp_replicate=vllm_config.parallel_config.data_parallel_size,
        dp_shard=1,  # vLLM doesn't use FSDP sharding
        cp=vllm_config.parallel_config.decode_context_parallel_size,
        tp=vllm_config.parallel_config.tensor_parallel_size,
        pp=vllm_config.parallel_config.pipeline_parallel_size,
        ep=1,  # Expert parallelism not used in vLLM inference yet
        etp=1,  # Expert tensor parallelism not used in vLLM inference yet
        world_size=world_size,
    )

    logger.info(
        f"Created ParallelDims from vLLM config: "
        f"DP={parallel_dims.dp_replicate}, TP={parallel_dims.tp}, "
        f"CP={parallel_dims.cp}, PP={parallel_dims.pp}"
    )

    return parallel_dims


def build_device_mesh_and_parallelize(
    model,
    parallelize_fn,
    parallel_dims: ParallelDims,
):
    """
    Build device mesh and apply parallelization to the model.

    Uses TorchTitan's ParallelDims to build the device mesh with proper validation
    and submesh creation, then applies tensor parallelism to the model using the
    provided parallelize function.

    Args:
        model: The TorchTitan model to parallelize
        parallelize_fn: Function to apply tensor parallelism (e.g., apply_qwen3_tp)
        parallel_dims: ParallelDims object with validated parallelism settings

    Returns:
        The device mesh object
    """
    # Use ParallelDims to build the device mesh
    # This handles all the complexity of:
    # - Validation of parallel dimensions
    # - Building multi-dimensional device mesh
    # - Creating all required submeshes (dp, dp_shard_cp, dp_cp, etc.)
    world_mesh = parallel_dims.world_mesh

    logger.info(f"Built device mesh using ParallelDims: {world_mesh}")

    # Apply tensor parallelism using provided function
    if parallel_dims.tp_enabled:
        tp_mesh = world_mesh["tp"]
        parallelize_fn(
            model=model,
            tp_mesh=tp_mesh,
            loss_parallel=False,
            enable_float8_tensorwise_tp=False,
            enable_async_tp=False,
        )
        logger.info(f"Applied Tensor Parallelism with TP={parallel_dims.tp}")

    return world_mesh
