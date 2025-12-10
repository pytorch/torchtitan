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

from torchtitan.config import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims


logger = init_logger(__name__)


def create_parallel_dims(job_config: JobConfig) -> ParallelDims:
    """
    Create ParallelDims from JobConfig.

    Maps JobConfig parallelism settings to TorchTitan's ParallelDims dataclass.
    This follows the same pattern as init_distributed() in train.py.

    Args:
        job_config: TorchTitan JobConfig object with parallelism settings

    Returns:
        ParallelDims object with parallelism settings validated
    """
    world_size = dist.get_world_size()

    parallelism_config = job_config.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=world_size,
    )

    logger.info(
        f"Created ParallelDims from JobConfig: "
        f"DP_replicate={parallel_dims.dp_replicate}, "
        f"DP_shard={parallel_dims.dp_shard}, "
        f"TP={parallel_dims.tp}, "
        f"CP={parallel_dims.cp}, "
        f"PP={parallel_dims.pp}"
    )

    return parallel_dims


def create_job_config_from_vllm_config(vllm_config: VllmConfig) -> JobConfig:
    """
    Create a minimal JobConfig from vLLM config.

    This extracts relevant settings from vLLM config and creates a JobConfig
    object that can be used with TorchTitan's parallelize_qwen3 function.
    We need to translate from vllm to torchtitan JobConfig because now the
    entrypoint is vLLM inference Engine.

    Args:
        vllm_config: vLLM configuration object

    Returns:
        JobConfig with settings mapped from vLLM config
    """
    job_config = JobConfig()

    # Training settings
    job_config.training.seq_len = vllm_config.model_config.max_model_len
    job_config.training.dtype = "bfloat16"  # vLLM inference uses bfloat16
    job_config.training.mixed_precision_param = "bfloat16"
    job_config.training.mixed_precision_reduce = "float32"

    # Parallelism settings
    parallel_config = vllm_config.parallel_config
    job_config.parallelism.tensor_parallel_degree = parallel_config.tensor_parallel_size
    job_config.parallelism.pipeline_parallel_degree = (
        parallel_config.pipeline_parallel_size
    )
    job_config.parallelism.data_parallel_shard_degree = 1  # vLLM doesn't use FSDP
    job_config.parallelism.data_parallel_replicate_degree = (
        parallel_config.data_parallel_size
    )
    job_config.parallelism.context_parallel_degree = (
        parallel_config.decode_context_parallel_size
    )
    job_config.parallelism.disable_loss_parallel = (
        True  # Inference doesn't need loss parallel
    )
    job_config.parallelism.enable_async_tensor_parallel = (
        False  # Disabled for inference
    )

    # Activation checkpoint - disabled for inference
    job_config.activation_checkpoint.mode = "none"

    # Compile - disabled for inference
    job_config.compile.enable = False

    # Model settings
    job_config.model.converters = []  # No converters for inference

    logger.info(
        f"Created JobConfig from vLLM config: "
        f"seq_len={job_config.training.seq_len}, "
        f"tp={job_config.parallelism.tensor_parallel_degree}, "
        f"pp={job_config.parallelism.pipeline_parallel_degree}"
    )

    return job_config


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

    return world_mesh
