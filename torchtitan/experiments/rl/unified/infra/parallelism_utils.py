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
from torchtitan.config import CommConfig, ParallelismConfig, TrainingConfig
from torchtitan.trainer import Trainer

JobConfig = Trainer.Config
from torchtitan.distributed import utils as dist_utils

from torchtitan.distributed.parallel_dims import ParallelDims
from vllm.config import VllmConfig
from vllm.logger import init_logger


logger = init_logger(__name__)


def create_parallel_dims_from_vllm_config(vllm_config: VllmConfig) -> ParallelDims:
    """
    Create ParallelDims from vLLM config and maps vLLM parallelism settings to TorchTitan's ParallelDims dataclass.

    This function is needed because vLLM doesn't separate model creation and
    parallelism application - it requires parallelization to be done inside
    the model constructor, so we are creating parallel_dims and apply parallelism
    in TorchTitanVLLMModelWrapper.__init__ function.

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
        CommConfig(),
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


def create_job_config_from_vllm_config(
    vllm_config: VllmConfig,
    model_name: str = "qwen3",
    hf_assets_path: str = "/path/to/hf/assets",
) -> JobConfig:
    """
    Create TorchTitan JobConfig from vLLM configuration.

    Args:
        vllm_config: vLLM configuration object containing model, parallel, and cache configs
        model_name: Model name to use (default: "qwen3")
        hf_assets_path: Path to HuggingFace assets directory (default: "/path/to/hf/assets")

    Returns:
        JobConfig object with settings mapped from vLLM config
    """
    # Create JobConfig with defaults
    job_config = JobConfig()

    job_config.hf_assets_path = hf_assets_path

    parallel_config = vllm_config.parallel_config
    job_config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=parallel_config.data_parallel_size,
        data_parallel_shard_degree=1,  # vLLM doesn't use FSDP sharding in inference
        context_parallel_degree=parallel_config.decode_context_parallel_size,
        tensor_parallel_degree=parallel_config.tensor_parallel_size,
        pipeline_parallel_degree=parallel_config.pipeline_parallel_size,
        expert_parallel_degree=1,  # Not used in vLLM inference yet
        expert_tensor_parallel_degree=1,  # Not used in vLLM inference yet
    )

    job_config.training = TrainingConfig(
        local_batch_size=1,  # Inference typically processes one batch at a time
        steps=1,  # Single step for inference
    )

    return job_config
