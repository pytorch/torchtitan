# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization utilities for vLLM + TorchTitan models.

DEPRECATED: This module contains legacy functions that convert vLLM config -> TorchTitan config.
This is the WRONG direction!

RECOMMENDED: Use TorchTitanConfigAdapter from config_adapter.py instead, which converts
TorchTitan config -> vLLM config (making TorchTitan the single source of truth).

Legacy functions in this module are kept for backward compatibility but will be removed
in a future version.
"""

import warnings

import torch.distributed as dist

from torchtitan.config.job_config import JobConfig, Model, Parallelism, Training
from torchtitan.distributed.parallel_dims import ParallelDims
from vllm.config import VllmConfig
from vllm.logger import init_logger


logger = init_logger(__name__)


def create_parallel_dims_from_vllm_config(vllm_config: VllmConfig) -> ParallelDims:
    """
    DEPRECATED: Create ParallelDims from vLLM config.

    WARNING: This function converts vLLM config -> TorchTitan ParallelDims,
    which makes vLLM the source of truth. This is the WRONG direction!

    RECOMMENDED: Use TorchTitanConfigAdapter.to_parallel_dims() instead, which
    converts TorchTitan model args -> ParallelDims (making TorchTitan the
    single source of truth).

    Args:
        vllm_config: vLLM configuration object

    Returns:
        ParallelDims object with parallelism settings validated

    Note:
        vLLM doesn't use FSDP sharding (dp_shard=1) or expert parallelism (ep=1, etp=1)
        in inference. These are set to default values.
    """
    warnings.warn(
        "create_parallel_dims_from_vllm_config() is deprecated. "
        "Use TorchTitanConfigAdapter.to_parallel_dims() instead to make "
        "TorchTitan config the single source of truth.",
        DeprecationWarning,
        stacklevel=2,
    )
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

    # --- Model Configuration ---
    model_config = vllm_config.model_config
    job_config.model = Model(
        name=model_name,
        hf_assets_path=hf_assets_path,
    )

    # --- Parallelism Configuration ---
    parallel_config = vllm_config.parallel_config
    job_config.parallelism = Parallelism(
        data_parallel_replicate_degree=parallel_config.data_parallel_size,
        data_parallel_shard_degree=1,  # vLLM doesn't use FSDP sharding in inference
        context_parallel_degree=parallel_config.decode_context_parallel_size,
        tensor_parallel_degree=parallel_config.tensor_parallel_size,
        pipeline_parallel_degree=parallel_config.pipeline_parallel_size,
        expert_parallel_degree=1,  # Not used in vLLM inference yet
        expert_tensor_parallel_degree=1,  # Not used in vLLM inference yet
    )

    # --- Training Configuration (minimal defaults for inference) ---
    # vLLM is primarily for inference, but we set reasonable defaults
    # in case the model is used for fine-tuning
    job_config.training = Training(
        local_batch_size=1,  # Inference typically processes one batch at a time
        steps=1,  # Single step for inference
    )

    return job_config
