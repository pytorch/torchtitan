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
