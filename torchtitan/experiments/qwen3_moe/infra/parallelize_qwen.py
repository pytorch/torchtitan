# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch
import torch.distributed as dist

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

# from checkpoint import load_weights_from_hf
from torchtitan.experiments.qwen3_moe.model.model import QwenForCausalLM

from torchtitan.tools.logging import logger


# Use DeepSeek-V2-Lite as a proxy
model_id = "deepseek-ai/DeepSeek-V2-Lite"


# from ..model.moe import MoE


# Get model parallel subgroup by name:
# e.g. "pp", "ep", None
def get_group(dim_name: Optional[str] = None) -> dist.ProcessGroup:
    glob = torch.distributed.device_mesh._mesh_resources.get_current_mesh()
    return glob.get_group(dim_name)


def parallelize_qwen(
    model: torch.nn.Module,
    parallel_dims: DeviceMesh,
    job_config,
):
    """
    Apply parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    logger.info("Applying parallelism to the model...")
    world_mesh = parallel_dims.world_mesh
    
    # Apply data parallelism to the model
    fsdp_mesh = world_mesh["dp_shard"]
    
    # Just apply FSDP for now
    from torch.distributed.fsdp import fully_shard
    fully_shard(model, mesh=fsdp_mesh)

    
    logger.info("Applied FSDP to Qwen model")
    return model
