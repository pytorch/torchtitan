# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims

# from checkpoint import load_weights_from_hf
from torchtitan.experiments.deepseek_v3.models.model import DeepseekForCausalLM
from torchtitan.experiments.deepseek_v3.models.model_config import (
    deepseek_config_registry,
)

from torchtitan.models.llama3.parallelize_llama import (
    apply_ac,
    apply_compile,
    apply_ddp,
    apply_fsdp,
    apply_tp,
)
from torchtitan.tools.logging import logger


# Use DeepSeek-V2-Lite as a proxy
model_id = "deepseek-ai/DeepSeek-V2-Lite"


# from ..model.moe import MoE


# Get model parallel subgroup by name:
# e.g. "pp", "ep", None
def get_group(dim_name: Optional[str] = None) -> dist.ProcessGroup:
    glob = torch.distributed.device_mesh._mesh_resources.get_current_mesh()
    return glob.get_group(dim_name)


def parallelize_deepseek(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    logger.info("Applying parallelism to the model...")

    return model
