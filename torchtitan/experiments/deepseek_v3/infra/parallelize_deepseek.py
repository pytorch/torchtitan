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
from torchtitan.experiments.deepseek_v3.model import DeepseekForCausalLM

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
    # model: nn.Module,
    world_mesh: DeviceMesh,
    device: torch.device,
    model_args,
    rank: int,
    # parallel_dims: ParallelDims,
    # job_config: JobConfig,
):
    """
    Apply parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    logger.info("Applying parallelism to the model...")
    world_size = int(os.environ["WORLD_SIZE"])

    pp_mesh = world_mesh["pp"]
    ep_mesh = world_mesh["ep"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    ep_size = ep_mesh.size()

    # Apply data parallelism
    fsdp_mesh = world_mesh["fsdp"]
    hsdp_mesh = world_mesh["ep", "fsdp"]

    hsdp_size = hsdp_mesh.size()

    # Apply model parallelism
    model_args.ep_size = ep_size
    model_args.num_stages = pp_size
    model_args.stage_idx = pp_rank
    logger.info(
        f"Parallelism: {rank=}, {ep_size=}, {pp_size=}, {model_args.ep_size=}, {model_args.num_stages=}, {model_args.stage_idx=}"
    )
    # print(model_args)
    # verify world size matches parallelized total
    parallelized_world_size = pp_size * hsdp_size
    logger.info(f"Total Parallelized World size {parallelized_world_size}")
    assert (
        world_size == parallelized_world_size
    ), f"mismatch between total world size {world_size=} and parallelized total {parallelized_world_size}"

    # Instantiate model
    with device, world_mesh:
        model = DeepseekForCausalLM(model_args)
    # Load weights
    # load_weights_from_hf(model, model_id, device)
    model.train()

    # Using `reshard_after_forward=False` to implement Zero-2, i.e. sharding the
    # optimizer (Zero-1) and gradients (Zero-2), but not the model weights.
    # Reason: the MoE is "sparsely activated" compared to the dense model, thus
    # it will be ineconomical re-gather the weights.
    for layer in model.model.layers.values():
        # Apply FSDP to experts
        if hasattr(layer.mlp, "experts"):
            for expert in layer.mlp.experts.values():
                fully_shard(expert, mesh=fsdp_mesh, reshard_after_forward=False)
        # Apply HSDP to other parts such as attention, layernorm, because they
        # are doing DDP on EP dimension
        fully_shard(layer, mesh=hsdp_mesh, reshard_after_forward=False)

    # Apply HSDP on root model (lm_head, embeddings, etc)
    fully_shard(model, mesh=hsdp_mesh, reshard_after_forward=False)

    return (
        model,
        pp_size,
        pp_rank,
        pp_mesh,
        ep_size,
        ep_rank,
    )
