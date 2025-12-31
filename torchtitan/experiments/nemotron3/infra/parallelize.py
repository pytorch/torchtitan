# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the NemotronH model.

import torch.nn as nn

from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def parallelize_nemotron3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:
    """
    Apply data parallelism (FSDP) to the Nemotron3 model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.

    Args:
        model: The Nemotron3 model to parallelize
        parallel_dims: Parallel dimensions configuration
        job_config: Job configuration

    Returns:
        The parallelized model
    """
    # Validate unsupported parallelisms
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Tensor Parallelism (TP) is not yet supported for Nemotron3. "
            "Please set parallelism.tensor_parallel_degree = 1."
        )
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline Parallelism (PP) is not yet supported for Nemotron3. "
            "Please set parallelism.pipeline_parallel_degree = 1."
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallelism (CP) is not yet supported for Nemotron3. "
            "Please set parallelism.context_parallel_degree = 1."
        )
    if parallel_dims.ep_enabled:
        raise NotImplementedError(
            "Expert Parallelism (EP) is not yet supported for Nemotron3. "
            "Please set parallelism.expert_parallel_degree = 1."
        )

    if parallel_dims.fsdp_enabled:
        # Determine the data parallel mesh dimensions
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh,
    param_dtype,
    reduce_dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply FSDP (Fully Sharded Data Parallel) to the model.

    This shards model parameters across GPUs to reduce memory usage.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # Determine reshard_after_forward behavior
    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, do not reshard after forward to avoid per-microbatch all-gathers
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    # Shard token embeddings
    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard each layer (Mamba, Attention, MLP, or MoE blocks)
    for layer_id, layer_block in model.layers.items():
        fully_shard(
            layer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard final norm and output together (don't reshard by default as they're used immediately)
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    # Apply FSDP to the entire model
    fully_shard(model, **fsdp_config)
