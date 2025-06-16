# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims

from torchtitan.models.llama3.parallelize_llama import (
    apply_ac,
    apply_compile,
    apply_ddp,
    apply_fsdp,
    apply_tp,
)
from torchtitan.tools.logging import logger

from ..model.moe import MoE


def parallelize_llama(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        if (
            job_config.parallelism.enable_async_tensor_parallel
            and not job_config.training.compile
        ):
            raise RuntimeError("Async TP requires --training.compile")

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=job_config.parallelism.enable_async_tensor_parallel,
        )

        apply_moe_tp(model, world_mesh["tp"])

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training.compile:
        apply_compile(model)

        # NOTE: needed for torch.compile to work with dynamic shapes in token-choice MoE
        torch._dynamo.config.capture_scalar_outputs = True

    dp_mesh: DeviceMesh | None = None
    if (
        parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

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

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        dp_mesh = world_mesh
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )

    return model


def apply_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    from torch.distributed.tensor import Partial, Replicate, Shard
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        PrepareModuleInputOutput,
    )

    from .expert_parallel import NoParallel, TensorParallel

    for transformer_block in model.layers.values():
        moe_layer_plan = {
            # input / output sharding on the seqlen dim
            # all-gather for input, reduce-scatter for output
            "moe": PrepareModuleInputOutput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
                use_local_input=True,
                output_layouts=(Partial(),),
                desired_output_layouts=(Shard(1),),
            ),
            # replicate computation for the router
            "moe.router.gate": NoParallel(),
            # input Replicate, output Partial
            "moe.experts": TensorParallel(output_layout=Partial()),
            "moe.shared_expert": TensorParallel(output_layout=Partial()),
        }
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=moe_layer_plan,
        )


def get_updated_expert_bias(
    tokens_per_expert,
    expert_bias,
    expert_bias_update_rate,
    reduce_mesh: DeviceMesh | None,
):
    """Update expert bias for biased expert routing. See https://arxiv.org/abs/2408.15664v1#

    Args:
        tokens_per_expert (torch.Tensor): The number of tokens assigned to each expert.
        expert_bias (torch.Tensor): The bias for each expert.
        expert_bias_udpate_rate (float): The update rate for the expert bias.
    """

    with torch.no_grad():
        # All Reduce Across TPxCPxDP group
        if reduce_mesh is not None:
            torch.distributed.all_reduce(
                tokens_per_expert,
                group=reduce_mesh.get_group(),
            )
        average_tokens = tokens_per_expert.sum(
            dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        updated_expert_bias = expert_bias + torch.sign(
            offset) * expert_bias_update_rate
        return updated_expert_bias


# for MoE auxiliary-loss-free load balancing
def update_router_expert_bias(model: torch.nn.Module,
                              world_mesh: DeviceMesh):
    """
    Update the expert bias of the router for a global batch.
    This requires all-reduce of tokens_per_expert across DPxCPxTP2EP ranks
    """
    tokens_per_expert_list = []
    expert_bias_list = []
    global_load_balance_coeff = None

    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise NotImplementedError(
            "Model structure not recognized for MoE expert bias update."
        )
    for transformer_block in layers.values():
        if transformer_block.moe_enabled:
            load_balance_coeff = transformer_block.moe.load_balance_coeff
            if load_balance_coeff is not None and load_balance_coeff > 0:
                if global_load_balance_coeff is None:
                    global_load_balance_coeff = load_balance_coeff
                else:
                    assert (
                        global_load_balance_coeff == load_balance_coeff
                    ), "All MoE layers must have the same load balance coefficient."
                tokens_per_expert_list.append(
                    transformer_block.moe.tokens_per_expert)
                expert_bias_list.append(transformer_block.moe.expert_bias)
            else:
                break

    if len(expert_bias_list) == 0:
        return

    if world_mesh is not None and world_mesh.size() > 1:
        try:
            load_balance_reduce_mesh = world_mesh["dp_cp_tp2ep"]
        except KeyError:
            load_balance_reduce_mesh = None
    else:
        load_balance_reduce_mesh = None

    stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)
    stacked_expert_bias = torch.stack(expert_bias_list, dim=0)
    stacked_updated_expert_bias = get_updated_expert_bias(
        stacked_tokens_per_expert,
        stacked_expert_bias,
        global_load_balance_coeff,
        load_balance_reduce_mesh,
    )

    for tokens_per_expert, expert_bias, updated_expert_bias in zip(
            tokens_per_expert_list, expert_bias_list,
            stacked_updated_expert_bias):
        tokens_per_expert.zero_()
        expert_bias.copy_(updated_expert_bias)
