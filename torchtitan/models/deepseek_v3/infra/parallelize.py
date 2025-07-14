# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.llama4.infra.expert_parallel import NoParallel
from torchtitan.experiments.llama4.infra.parallelize import apply_fsdp, apply_moe_ep_tp
from torchtitan.models.llama3.infra.parallelize import apply_ac, apply_ddp
from torchtitan.tools.logging import logger


# Adapted from llama4/infra/parallelize.py
def parallelize_deepseekv3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    world_mesh = parallel_dims.world_mesh
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallel_dims.tp_enabled:
        if job_config.parallelism.enable_async_tensor_parallel:
            # TODO(jianiw): This branch needs to be tested and enabled
            raise NotImplementedError(
                "Currently, async TP is not tested for deepseekv3. \
                torch.compile is not supported yet, which is required for async TP."
            )

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        if enable_float8_tensorwise_tp:
            # TODO(jianiw): This branch needs to be tested and enabled
            raise NotImplementedError(
                "Currently, float8 tensorwise TP is not tested for deepseekv3"
            )

        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            enable_async_tp=False,
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled and parallel_dims.ep_enabled
                else None
            ),
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if job_config.training.compile:
        raise NotImplementedError("torch.compile is not supported yet for deepseekv3")

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            dp_mod_ep_mesh=(
                world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
                if dp_mod_ep_mesh_dim_names
                else None
            ),
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


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    rowwise_parallel, colwise_parallel, prepare_module_input = (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), Replicate()),
                desired_input_layouts=(Replicate(), Replicate()),
            ),
            # use_local_output=False make the output to be a DTensor instead of a plain Tensor
            "attention.wkv_a": NoParallel(use_local_output=False),
            "attention.wkv_b": colwise_parallel(use_local_output=False),
            "attention.kv_norm": NoParallel(use_local_output=False),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }

        if transformer_block.attention.q_lora_rank == 0:
            layer_plan.update(
                {
                    "attention.wq": colwise_parallel(
                        use_local_output=False
                    ),  # This is only used when q_lora_rank==0
                }
            )
        else:
            layer_plan.update(
                {
                    "attention.wq_a": NoParallel(use_local_output=False),
                    "attention.wq_b": colwise_parallel(use_local_output=False),
                    "attention.q_norm": NoParallel(use_local_output=False),
                }
            )

        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )
