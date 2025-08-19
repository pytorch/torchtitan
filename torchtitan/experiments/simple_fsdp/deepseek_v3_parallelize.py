# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.llama4.infra.parallelize import apply_moe_ep_tp
from torchtitan.models.deepseek_v3.infra.parallelize import apply_non_moe_tp
from torchtitan.models.llama3.infra.parallelize import apply_ac
from torchtitan.tools.logging import logger

from .simple_fsdp import data_parallel, MixedPrecisionPolicy

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

    if (
        job_config.parallelism.context_parallel_degree > 1
        and model.model_args.use_flex_attn
    ):
        raise NotImplementedError("CP support for FlexAttention is still in progress.")

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
            etp_enabled=parallel_dims.etp_enabled,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
    )

    # apply data parallel
    dp_mesh: DeviceMesh | None = None
    if (
        parallel_dims.fsdp_enabled
        or parallel_dims.ep_enabled
        or parallel_dims.dp_replicate_enabled
    ):
        if parallel_dims.dp_replicate_enabled:
            if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
                dp_mode = "hybrid_shard"
            else:
                dp_mesh_dim_names = ("dp_replicate",)
                dp_mode = "replicate"
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
            dp_mode = "fully_shard"

        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        dp_mod_ep_mesh_dim_names = []
        ep_modules = []
        ep_shared_experts = []
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")
        for _, transformer_block in model.layers.items():
            if transformer_block.moe_enabled:
                ep_modules.append(transformer_block.moe.experts)
                ep_shared_experts.append(transformer_block.moe.shared_experts)

        if not parallel_dims.tp_enabled and parallel_dims.ep_enabled:
            tp_ep_mesh = world_mesh["ep"]
        elif parallel_dims.tp_enabled and parallel_dims.ep_enabled:
            tp_ep_mesh = world_mesh["ep", "tp"]
        else:
            tp_ep_mesh = None

        model = data_parallel(
            model,
            dp_mesh,
            dp_mode,
            ac_mode=job_config.activation_checkpoint.mode,
            mp_policy=mp_policy,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            tp_ep_mesh=tp_ep_mesh,
            dp_mod_ep_mesh=world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
            if parallel_dims.ep_enabled
            else None,
            ep_modules=ep_modules,
            ep_shared_experts=ep_shared_experts,
        )
        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

    if job_config.training.compile:
        torch._inductor.config.reorder_for_peak_memory = False
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, fullgraph=True)

    return model
