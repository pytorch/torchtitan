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
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.deepseek_v3.infra.parallelize import (
    apply_ac,
    apply_moe_ep_tp,
    apply_non_moe_tp,
)
from torchtitan.tools.logging import logger

from ..simple_fsdp import data_parallel, MixedPrecisionPolicy


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
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}.
        """

    if (
        job_config.parallelism.context_parallel_degree > 1
        and model.model_args.use_flex_attn
    ):
        raise NotImplementedError("CP support for FlexAttention is still in progress.")

    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        if enable_float8_tensorwise_tp:
            # TODO(jianiw): This branch needs to be tested and enabled
            raise NotImplementedError(
                "Currently, float8 tensorwise TP is not tested for deepseekv3"
            )

        use_flex_attn = getattr(model.model_args, "use_flex_attn", False)
        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            use_flex_attn=use_flex_attn,
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled
                and parallel_dims.ep_enabled
                and parallel_dims.etp_enabled
                else None
            ),
            etp_enabled=parallel_dims.etp_enabled,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config)

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

        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")
        dp_mod_ep_mesh = world_mesh[tuple(dp_mod_ep_mesh_dim_names)]

        for _, transformer_block in model.layers.items():
            if transformer_block.moe_enabled and parallel_dims.ep_enabled:
                experts_shard_dim = 0
                assert dp_mod_ep_mesh is not None
                assert hasattr(transformer_block, "moe")
                if (
                    dp_mod_ep_mesh.size() * parallel_dims.ep
                    > transformer_block.moe.experts.num_experts
                ):
                    experts_shard_dim = 1

                # when EP is enable, the routed experts' gradient reduction is done over
                # dp_mod_ep_mesh instead of whole dp_mesh.
                # we add a `fsdp_gradient_divide_factor` to scale gradient over dp_mesh
                # to be consistent with data.
                # TODO (ruisizhang123): update the logic following the link below instead
                # of using a reduction_divide_factor
                # https://github.com/pytorch/torchtitan/pull/1803#discussion_r2415190883
                transformer_block.moe.experts = data_parallel(
                    transformer_block.moe.experts,
                    dp_mod_ep_mesh,
                    dp_mode,
                    ac_mode=job_config.activation_checkpoint.mode,
                    mp_policy=mp_policy,
                    shard_dim=experts_shard_dim,
                    reduction_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
                )

        model = data_parallel(
            model,
            dp_mesh,
            dp_mode,
            ac_mode=job_config.activation_checkpoint.mode,
            mp_policy=mp_policy,
        )

        logger.info(
            "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
        )

    if job_config.compile.enable:
        torch._inductor.config.reorder_for_peak_memory = False
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, backend=job_config.compile.backend, fullgraph=True)

    return model
