# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import apply_tp
from torchtitan.tools.logging import logger

from ..backend import get_compile_backend_with_passes

from ..simple_fsdp import data_parallel, MixedPrecisionPolicy


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
}


def get_transformer_block_buckets(model) -> list[list[str] | str]:
    module_list = [
        model.tok_embeddings,
        [model.norm, model.output],
    ]
    for layer_id, transformer_block in model.layers.items():
        module_list.append(transformer_block)

    def convert_modules_to_fqns(modules, module_to_fqn_mapping):
        """Convert a (possibly nested) list of modules to FQN strings."""
        result = []
        for m in modules:
            if isinstance(m, list):
                if fqn_list := convert_modules_to_fqns(m, module_to_fqn_mapping):
                    result.append(fqn_list)
            else:
                if fqn := module_to_fqn_mapping.get(m):
                    result.append(fqn)
        return result

    module_to_name = {m: n for n, m in model.named_modules()}
    module_fqns = convert_modules_to_fqns(module_list, module_to_name)
    return module_fqns


def parallelize_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
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
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.world_mesh["tp"]
        apply_tp(
            model,
            tp_mesh,
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
        )
        maybe_enable_async_tp(job_config, tp_mesh)

    if job_config.activation_checkpoint.mode != "none":
        attn_type = getattr(model.model_args, "attn_type", "sdpa")
        use_flex_attn = attn_type == "flex"
        model_compile_enabled = (
            job_config.compile.enable and "model" in job_config.compile.components
        )
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )

    # apply data parallel
    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
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

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )

        model = data_parallel(
            model,
            parallel_dims.world_mesh[tuple(dp_mesh_dim_names)],
            mode=dp_mode,
            mp_policy=mp_policy,
        )
        logger.info(
            "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
        )

    if job_config.compile.enable and "model" in job_config.compile.components:
        torch._inductor.config.reorder_for_peak_memory = False

        match job_config.parallelism.fsdp_reshard_after_forward:
            case "always":
                fsdp_reshard_after_forward = True
            case "never":
                fsdp_reshard_after_forward = False
            case "default":
                # For PP, by default do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                fsdp_reshard_after_forward = not parallel_dims.pp_enabled
            case _:
                raise ValueError(
                    f"Invalid fsdp_reshard_after_forward_policy: {job_config.parallelism.fsdp_reshard_after_forward}."
                )

        backend = get_compile_backend_with_passes(
            job_config.compile,
            fsdp_reshard_after_forward,
            get_transformer_block_buckets(model),
        )
        model = torch.compile(
            model,
            backend=backend,
            fullgraph=True,
        )

    return model
