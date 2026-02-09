# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims

from torchtitan.experiments.compiler_toolkit.graph_utils import (
    CompiledModule,
    joint_graph_builder,
    make_compiler_with_passes,
)
from torchtitan.experiments.simple_fsdp.reshard_after_forward import (
    annotate_fsdp_all_gather,
)
from torchtitan.experiments.simple_fsdp.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.tools.logging import logger


def _get_dp_mesh(parallel_dims: ParallelDims) -> DeviceMesh:
    if parallel_dims.dp_replicate_enabled:
        if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
            dp_mesh_dim_names = ["dp_replicate", "fsdp"]
        else:
            dp_mesh_dim_names = ["dp_replicate"]
    else:
        dp_mesh_dim_names = ["fsdp"]

    return parallel_dims.get_mesh(dp_mesh_dim_names)


def _get_spmd_mesh(parallel_dims: ParallelDims) -> DeviceMesh:
    return _get_dp_mesh(parallel_dims)


def apply_dp(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:
    if parallel_dims.dp_replicate_enabled:
        if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
            dp_mode = "hybrid_shard"
        else:
            dp_mode = "replicate"
    else:
        dp_mode = "fully_shard"

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
    )

    model = data_parallel(
        model,
        _get_dp_mesh(parallel_dims),
        mode=dp_mode,
        mp_policy=mp_policy,
        full_dtensor=True,
    )
    logger.info(
        "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
    )
    return model


def parallelize_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:
    # This workflow still reuses the legacy apply-one-parallelization-per-time style.
    # We will change it to a more holistic approach by figuring out the sharding
    # for each parameter first and then apply the transformation to the entire model.
    if parallel_dims.cp_enabled:
        raise NotImplementedError("CP is not implemented yet.")

    if parallel_dims.tp_enabled:
        raise NotImplementedError("TP is not implemented yet.")

    if job_config.activation_checkpoint.mode != "none":
        raise NotImplementedError("AC is not implemented yet.")

    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
    ):
        model = apply_dp(model, parallel_dims, job_config)

    return apply_compile(model, parallel_dims, job_config)


def parallelize_buffers(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> nn.Module:
    # This is also a temporarily workaround, we need to figure out the best
    # way to understand the sharding for both parameters and buffers and
    # apply the sharding.
    spmd_mesh = _get_spmd_mesh(parallel_dims)
    placements = (Replicate() for _ in range(spmd_mesh.ndim))
    for m in model.modules():
        buffers = {
            name: DTensor.from_local(b, spmd_mesh, placements)
            for name, b in m.named_buffers(recurse=False)
        }
        for name, b in buffers.items():
            setattr(m, name, b)

    return model


def build_parallelize_inputs_fn(
    parallel_dims: ParallelDims,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[DTensor, DTensor]]:

    spmd_mesh = _get_spmd_mesh(parallel_dims)

    spmd_placements = []
    # Note: placements order must match mesh dimension order ["dp_replicate", "fsdp"]
    if parallel_dims.dp_replicate_enabled:
        spmd_placements.append(Shard(0))
    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        spmd_placements.append(Shard(0))

    def parallelize_inputs(
        inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[DTensor, DTensor]:
        inputs = DTensor.from_local(inputs, spmd_mesh, spmd_placements)
        labels = DTensor.from_local(labels, spmd_mesh, spmd_placements)
        return inputs, labels

    return parallelize_inputs


def joint_custom_pass_builder(
    parallel_dims: ParallelDims, job_config: JobConfig
) -> Callable:
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
                "Invalid fsdp_reshard_after_forward_policy: "
                f"{job_config.parallelism.fsdp_reshard_after_forward}."
            )

    def joint_ac_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        gm = annotate_fsdp_all_gather(gm, fsdp_reshard_after_forward)
        gm.recompile()
        return gm

    def joint_custom_pass(joint_with_descriptors) -> None:
        # TODO: Is this safe? Or should we use update_joint_with_descriptors from auto_parallel?
        joint_with_descriptors.graph_module = joint_ac_pass(
            joint_with_descriptors.graph_module
        )


def apply_compile(
    model: nn.Module, parallel_dims: ParallelDims, job_config: JobConfig
) -> nn.Module:
    # TODO: This API just implements compiler toolkit.
    # We should also add torch.compile() support

    if not (job_config.compile.enable and "model" in job_config.compile.graph_passes):
        return model

    compiler_passes = []
    # Create compilers with specified passes (defaults to no passes)
    fw_compiler, bw_compiler = make_compiler_with_passes(
        compiler_passes, dump_folder=job_config.job.dump_folder
    )

    # Create custom joint_graph_builder with llama-specific compilers and validation
    llama_joint_graph_builder = functools.partial(
        joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_pass=joint_custom_pass_builder(parallel_dims, job_config),
        dump_folder=job_config.job.dump_folder,
    )

    # Full DTensor trainer will convert the inputs to DTensor, so we don't
    # need CompiledModule to do it.
    def dummy_parallelize_inputs(
        mesh: DeviceMesh, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return args, kwargs

    return CompiledModule(
        model, parallel_dims, llama_joint_graph_builder, dummy_parallelize_inputs
    )
