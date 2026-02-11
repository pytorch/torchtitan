# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.distributed._tensor import distribute_tensor
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement

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
    _register_parametrization,
    MixedPrecisionPolicy,
    ReplicateComputation,
)


def get_dense_spmd_mesh(parallel_dims: ParallelDims) -> DeviceMesh:
    mesh_names = [
        n for n in ["dp_replicate", "fsdp"] if parallel_dims.get_optional_mesh(n)
    ]

    return parallel_dims.get_mesh(mesh_names)


def get_dp_placements(
    parallel_dims: ParallelDims,
) -> tuple[tuple[Placement, ...], tuple[Placement, ...]]:
    placements: list[Placement] = []
    if parallel_dims.dp_replicate_enabled:
        if parallel_dims.dp_shard_enabled:
            placements = [Replicate(), Shard(0)]
        else:
            placements = [Replicate()]
    elif parallel_dims.dp_shard_enabled:
        placements = [Shard(0)]

    buffer_placements = tuple(Replicate() for _ in range(len(placements)))
    return tuple(placements), buffer_placements


def distribute_state(
    module: nn.Module,
    name: str,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> None:
    try:
        state = module.get_parameter(name)
    except AttributeError:
        try:
            state = module.get_buffer(name)
        except AttributeError as err:
            raise AttributeError(
                f"Module {module} does not have parameter or buffer {name}"
            ) from err

    distributed_state = distribute_tensor(state, mesh, placements)
    is_parameter = isinstance(state, nn.Parameter)
    if is_parameter:
        module.register_parameter(
            name,
            nn.Parameter(distributed_state),
        )
    else:
        module.register_buffer(
            name,
            distributed_state,
        )


def parallelize_tok_embeddings(
    module: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> None:
    # NOTE: The repeat code here and other parallelize_* is left here
    # by intention. We want to understand the code pieces that can be used
    # and provid them as util functions once we enabled more parallelisms.
    mesh = get_dense_spmd_mesh(parallel_dims)
    placements, _ = get_dp_placements(parallel_dims)
    distribute_state(module, "weight", mesh, placements)


def parallelize_attention_norm(
    module: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> None:
    mesh = get_dense_spmd_mesh(parallel_dims)
    placements, _ = get_dp_placements(parallel_dims)
    distribute_state(module, "weight", mesh, placements)


def parallelize_attention(
    module: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> None:
    mesh = get_dense_spmd_mesh(parallel_dims)
    placements, _ = get_dp_placements(parallel_dims)
    for name in ["wq", "wk", "wv", "wo"]:
        distribute_state(module.get_submodule(name), "weight", mesh, placements)


def parallelize_ffn_norm(
    module: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> None:
    mesh = get_dense_spmd_mesh(parallel_dims)
    placements, _ = get_dp_placements(parallel_dims)
    distribute_state(module, "weight", mesh, placements)


def parallelize_feed_forward(
    module: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> None:
    mesh = get_dense_spmd_mesh(parallel_dims)
    placements, _ = get_dp_placements(parallel_dims)
    for name in ["w1", "w2", "w3"]:
        distribute_state(module.get_submodule(name), "weight", mesh, placements)


def parallelize_norm(
    module: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> None:
    mesh = get_dense_spmd_mesh(parallel_dims)
    placements, _ = get_dp_placements(parallel_dims)
    distribute_state(module, "weight", mesh, placements)


def parallelize_output(
    module: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> None:
    mesh = get_dense_spmd_mesh(parallel_dims)
    placements, _ = get_dp_placements(parallel_dims)
    distribute_state(module, "weight", mesh, placements)


def parallelize_outer_model(
    module: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> None:
    mesh = get_dense_spmd_mesh(parallel_dims)
    _, buffer_placements = get_dp_placements(parallel_dims)
    for name, buffer in module.named_buffers():
        distribute_state(module, name, mesh, buffer_placements)


def parallelize_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:

    if parallel_dims.cp_enabled:
        raise NotImplementedError("CP is not implemented yet.")

    if parallel_dims.tp_enabled:
        raise NotImplementedError("TP is not implemented yet.")

    if job_config.activation_checkpoint.mode != "none":
        raise NotImplementedError("AC is not implemented yet.")

    def parallelize_module(name, module):
        parallelize_fns = {
            "tok_embeddings": parallelize_tok_embeddings,
            "attention": parallelize_attention,
            "attention_norm": parallelize_attention_norm,
            "feed_forward": parallelize_feed_forward,
            "ffn_norm": parallelize_ffn_norm,
            "norm": parallelize_norm,
            "output": parallelize_output,
        }

        no_parallelize_fn = True
        for pattern, parallelize_fn in parallelize_fns.items():
            if pattern == name:
                parallelize_fn(module, parallel_dims, job_config)
                no_parallelize_fn = False
                break

        if no_parallelize_fn:
            raise ValueError(f"Cannot find the parallel_fn for {name}")

    for name, child in model.named_children():
        if name != "layers":
            parallelize_module(name, child)

    for layer in model.layers.values():
        for name, child in layer.named_children():
            parallelize_module(name, child)

    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
    ):
        fsdp_mesh_names = [
            n for n in ["dp_replicate", "fsdp"] if parallel_dims.get_optional_mesh(n)
        ]
        fsdp_mesh = parallel_dims.get_mesh(fsdp_mesh_names)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )

        for mod in model.modules():
            params_dict = dict(mod.named_parameters(recurse=False))
            _register_parametrization(
                mod,
                list(params_dict.keys()),
                ReplicateComputation(
                    fsdp_mesh,
                    (),
                    mode="full_dtensor",
                    mp_policy=mp_policy,
                    full_dtensor=True,
                ),
            )

    return apply_compile(model, parallel_dims, job_config)


def parallelize_buffers(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> nn.Module:
    # This is also a temporarily workaround, we need to figure out the best
    # way to understand the sharding for both parameters and buffers and
    # apply the sharding.
    spmd_mesh = get_dense_spmd_mesh(parallel_dims)
    placements = tuple(Replicate() for _ in range(spmd_mesh.ndim))
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
    mesh = get_dense_spmd_mesh(parallel_dims)

    placements = []
    # Note: placements order must match mesh dimension order ["dp_replicate", "fsdp"]
    if parallel_dims.dp_replicate_enabled:
        placements.append(Shard(0))
    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        placements.append(Shard(0))

    assert mesh.ndim == len(placements)

    def parallelize_inputs(
        inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[DTensor, DTensor]:
        inputs = DTensor.from_local(inputs, mesh, placements)
        labels = DTensor.from_local(labels, mesh, placements)
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

    return joint_custom_pass


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
