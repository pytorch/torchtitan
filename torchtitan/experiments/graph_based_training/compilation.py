# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified compilation dispatcher for graph-based training.

Dispatches to JIT (torch.compile) or AOT (joint graph capture) compilation
based on the configured mode.
"""

import functools

import torch

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from .common_utils import parallelize_inputs
from .graph_utils import CompiledModule, joint_graph_builder, make_compiler_with_passes
from .jit_backend import get_jit_compile_backend
from .passes import (
    fsdp_reshard_after_fwd_pass,
    inductor_decomposition_pass,
    validate_and_get_passes,
    validate_flex_attn_annotation_pass,
)


def _get_reshard_policy(job_config: JobConfig, parallel_dims: ParallelDims) -> bool:
    """Determine fsdp_reshard_after_forward policy."""
    match job_config.parallelism.fsdp_reshard_after_forward:
        case "always":
            return True
        case "never":
            return False
        case "default":
            # For PP, by default do not reshard after forward to avoid
            # per-microbatch all-gathers, which can be expensive and non-overlapped
            return not parallel_dims.pp_enabled
        case _:
            raise ValueError(
                f"Invalid fsdp_reshard_after_forward_policy: "
                f"{job_config.parallelism.fsdp_reshard_after_forward}."
            )


def apply_compilation(
    model: torch.nn.Module,
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    mode: str,
    transformer_block_buckets: list | None = None,
) -> torch.nn.Module:
    """
    Unified entry point for both JIT and AOT compilation.

    Args:
        model: The parallelized model (after TP, AC, DP)
        job_config: Job configuration
        parallel_dims: Parallel dimensions
        mode: "jit" or "aot"
        transformer_block_buckets: Model-specific bucket plans for
            transformer_block_bucketing pass

    Returns:
        The compiled model
    """
    pass_names = getattr(job_config.compile, "passes", [])
    fsdp_reshard_after_forward = _get_reshard_policy(job_config, parallel_dims)

    joint_passes, fwd_bwd_passes = validate_and_get_passes(
        pass_names, mode, transformer_block_buckets=transformer_block_buckets
    )

    if mode == "jit":
        return _apply_jit(
            model,
            job_config,
            fsdp_reshard_after_forward,
            pass_names,
            transformer_block_buckets,
        )
    elif mode == "aot":
        return _apply_aot(
            model,
            parallel_dims,
            job_config,
            fsdp_reshard_after_forward,
            joint_passes,
            fwd_bwd_passes,
        )
    else:
        raise ValueError(f"Unknown compilation mode: {mode}")


def _apply_jit(
    model: torch.nn.Module,
    job_config: JobConfig,
    fsdp_reshard_after_forward: bool,
    pass_names: list[str],
    transformer_block_buckets: list | None,
) -> torch.nn.Module:
    """Apply JIT compilation via torch.compile with custom backend."""
    torch._inductor.config.reorder_for_peak_memory = False

    backend = get_jit_compile_backend(
        job_config.compile,
        fsdp_reshard_after_forward,
        pass_names,
        transformer_block_buckets,
    )
    model = torch.compile(model, backend=backend, fullgraph=True)
    logger.info("Applied JIT compilation (torch.compile) to the model")
    return model


def _apply_aot(
    model: torch.nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    fsdp_reshard_after_forward: bool,
    joint_passes: list,
    fwd_bwd_passes: list,
) -> CompiledModule:
    """Apply AOT compilation via joint graph capture."""
    # Build joint custom passes list:
    # 1. validate_flex_attn_annotation (always applied)
    # 2. user-configured joint passes (excluding inductor_decomposition,
    #    which is handled specially by joint_graph_builder since it needs
    #    runtime context like model, joint_with_descriptors, etc.)
    # 3. fsdp_reshard_after_fwd (always applied)
    joint_custom_passes = [validate_flex_attn_annotation_pass]
    joint_custom_passes.extend(
        p for p in joint_passes if p is not inductor_decomposition_pass
    )
    joint_custom_passes.append(
        functools.partial(
            fsdp_reshard_after_fwd_pass,
            reshard_after_forward=fsdp_reshard_after_forward,
        )
    )

    # Build forward/backward compilers with fwd/bwd passes
    fw_compiler, bw_compiler = make_compiler_with_passes(
        fwd_bwd_passes, dump_folder=job_config.job.dump_folder
    )

    # Create joint graph builder with configured passes
    aot_joint_graph_builder = functools.partial(
        joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_passes=joint_custom_passes,
        dump_folder=job_config.job.dump_folder,
        job_config=job_config,
    )

    # TODO: CompiledModule should take sample input as well, so that we can
    # compile ahead of time.
    model = CompiledModule(
        model, parallel_dims, aot_joint_graph_builder, parallelize_inputs
    )
    logger.info("Applied AOT compilation (joint graph capture) to the model")
    return model
