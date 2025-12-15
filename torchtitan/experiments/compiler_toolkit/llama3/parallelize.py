# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools

import torch
from torch.fx.traceback import annotate_fn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.compiler_toolkit.common_utils import (
    disable_compile,
    parallelize_inputs,
    register_blockmask_pytree_node,
)

from torchtitan.experiments.compiler_toolkit.graph_utils import (
    CompiledModule,
    get_compiler_passes_from_config,
    get_joint_custom_passes_from_config,
    joint_graph_builder,
    make_compiler_with_passes,
)
from torchtitan.experiments.simple_fsdp.llama3.parallelize import (
    parallelize_llama as simple_fsdp_parallelize_llama,
)


def annotate_llama() -> None:
    from torchtitan.models.attention import FlexAttentionWrapper

    # annotate flex_attention with compile_with_inductor
    FlexAttentionWrapper.forward = annotate_fn(
        {"compile_with_inductor": "flex_attention"}
    )(FlexAttentionWrapper.forward)


def parallelize_llama(
    model: torch.nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> CompiledModule:
    """
    Parallelize and compile a Llama model with optional custom compiler passes.

    Args:
        model: The model to parallelize
        parallel_dims: Parallel dimensions configuration
        job_config: Job configuration

    Returns:
        CompiledModule wrapping the parallelized and compiled model
    """
    annotate_llama()

    register_blockmask_pytree_node()

    # Disable torch.compile over the model in the compiler toolkit style workflow
    with disable_compile(job_config):
        model = simple_fsdp_parallelize_llama(model, parallel_dims, job_config)

    # Get joint custom passes from config
    joint_custom_passes = get_joint_custom_passes_from_config(parallel_dims, job_config)

    # Get compiler passes from config
    compiler_passes = get_compiler_passes_from_config(model, job_config)

    # Create compilers with specified passes (defaults to no passes)
    fw_compiler, bw_compiler = make_compiler_with_passes(
        compiler_passes, dump_folder=job_config.job.dump_folder
    )

    # Create custom joint_graph_builder with llama-specific compilers
    llama_joint_graph_builder = functools.partial(
        joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_passes=joint_custom_passes,
        dump_folder=job_config.job.dump_folder,
    )

    # TODO: CompiledModule should take sample input as well, so that we can
    # compile ahead of time.
    model = CompiledModule(
        model, parallel_dims, llama_joint_graph_builder, parallelize_inputs
    )

    return model
