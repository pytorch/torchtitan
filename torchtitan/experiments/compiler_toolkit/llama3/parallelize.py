# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools

import torch
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing

from torch.fx.passes.regional_inductor import regional_inductor
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
    joint_graph_builder,
)
from torchtitan.experiments.simple_fsdp.llama3.parallelize import (
    parallelize_llama as simple_fsdp_parallelize_llama,
)

from torchtitan.tools.logging import logger


# TODO: support passing configs into schedule_overlap_bucketing
def autobucketing_reordering_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def compiler(name: str, gm: torch.fx.GraphModule, example_inputs):
    logger.info(f"{name} before compiler:")
    logger.info(gm.print_readable(print_output=False))

    gm = autobucketing_reordering_pass(gm)

    gm = regional_inductor(gm, example_inputs)

    logger.info(f"{name} after compiler:")
    logger.info(gm.print_readable(print_output=False))
    return gm


def fw_compiler(gm: torch.fx.GraphModule, example_inputs) -> None:
    return compiler("fwd_gm", gm, example_inputs)


def bw_compiler(gm: torch.fx.GraphModule, example_inputs) -> None:
    return compiler("bwd_gm", gm, example_inputs)


def validate_flex_attention_annotation(joint_with_descriptors):
    """Verify user annotations show up in the graph."""
    for node in joint_with_descriptors.graph_module.graph.nodes:
        if node.target in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            assert "compile_with_inductor" in node.meta.get("custom", {})


def annotate_llama() -> None:
    from torchtitan.models.attention import FlexAttentionWrapper
    from torchtitan.models.llama3.model.model import TransformerBlock
    
    # Mark TransformerBlock.forward as nested_compile_region
    TransformerBlock.forward = torch.compiler.nested_compile_region(TransformerBlock.forward)

    # annotate flex_attention with compile_with_inductor
    FlexAttentionWrapper.forward = annotate_fn(
        {"compile_with_inductor": "flex_attention"}
    )(FlexAttentionWrapper.forward)


def parallelize_llama(
    model: torch.nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> CompiledModule:

    annotate_llama()

    if job_config.model.flavor.endswith("flex_attn"):
        register_blockmask_pytree_node()

    # Disable torch.compile over the model in the compiler toolkit style workflow
    with disable_compile(job_config):
        model = simple_fsdp_parallelize_llama(model, parallel_dims, job_config)

    # Create custom joint_graph_builder with llama-specific compilers and validation
    llama_joint_graph_builder = functools.partial(
        joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_pass=validate_flex_attention_annotation,
    )

    # TODO: CompiledModule should take sample input as well, so that we can
    # compile ahead of time.
    model = CompiledModule(
        model, parallel_dims, llama_joint_graph_builder, parallelize_inputs
    )

    return model
