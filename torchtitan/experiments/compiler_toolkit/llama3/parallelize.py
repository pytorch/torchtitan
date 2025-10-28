# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing

from torch.distributed.tensor import DTensor
from torch.fx.passes.regional_inductor import regional_inductor

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.compiler_toolkit.common_utils import (
    disable_compile,
    parallelize_inputs,
)

from torchtitan.experiments.compiler_toolkit.graph_utils import (
    CompiledModule,
    export_joint,
)
from torchtitan.experiments.simple_fsdp.llama3.model import SimpleFSDPTransformer
from torchtitan.experiments.simple_fsdp.llama3.parallelize import (
    parallelize_llama as simple_fsdp_parallelize_llama,
)

from torchtitan.tools.logging import logger


def joint_graph_builder(model, *args, **kwargs):
    assert isinstance(model, SimpleFSDPTransformer)
    assert isinstance(args, tuple)
    for arg in args:
        assert isinstance(arg, DTensor)

    # get joint graph
    (
        joint_with_descriptors,
        tracing_context,
    ) = export_joint(model, args, kwargs)

    # verify user annotation show up in the graph
    for node in joint_with_descriptors.graph_module.graph.nodes:
        if node.target in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            assert "compile_with_inductor" in node.meta.get("custom", {})

    def compiler(gm: torch.fx.GraphModule, example_inputs):
        logger.info("Before compiler:")
        logger.info(gm.print_readable(print_output=False))

        # gm = schedule_overlap_bucketing(gm)

        gm = regional_inductor(gm, example_inputs)

        logger.info("After compiler:")
        logger.info(gm.print_readable(print_output=False))
        return gm

    with tracing(tracing_context):
        fn = aot_compile_joint_with_descriptors(
            joint_with_descriptors, fw_compiler=compiler, bw_compiler=compiler
        )

    def wrapper_fn(args, kwargs):
        inputs = [
            *model.parameters(),
            *model.buffers(),
            *args,
        ]
        return fn(*inputs, **kwargs)

    return wrapper_fn


def annotate_model() -> None:
    from torch.fx.traceback import annotate_fn
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

    annotate_model()

    # Disable torch.compile over the model in the compiler toolkit style workflow
    with disable_compile(job_config):
        model = simple_fsdp_parallelize_llama(model, parallel_dims, job_config)

    # TODO: CompiledModule should take sample input as well, so that we can
    # compile ahead of time.
    model = CompiledModule(
        model, parallel_dims, joint_graph_builder, parallelize_inputs
    )

    return model
