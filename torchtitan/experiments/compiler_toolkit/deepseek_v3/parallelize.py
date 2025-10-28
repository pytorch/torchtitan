# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing

from torch.distributed.tensor import DTensor, Replicate

from torch.fx.traceback import annotate_fn
from torch.utils._pytree import tree_map
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.experiments.compiler_toolkit.common_utils import disable_compile

from torchtitan.experiments.compiler_toolkit.graph_utils import (
    CompiledModule,
    export_joint,
)

from torchtitan.experiments.simple_fsdp.deepseek_v3.parallelize import (
    parallelize_deepseekv3 as simple_fsdp_parallelize_deepseekv3,
)
from torchtitan.models.moe.moe import MoE
from torchtitan.tools.logging import logger


def joint_graph_builder(model, *args, **kwargs):
    assert isinstance(args, tuple)
    for arg in args:
        assert isinstance(arg, DTensor)

    # get joint graph
    (
        joint_with_descriptors,
        tracing_context,
    ) = export_joint(model, args, kwargs)

    def fw_compiler(gm: torch.fx.GraphModule, example_inputs):
        logger.info("fwd_gm:")
        logger.info(gm.print_readable(print_output=False))

        # logger.info("fwd_gm after compiler:")
        # logger.info(gm.print_readable(print_output=False))
        return gm

    def bw_compiler(gm: torch.fx.GraphModule, example_inputs):
        logger.info("bwd_gm:")
        logger.info(gm.print_readable(print_output=False))

        # logger.info("bwd_gm after compiler:")
        # logger.info(gm.print_readable(print_output=False))
        return gm

    with tracing(tracing_context):
        fn = aot_compile_joint_with_descriptors(
            joint_with_descriptors, fw_compiler=fw_compiler, bw_compiler=bw_compiler
        )

    def wrapper_fn(args, kwargs):
        inputs = [
            *model.parameters(),
            *model.buffers(),
            *args,
        ]
        return fn(*inputs, **kwargs)

    return wrapper_fn


def parallelize_inputs(world_mesh, args, kwargs):
    def to_dtensor(tensor):
        if isinstance(tensor, torch.Tensor):
            return DTensor.from_local(tensor, world_mesh["tp"], [Replicate()])
        return tensor

    dt_args = tree_map(to_dtensor, args)
    dt_kwargs = tree_map(to_dtensor, kwargs)

    return dt_args, dt_kwargs


def annotate_model() -> None:
    # annotate the MoE with dispatch, compute and combine
    ExpertParallel._token_dispatch = annotate_fn({"EP": "dispatch"})(
        ExpertParallel._token_dispatch
    )
    ExpertParallel._token_combine = annotate_fn({"EP": "combine"})(
        ExpertParallel._token_combine
    )
    MoE.forward = annotate_fn({"EP": "compute"})(MoE.forward)


def parallelize_deepseekv3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> CompiledModule:

    annotate_model()

    # Disable torch.compile over the model in the compiler toolkit style workflow
    with disable_compile(job_config):
        model = simple_fsdp_parallelize_deepseekv3(model, parallel_dims, job_config)

    # TODO: CompiledModule should take sample input as well, so that we can
    # compile ahead of time.
    model = CompiledModule(
        model, parallel_dims, joint_graph_builder, parallelize_inputs
    )

    return model
