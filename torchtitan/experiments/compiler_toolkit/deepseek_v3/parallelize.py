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
    joint_graph_builder,
)

from torchtitan.experiments.simple_fsdp.deepseek_v3.parallelize import (
    parallelize_deepseekv3 as simple_fsdp_parallelize_deepseekv3,
)
from torchtitan.tools.logging import logger


def fw_compiler(gm: torch.fx.GraphModule, example_inputs) -> torch.fx.GraphModule:
    logger.info("fwd_gm:")
    logger.info(gm.print_readable(print_output=False))
    return gm


def bw_compiler(gm: torch.fx.GraphModule, example_inputs) -> torch.fx.GraphModule:
    logger.info("bwd_gm:")
    logger.info(gm.print_readable(print_output=False))
    return gm


def annotate_deepseekv3() -> None:
    from torchtitan.distributed.expert_parallel import ExpertParallel
    from torchtitan.models.moe.moe import MoE

    # annotate the MoE with dispatch, compute and combine
    ExpertParallel._token_dispatch = annotate_fn({"EP": "dispatch"})(
        ExpertParallel._token_dispatch
    )
    ExpertParallel._token_combine = annotate_fn({"EP": "combine"})(
        ExpertParallel._token_combine
    )
    MoE.forward = annotate_fn({"EP": "compute"})(MoE.forward)


def parallelize_deepseekv3(
    model: torch.nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> CompiledModule:

    annotate_deepseekv3()

    if job_config.model.flavor.endswith("flex_attn"):
        register_blockmask_pytree_node()

    # Disable torch.compile over the model in the compiler toolkit style workflow
    with disable_compile(job_config):
        model = simple_fsdp_parallelize_deepseekv3(model, parallel_dims, job_config)

    deepseekv3_joint_graph_builder = functools.partial(
        joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_pass=None,
    )

    # TODO: CompiledModule should take sample input as well, so that we can
    # compile ahead of time.
    model = CompiledModule(
        model, parallel_dims, deepseekv3_joint_graph_builder, parallelize_inputs
    )

    return model
