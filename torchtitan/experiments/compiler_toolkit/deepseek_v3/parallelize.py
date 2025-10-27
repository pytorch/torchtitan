# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from contextlib import contextmanager

import torch
import torch.nn as nn

from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing

from torch.distributed.tensor import DTensor, Replicate

from torch.fx.traceback import annotate_fn
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.expert_parallel import ExpertParallel

from torchtitan.experiments.compiler_toolkit.graph_utils import export_joint
from torchtitan.experiments.simple_fsdp.deepseek_v3.parallelize import (
    parallelize_deepseekv3 as simple_fsdp_parallelize_deepseekv3,
)
from torchtitan.models.moe.moe import MoE
from torchtitan.tools.logging import logger


@contextmanager
def disable_compile(job_config: JobConfig):
    """Context manager to temporarily disable compilation."""
    original_value = job_config.compile.enable
    job_config.compile.enable = False
    try:
        yield
    finally:
        job_config.compile.enable = original_value


class CompiledModule(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module, parallel_dims, **overrides):
        super().__init__()
        self.inner = inner  # register as submodule
        self.parallel_dims = parallel_dims

        self.joint_graph_module = None
        self._overrides = overrides  # for custom hooks

    def __getattr__(self, name):
        # check overrides
        if "_overrides" in self.__dict__ and name in self._overrides:
            return self._overrides[name]
        try:
            # let nn.Module handle registered stuff
            return super().__getattr__(name)
        except AttributeError:
            # fallback to inner model
            return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if "_overrides" in self.__dict__ and name in self._overrides:
            self._overrides[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if "_overrides" in self.__dict__ and name in self._overrides:
            del self._overrides[name]
        else:
            super().__delattr__(name)

    def forward(self, *args, **kwargs):
        assert "forward" not in self._overrides, "forward cannot be overridden"
        dt_args = tuple(
            DTensor.from_local(arg, self.parallel_dims.world_mesh["tp"], [Replicate()])
            for arg in args
        )
        if self.joint_graph_module is None:
            self.joint_graph_module = joint_graph_builder(
                self.inner, *dt_args, **kwargs
            )

        # calling the line below returns control to torchtitan's runner
        # letting it call the backward, and optimizer.

        # TODO: add support for kwargs
        return self.joint_graph_module(args)


def joint_graph_builder(model, *inputs, **kwargs):
    assert isinstance(inputs, tuple)
    for input in inputs:
        assert isinstance(input, DTensor)

    # get joint graph
    (
        joint_with_descriptors,
        tracing_context,
    ) = export_joint(model, inputs)

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

    def wrapper_fn(args):
        input = [
            *model.parameters(),
            *model.buffers(),
            *args,
        ]
        return fn(*input)

    return wrapper_fn


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

    # Diable torch.compile over the model in the compiler toolkit style workflow
    with disable_compile(job_config):
        model = simple_fsdp_parallelize_deepseekv3(model, parallel_dims, job_config)

    # TODO: CompiledModule should take sample input as well, so that we can
    # compile ahead of time.
    model = CompiledModule(model, parallel_dims)

    return model
