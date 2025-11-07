# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Callable, Optional

import torch
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    JointWithDescriptors,
)
from torch._guards import tracing, TracingContext
from torch.distributed.tensor import DTensor
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def _clear_traced_params_buffers(
    traced_module: torch.fx.GraphModule, const_keys: list[str]
) -> None:
    """Remove all parameters and buffers from traced module before restoring."""
    for key in const_keys:
        assert key in traced_module._buffers.keys()
        # We don't want constants to show up as a buffer in the state dict.
        # Instead they should just be a direct attribute.
        buffer = getattr(traced_module, key)
        torch.fx.graph_module._del_attr(traced_module, key)
        setattr(traced_module, key, buffer)


def export_joint(
    model, args, kwargs=None
) -> tuple[JointWithDescriptors, TracingContext]:
    if kwargs is None:
        kwargs = {}
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    with (
        # TODO Investigate error on MOE model with use_grouped_mm=False.
        # For repro, see: https://gist.github.com/zhxchen17/d794ff58236243d9faddf713b9fc6a61
        torch._dynamo.config.patch(fake_tensor_cache_enabled=False),
        torch.fx.traceback.preserve_node_meta(),
    ):
        gm = dynamo_graph_capture_for_export(model)(*args, **kwargs)
        logger.info("Dynamo gm:")
        logger.info(gm.print_readable(print_output=False))
        tracing_context = gm.meta["tracing_context"]

    with tracing(tracing_context):
        return (
            aot_export_joint_with_descriptors_alone(gm, args, kwargs),
            tracing_context,
        )


def aot_export_joint_with_descriptors_alone(model, args, kwargs=None):
    if kwargs is None:
        kwargs = {}
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    with contextlib.ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            args,
            kwargs,
        )
        return joint_with_descriptors


def joint_graph_builder(
    model: torch.nn.Module,
    model_args: tuple,
    model_kwargs: dict,
    fw_compiler: Optional[Callable] = None,
    bw_compiler: Optional[Callable] = None,
    joint_custom_pass: Optional[Callable] = None,
):
    """
    Build a joint forward-backward graph for the model with optional custom compilers.

    Args:
        model: The model to compile
        model_args: Tuple of model input arguments (should be DTensors)
        model_kwargs: Dict of model input keyword arguments
        fw_compiler: Optional custom forward compiler function
        bw_compiler: Optional custom backward compiler function
        joint_custom_pass: Optional custom pass to run on the joint graph
    """
    assert isinstance(model_args, tuple)
    for arg in model_args:
        assert isinstance(arg, DTensor)

    # get joint graph
    (
        joint_with_descriptors,
        tracing_context,
    ) = export_joint(model, model_args, model_kwargs)

    # Optional validation
    if joint_custom_pass is not None:
        joint_custom_pass(joint_with_descriptors)

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


class CompiledModule(torch.nn.Module):
    def __init__(
        self,
        inner: torch.nn.Module,
        parallel_dims: ParallelDims,
        joint_graph_builder: Callable,
        parallelize_inputs: Callable,
        **overrides,
    ) -> None:
        super().__init__()
        self.inner = inner  # register as submodule
        self.parallel_dims = parallel_dims

        self.joint_graph_builder = joint_graph_builder
        self.joint_graph_module = None

        self.parallelize_inputs = parallelize_inputs

        self._overrides = overrides  # for custom hooks

    def __getattr__(self, name: str):
        # check overrides
        if "_overrides" in self.__dict__ and name in self._overrides:
            return self._overrides[name]
        try:
            # let nn.Module handle registered stuff
            return super().__getattr__(name)
        except AttributeError:
            # fallback to inner model
            return getattr(self.inner, name)

    def __setattr__(self, name: str, value) -> None:
        if "_overrides" in self.__dict__ and name in self._overrides:
            self._overrides[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if "_overrides" in self.__dict__ and name in self._overrides:
            del self._overrides[name]
        else:
            super().__delattr__(name)

    def forward(self, *args, **kwargs):
        assert "forward" not in self._overrides, "forward cannot be overridden"

        dt_args, dt_kwargs = self.parallelize_inputs(
            self.parallel_dims.world_mesh, args, kwargs
        )

        if self.joint_graph_module is None:
            self.joint_graph_module = self.joint_graph_builder(
                self.inner, dt_args, dt_kwargs
            )

        # calling the line below returns control to torchtitan's runner
        # letting it call the backward, and optimizer.
        return self.joint_graph_module(args, kwargs)
