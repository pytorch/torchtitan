# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Callable, Optional

import torch
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
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


def _restore_state_dict(
    original_module: torch.nn.Module, traced_module: torch.fx.GraphModule
) -> None:
    """
    TODO: move this into torch.export
    Restores the state dict of the traced module to match the original module exactly.
    Preserves the original FQNs with dots, creating intermediate empty modules as needed.
    Ensures that the ordering of parameters/buffers matches the original module.
    """
    # Build ID-based lookups for traced module params/buffers
    traced_params: dict[int, tuple[str, torch.nn.Parameter]] = {}
    for name, param in traced_module.named_parameters(remove_duplicate=False):
        traced_params[id(param)] = (name, param)

    traced_buffers: dict[int, tuple[str, torch.Tensor]] = {}
    for name, buffer in traced_module.named_buffers(remove_duplicate=False):
        traced_buffers[id(buffer)] = (name, buffer)

    # Build mapping from old names to new names for graph node updates
    name_mapping: dict[str, str] = {}

    # Restore parameters in the order they appear in original module
    for orig_name, orig_param in original_module.named_parameters(
        remove_duplicate=False
    ):
        if id(orig_param) in traced_params:
            # This param exists in traced module - restore it with original FQN
            traced_name, traced_param = traced_params[id(orig_param)]
            torch.fx.graph_module._assign_attr(traced_param, traced_module, orig_name)
            torch.fx.graph_module._del_attr(traced_module, traced_name)
            name_mapping[traced_name] = orig_name
        else:
            # This param doesn't exist in traced module - add it
            torch.fx.graph_module._assign_attr(orig_param, traced_module, orig_name)

    # Restore buffers in the order they appear in original module
    for orig_name, orig_buffer in original_module.named_buffers(remove_duplicate=False):
        if id(orig_buffer) in traced_buffers:
            # This buffer exists in traced module - restore it with original FQN
            traced_name, traced_buffer = traced_buffers[id(orig_buffer)]
            torch.fx.graph_module._assign_attr(orig_buffer, traced_module, orig_name)
            name_mapping[traced_name] = orig_name
            torch.fx.graph_module._del_attr(traced_module, traced_name)
        else:
            # This buffer doesn't exist in traced module - add it
            torch.fx.graph_module._assign_attr(orig_buffer, traced_module, orig_name)

    param_names = [v[0] for v in traced_params.values()]
    buffer_names = [v[0] for v in traced_buffers.values()]
    const_keys = set(param_names + buffer_names).difference(set(name_mapping.keys()))

    _clear_traced_params_buffers(traced_module, const_keys)

    # Update get_attr nodes in the graph to use the correct FQNs
    for node in traced_module.graph.nodes:
        if node.op == "get_attr" and node.target in name_mapping:
            node.target = name_mapping[node.target]

    traced_module.recompile()


def export_joint(
    model, args, kwargs=None
) -> tuple[JointWithDescriptors, TracingContext]:
    if kwargs is None:
        kwargs = {}
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    with torch._dynamo.config.patch(
        install_free_tensors=True
    ), torch.fx.traceback.preserve_node_meta():
        # TODO: switch to use the official graph_capture API once it is ready
        gm = _dynamo_graph_capture_for_export(model)(*args, **kwargs)

        # Restore the state dict to match the original module
        _restore_state_dict(model, gm)

        logger.info("Dynamo gm:")
        logger.info(gm.print_readable(print_output=False))

        fake_mode = gm.meta.get("fake_mode", None)
        tracing_context = TracingContext(fake_mode)

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
