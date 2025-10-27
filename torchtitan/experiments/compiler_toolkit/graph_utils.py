# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import torch
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import (
    aot_export_joint_with_descriptors,
    JointWithDescriptors,
)
from torch._guards import tracing, TracingContext
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


def export_joint(model, inputs) -> tuple[JointWithDescriptors, TracingContext]:
    assert isinstance(inputs, tuple)
    with torch._dynamo.config.patch(
        install_free_tensors=True
    ), torch.fx.traceback.preserve_node_meta():
        # TODO: switch to use the official graph_capture API once it is ready
        gm = _dynamo_graph_capture_for_export(model)(*inputs)

        # Restore the state dict to match the original module
        _restore_state_dict(model, gm)

        logger.info("Dynamo gm:")
        logger.info(gm.print_readable(print_output=False))

        fake_mode = gm.meta.get("fake_mode", None)
        tracing_context = TracingContext(fake_mode)

    with tracing(tracing_context):
        return aot_export_joint_with_descriptors_alone(gm, inputs), tracing_context


def aot_export_joint_with_descriptors_alone(model, inputs):
    assert isinstance(inputs, tuple)
    with contextlib.ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            inputs,
        )
        return joint_with_descriptors
