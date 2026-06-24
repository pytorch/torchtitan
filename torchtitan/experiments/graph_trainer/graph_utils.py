# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Legacy utilities for the deprecated AOT compile mode. This module is no
# longer used by the main training path (aot_fx_trace) but is kept for
# test infrastructure (export_joint) and potential backward compatibility.

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import torch
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    JointWithDescriptors,
)
from torch._guards import tracing, TracingContext
from torch.utils._pytree import TreeSpec

from torchtitan.config import CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.module import Module


def export_joint(
    model,
    args,
    kwargs=None,
    dump_folder: str | None = None,
    precompile: bool = False,
) -> tuple[JointWithDescriptors, TracingContext]:
    """
    Export joint forward-backward graph with AOT Autograd.

    Args:
        model: The model to export
        args: Tuple of input arguments
        kwargs: Dict of keyword arguments for the model
        dump_folder: Optional folder to dump the graph to
        precompile: If True, enable compile-on-one-rank (CooR) so that
            ProcessGroups flow through the graph as opaque inputs instead
            of being baked as string-literal PG names. This makes the
            serialized artifact rank-agnostic.
    """
    if kwargs is None:
        kwargs = {}
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    import torch.distributed.config as dist_config

    coor_ctx = (
        dist_config.patch("compile_on_one_rank", True)
        if precompile
        else contextlib.nullcontext()
    )
    with coor_ctx:
        with (
            torch._dynamo.config.patch(fake_tensor_cache_enabled=False),
            torch.fx.traceback.preserve_node_meta(),
        ):
            gm = dynamo_graph_capture_for_export(model)(*args, **kwargs)

            tracing_context = gm.meta["tracing_context"]

        with tracing(tracing_context):
            return (
                aot_export_joint_with_descriptors_alone(gm, args, kwargs),
                tracing_context,
            )


def aot_export_joint_with_descriptors_alone(model, args, kwargs=None):
    """
    Export joint forward-backward graph with AOT Autograd.

    Args:
        model: The model to export
        args: Tuple of input arguments
        kwargs: Dict of keyword arguments for the model
    """
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
    fw_compiler: Callable | None = None,
    bw_compiler: Callable | None = None,
    joint_custom_passes: list[Callable] | None = None,
    dump_folder: str | None = None,
    compile_config: CompileConfig | None = None,
    serializable: bool = False,
    on_compile: Callable[[Any, TreeSpec | None], None] | None = None,
):
    """
    Build a joint forward-backward graph for the model with optional custom compilers.

    Args:
        model: The model to compile
        model_args: Tuple of model input arguments
        model_kwargs: Dict of model input keyword arguments
        fw_compiler: Optional custom forward compiler function
        bw_compiler: Optional custom backward compiler function
        joint_custom_passes: list of custom passes to run on the joint graph
        dump_folder: Optional folder to dump the graph to
        compile_config: Compile configuration
        serializable: If True, compile with serialization support
        on_compile: Optional callback invoked after compilation with
            (compiled_fn, out_spec)
    """
    assert isinstance(model_args, tuple)

    # get joint graph
    (joint_with_descriptors, tracing_context,) = export_joint(
        model,
        model_args,
        model_kwargs,
        dump_folder=dump_folder,
        precompile=serializable,
    )

    # run custom passes on joint-graph before partitioner
    if joint_custom_passes is not None:
        for joint_custom_pass in joint_custom_passes:
            joint_with_descriptors.graph_module = joint_custom_pass(
                joint_with_descriptors.graph_module
            )

    with tracing(tracing_context):
        fn = aot_compile_joint_with_descriptors(
            joint_with_descriptors,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            serializable=serializable,
        )

    if on_compile is not None:
        on_compile(fn, joint_with_descriptors.out_spec)

    def wrapper_fn(args, kwargs):
        params = [p for _, p in model.named_parameters(remove_duplicate=False)]
        inputs = [
            *params,
            *model.buffers(),
            *args,
        ]
        return fn(*inputs, **kwargs)

    return wrapper_fn


class CompiledModule(Module):
    def __init__(
        self,
        inner: torch.nn.Module,
        parallel_dims: ParallelDims,
        joint_graph_builder: Callable,
        parallelize_inputs: Callable,
        precompiled_fn: Callable | None = None,
        **overrides,
    ) -> None:
        super().__init__()
        self.inner = inner  # register as submodule
        self.parallel_dims = parallel_dims

        self.joint_graph_builder = joint_graph_builder
        self.joint_graph_module = None
        self.precompiled_fn = precompiled_fn

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

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        # Explicitly delegate to inner model. Without this override,
        # Module.init_states would be found via MRO before the overwritten
        # __getattr__ is triggered, silently skipping weight initialization.
        # This is similar to state_dict, load_state_dict, ...
        self.inner.init_states(buffer_device=buffer_device)

    def state_dict(self, *args, **kwargs) -> Any:
        return self.inner.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs) -> Any:
        return self.inner.load_state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs) -> Any:
        return self.inner.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs) -> Any:
        return self.inner.parameters(*args, **kwargs)

    def forward(self, *args, **kwargs):
        assert "forward" not in self._overrides, "forward cannot be overridden"

        dt_args, dt_kwargs = self.parallelize_inputs(self.parallel_dims, args, kwargs)

        if self.joint_graph_module is None:
            if self.precompiled_fn is not None:
                self.joint_graph_module = self.precompiled_fn
            else:
                self.joint_graph_module = self.joint_graph_builder(
                    self.inner, dt_args, dt_kwargs
                )

        # calling the line below returns control to torchtitan's runner
        # letting it call the backward, and optimizer.
        return self.joint_graph_module(dt_args, dt_kwargs)
