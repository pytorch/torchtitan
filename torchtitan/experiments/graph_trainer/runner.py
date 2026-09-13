# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _unwrap_subclasses,
    _wrap_subclasses,
    extract_train_state,
    TracedResult,
)


__all__ = ["GraphRunner"]


def _resolve_state_tensors(
    traced_result: TracedResult,
    module: nn.Module | None,
    graph_state: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, ...]:
    """Resolve persistent state tensors in their trace-time input order."""
    model_state, _ = extract_train_state(module)
    if list(model_state) != traced_result.state_fqns:
        raise ValueError(
            "module has different parameter/buffer names than during tracing.\n"
            f"  Traced: {traced_result.state_fqns}\n"
            f"  Got:    {list(model_state)}"
        )

    graph_state_t = graph_state or {}
    if tuple(graph_state_t) != traced_result.graph_state.fqns:
        raise ValueError(
            "graph state has different names than during tracing.\n"
            f"  Traced: {traced_result.graph_state.fqns}\n"
            f"  Got:    {tuple(graph_state_t)}"
        )
    return tuple([*model_state.values(), *graph_state_t.values()])


def _flat_input_metadata(value: Any) -> tuple[Any, ...]:
    """Describe input properties that must remain stable after binding."""
    if not isinstance(value, torch.Tensor):
        return (type(value),)
    stride = value.stride() if value.layout == torch.strided else None
    storage_offset = value.storage_offset() if value.layout == torch.strided else None
    return (
        value.dtype,
        value.device,
        value.layout,
        value.size(),
        stride,
        storage_offset,
        value.requires_grad,
    )


def _state_input_signature(value: Any) -> tuple[Any, ...]:
    """Combine stable input metadata with its storage identity when available."""
    storage_ptr = None
    if (
        isinstance(value, torch.Tensor)
        and not isinstance(value, FakeTensor)
        and value.device.type != "meta"
        and value.layout == torch.strided
    ):
        storage_ptr = value.untyped_storage().data_ptr()
    return (*_flat_input_metadata(value), storage_ptr)


def _flatten_state_inputs(
    traced_result: TracedResult,
    state_tensors: tuple[torch.Tensor, ...],
) -> tuple[Any, ...]:
    """Flatten state tensors and validate compatibility with the traced inputs."""
    flat_inputs, layouts = _unwrap_subclasses(list(state_tensors))
    expected_layouts = {
        index: layout
        for index, layout in traced_result.input_subclass_layouts.items()
        if index < len(state_tensors)
    }
    if layouts != expected_layouts:
        raise ValueError(
            "state inputs have a different tensor-subclass layout than during tracing"
        )

    if traced_result.example_inputs:
        expected_inputs = traced_result.example_inputs[
            : traced_result.num_static_inputs
        ]
    else:
        expected_inputs = tuple(
            node.meta.get("val")
            for node in traced_result.gm.graph.nodes
            if node.op == "placeholder"
        )[: traced_result.num_static_inputs]

    if len(expected_inputs) != len(flat_inputs):
        raise ValueError("state inputs have a different flattened arity than tracing")
    metadata_available = all(value is not None for value in expected_inputs)
    if metadata_available and any(
        _flat_input_metadata(actual) != _flat_input_metadata(expected)
        for actual, expected in zip(flat_inputs, expected_inputs, strict=True)
    ):
        raise ValueError(
            "state inputs have different tensor metadata than during tracing"
        )
    return tuple(flat_inputs)


class GraphRunner:
    """Run a traced graph using state inputs prepared once at initialization.

    Initialization resolves the model and graph-state tensors, flattens them,
    and records their object, metadata, and storage identities.
    Subsequent calls reuse the flattened state inputs to avoid runtime overhead.
    In-place optimizer and buffer updates remain visible through the retained tensors.

    graph-input order, before tensor-subclass unwrapping:

    1. module parameters, ``named_parameters()`` order
    2. module buffers, ``named_buffers()`` order
    3. trainer-owned graph-state tensors,
        currently ``parameter.grad`` accumulation buffers, ``graph_state`` insertion order
    4. optimizer-state leaves (currently unsupported by ``GraphRunner``)
    5. runtime meshes, in the supplied sequence order
    6. user-input leaves from the ``(args, kwargs)`` pytree

    A tensor subclass expands into its plain-tensor leaves at its position in
    that sequence. ``GraphRunner`` caches the flattened inputs from groups 1-3,
    requires group 4 to be empty, retains group 5, and flattens only group 6 on
    each invocation.

    Call :meth:`validate_state` explicitly to diagnose state replacement after
    binding.

    Args:
        traced_result: Finalized graph and metadata produced by the FX tracer.
        module: Module whose parameters and buffers are graph inputs. Their
            tensor objects are retained, so in-place updates remain visible.
        graph_state: Trainer-owned persistent inputs mutated by the graph.
            Currently these are gradient accumulation buffers also exposed as
            ``parameter.grad``. Names and order must match the trace.
        runtime_meshes: Device meshes passed to a precompiled graph, in the same
            order as during tracing.
        validate_user_inputs: Whether to validate the pytree structure of each
            runtime call against the trace-time inputs.
    """

    def __init__(
        self,
        traced_result: TracedResult,
        *,
        module: nn.Module | None = None,
        graph_state: dict[str, torch.Tensor] | None = None,
        runtime_meshes: Sequence[DeviceMesh] = (),
        validate_user_inputs: bool = False,
    ) -> None:
        if traced_result.num_optimizer_state_inputs != 0:
            raise ValueError("GraphRunner does not support traced optimizer state")

        self._traced_result = traced_result
        self._module = module
        self._graph_state = graph_state
        self._runtime_meshes = tuple(runtime_meshes)
        self._validate_user_inputs = validate_user_inputs

        if len(self._runtime_meshes) != traced_result.num_runtime_mesh_inputs:
            raise ValueError(
                "GraphRunner received a different number of runtime meshes than "
                f"during tracing: expected {traced_result.num_runtime_mesh_inputs}, "
                f"got {len(self._runtime_meshes)}"
            )

        self._state_tensors = _resolve_state_tensors(
            traced_result,
            module,
            graph_state,
        )
        self._flat_state_inputs = _flatten_state_inputs(
            traced_result,
            self._state_tensors,
        )
        self._state_signatures = tuple(
            _state_input_signature(value) for value in self._flat_state_inputs
        )
        self._validate_runtime_meshes()

    def _validate_runtime_meshes(self) -> None:
        """Validate runtime meshes against the inputs used to load the graph."""
        if not self._runtime_meshes or not self._traced_result.example_inputs:
            return
        start = self._traced_result.num_static_inputs
        expected = self._traced_result.example_inputs[
            start : start + len(self._runtime_meshes)
        ]
        if len(expected) != len(self._runtime_meshes) or any(
            actual is not traced
            for actual, traced in zip(self._runtime_meshes, expected, strict=True)
        ):
            raise ValueError("runtime meshes differ from the traced graph inputs")

    def validate_state(self) -> None:
        """Fail if bound state objects, subclass leaves, or storage changed."""
        current_state_tensors = _resolve_state_tensors(
            self._traced_result,
            self._module,
            self._graph_state,
        )
        current_flat_state_inputs = _flatten_state_inputs(
            self._traced_result,
            current_state_tensors,
        )

        if any(
            actual is not expected
            for actual, expected in zip(
                current_state_tensors,
                self._state_tensors,
                strict=True,
            )
        ) or any(
            actual is not expected
            for actual, expected in zip(
                current_flat_state_inputs,
                self._flat_state_inputs,
                strict=True,
            )
        ):
            raise RuntimeError("GraphRunner state objects changed after binding")
        if tuple(
            _state_input_signature(value) for value in current_flat_state_inputs
        ) != (self._state_signatures):
            raise RuntimeError("GraphRunner state storage changed after binding")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the graph with dynamic inputs and reconstruct its outputs."""
        user_inputs_flat, runtime_spec = pytree.tree_flatten((args, kwargs))
        if (
            self._validate_user_inputs
            and runtime_spec != self._traced_result.user_inputs_spec
        ):
            raise ValueError(
                f"input spec mismatch: runtime {runtime_spec} != "
                f"trace-time {self._traced_result.user_inputs_spec}"
            )
        if any(isinstance(leaf, nn.Module) for leaf in user_inputs_flat):
            raise ValueError(
                "GraphRunner requires explicit tensor inputs, not nn.Module "
                "instances. Capture nn.Modules in the function closure or bind "
                "them through the module argument."
            )

        flat_user_inputs, _ = _unwrap_subclasses(user_inputs_flat)
        flat_inputs = [
            *self._flat_state_inputs,
            *self._runtime_meshes,
            *flat_user_inputs,
        ]

        with torch.no_grad():
            flat_outputs = self._traced_result.gm(*flat_inputs)
        wrapped = _wrap_subclasses(
            flat_outputs,
            self._traced_result.num_flat_outputs,
            self._traced_result.output_subclass_layouts,
        )
        return pytree.tree_unflatten(wrapped, self._traced_result.output_spec)
