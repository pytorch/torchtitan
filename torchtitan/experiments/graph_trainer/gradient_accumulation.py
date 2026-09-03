# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _flat_tensor_ranges,
    SubclassLayout,
    TracedResult,
)


StorageKey = tuple[torch.device, int]


def _storage_keys(tensor: torch.Tensor) -> tuple[StorageKey, ...]:
    if isinstance(tensor, FakeTensor):
        return ()
    if isinstance(tensor, DTensor):
        return _storage_keys(tensor.to_local())
    if is_traceable_wrapper_subclass(tensor):
        attrs, _ = tensor.__tensor_flatten__()
        return tuple(
            storage_key
            for attr in attrs
            if isinstance(inner := getattr(tensor, attr), torch.Tensor)
            for storage_key in _storage_keys(inner)
        )
    if tensor.numel() == 0:
        return ()
    return ((tensor.device, tensor.untyped_storage().data_ptr()),)


@dataclass(frozen=True, slots=True)
class GraphGradientState:
    """Own persistent, optimizer-visible gradients for traced execution.

    Entries at the same index describe one trainable parameter. Each buffer is
    assigned to its parameter's ``grad`` field and passed to the traced graph,
    which accumulates gradients into it in-place.

    Attributes:
        parameter_fqns: Parameter names in graph-state order.
        parameters: Live trainable parameters owned by the optimizer.
        parameter_storage_keys: Parameter storage identities captured at creation.
        buffers: Immutable gradient-buffer tuple in parameter order.
        buffer_storage_keys: Buffer storage identities used to detect replacement.
        graph_state: Ordered parameter-name-to-gradient-buffer mapping.
    """

    parameter_fqns: tuple[str, ...]
    parameters: tuple[torch.Tensor, ...]
    parameter_storage_keys: tuple[tuple[StorageKey, ...], ...]
    buffers: tuple[torch.Tensor, ...]
    buffer_storage_keys: tuple[tuple[StorageKey, ...], ...]
    graph_state: dict[str, torch.Tensor]

    @classmethod
    def create(
        cls,
        model: nn.Module,
        optimizers: Iterable[torch.optim.Optimizer],
    ) -> GraphGradientState:
        cls._validate_module_grad_contracts(model)
        named_parameters = [
            (fqn, parameter)
            for fqn, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ]
        cls._validate_unique_parameters(named_parameters)
        parameters = tuple(parameter for _, parameter in named_parameters)
        cls._validate_optimizer_params_membership(parameters, optimizers)

        buffers = tuple(torch.zeros_like(parameter) for parameter in parameters)
        state = cls(
            parameter_fqns=tuple(fqn for fqn, _ in named_parameters),
            parameters=parameters,
            parameter_storage_keys=tuple(
                _storage_keys(parameter) for parameter in parameters
            ),
            buffers=buffers,
            buffer_storage_keys=tuple(_storage_keys(buffer) for buffer in buffers),
            graph_state={
                fqn: buffer
                for (fqn, _), buffer in zip(named_parameters, buffers, strict=True)
            },
        )
        state.bind_buffers_to_params_grads()
        return state

    @staticmethod
    def _validate_module_grad_contracts(model: nn.Module) -> None:
        for fqn, module in model.named_modules():
            runtime_policy = getattr(module, "_runtime_policy", None)
            if (
                getattr(module, "inplace_wgrad_accum", False) is True
                or getattr(runtime_policy, "inplace_wgrad_accum", False) is True
            ):
                module_fqn = fqn or "<root>"
                raise ValueError(
                    "GraphTrainer in-graph gradient accumulation does not "
                    "support modules that read or mutate parameter.grad during "
                    f"backward; {module_fqn!r} enables inplace_wgrad_accum"
                )

    @staticmethod
    def _validate_unique_parameters(
        named_parameters: Sequence[tuple[str, torch.Tensor]],
    ) -> None:
        names_by_identity: dict[int, str] = {}
        names_by_storage: dict[StorageKey, str] = {}
        for fqn, parameter in named_parameters:
            if is_traceable_wrapper_subclass(parameter) and not isinstance(
                parameter, DTensor
            ):
                raise NotImplementedError(
                    "GraphTrainer in-graph gradient accumulation only supports "
                    "plain tensors and DTensor parameters, got "
                    f"{type(parameter).__name__} for {fqn!r}"
                )
            parameter_id = id(parameter)
            if parameter_id in names_by_identity:
                raise ValueError(
                    "GraphTrainer in-graph gradient accumulation does not support "
                    f"tied parameter {fqn!r}, also registered as "
                    f"{names_by_identity[parameter_id]!r}"
                )
            names_by_identity[parameter_id] = fqn

            for storage_key in _storage_keys(parameter):
                if storage_key in names_by_storage:
                    raise ValueError(
                        "GraphTrainer in-graph gradient accumulation does not "
                        f"support parameters sharing storage: "
                        f"{names_by_storage[storage_key]!r} and {fqn!r}"
                    )
                names_by_storage[storage_key] = fqn

    @staticmethod
    def _validate_optimizer_params_membership(
        parameters: Sequence[torch.Tensor],
        optimizers: Iterable[torch.optim.Optimizer],
    ) -> None:
        optimizer_parameters = [
            parameter
            for optimizer in optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation requires every "
                "parameter to occur in exactly one optimizer parameter group"
            )
        if set(optimizer_ids) != {id(parameter) for parameter in parameters}:
            raise ValueError(
                "GraphTrainer model parameters and optimizer parameters do not match"
            )

    def bind_buffers_to_params_grads(self) -> None:
        """Bind newly allocated buffers to ``parameter.grad``."""
        for fqn, parameter in zip(self.parameter_fqns, self.parameters, strict=True):
            if parameter.grad is not None:
                raise RuntimeError(
                    "GraphTrainer gradient state must be initialized with empty "
                    f"parameter gradients, but {fqn!r} already has a gradient"
                )
        for parameter, buffer in zip(self.parameters, self.buffers, strict=True):
            parameter.grad = buffer

    def validate_parameters(self, parameters: Sequence[torch.Tensor]) -> None:
        """Validate that tracing still uses the parameters bound at creation."""
        if len(parameters) != len(self.parameters) or any(
            actual is not expected
            for actual, expected in zip(parameters, self.parameters, strict=True)
        ):
            raise RuntimeError(
                "GraphTrainer parameters changed after gradient buffers were created"
            )
        self.validate_bindings()

        for fqn, parameter, storage_key in zip(
            self.parameter_fqns,
            self.parameters,
            self.parameter_storage_keys,
            strict=True,
        ):
            if _storage_keys(parameter) != storage_key:
                raise RuntimeError(
                    "GraphTrainer requires stable parameter storage for "
                    f"{fqn!r}; its data pointer changed"
                )

    def validate_optimizers(
        self,
        optimizers: Iterable[torch.optim.Optimizer],
    ) -> None:
        """Validate that optimizers still own exactly the bound parameters."""
        self._validate_optimizer_params_membership(self.parameters, optimizers)

    def validate_bindings(self) -> None:
        """Validate stable optimizer-visible gradient identities."""
        if tuple(self.graph_state) != self.parameter_fqns:
            raise RuntimeError(
                "GraphTrainer gradient-state names or order changed after tracing"
            )
        for fqn, parameter, buffer, storage_key in zip(
            self.parameter_fqns,
            self.parameters,
            self.buffers,
            self.buffer_storage_keys,
            strict=True,
        ):
            if self.graph_state[fqn] is not buffer:
                raise RuntimeError(
                    "GraphTrainer requires a stable graph-state buffer for "
                    f"{fqn!r}; the mapping value was replaced"
                )
            if parameter.grad is not buffer:
                raise RuntimeError(
                    "GraphTrainer requires a stable parameter.grad buffer for "
                    f"{fqn!r}; an optimizer or hook replaced it"
                )
            if _storage_keys(buffer) != storage_key:
                raise RuntimeError(
                    "GraphTrainer requires stable gradient-buffer storage for "
                    f"{fqn!r}; its data pointer changed"
                )


def _subclass_context_without_strides(value: Any) -> Any:
    """Ignore source strides; ``add_`` retains the destination buffer layout."""
    return pytree.tree_map(
        lambda item: item._replace(stride=()) if isinstance(item, TensorMeta) else item,
        value,
        is_leaf=lambda item: isinstance(item, TensorMeta),
    )


def _graph_state_leaf_offsets(
    fqn: str,
    buffer_layout: SubclassLayout | None,
    gradient_layout: SubclassLayout | None,
) -> tuple[tuple[int, ...], int | None]:
    if (buffer_layout is None) != (gradient_layout is None):
        raise ValueError(f"Gradient tensor subclass does not match buffer for {fqn!r}")
    if buffer_layout is None or gradient_layout is None:
        return (0,), None

    buffer_meta = buffer_layout.meta
    gradient_meta = gradient_layout.meta
    if buffer_meta is None or gradient_meta is None:
        raise ValueError(f"Missing tensor-subclass metadata for {fqn!r}")
    if (
        buffer_meta.cls is not gradient_meta.cls
        or buffer_meta.attrs != gradient_meta.attrs
        or _subclass_context_without_strides(buffer_meta.ctx)
        != _subclass_context_without_strides(gradient_meta.ctx)
        or buffer_meta.outer_size != gradient_meta.outer_size
    ):
        raise ValueError(
            "Gradient tensor subclass metadata does not match buffer for "
            f"{fqn!r}: buffer={buffer_meta!r}, gradient={gradient_meta!r}"
        )
    if buffer_layout.num_tensors != gradient_layout.num_tensors:
        raise ValueError(
            f"Gradient tensor subclass leaves do not match buffer for {fqn!r}"
        )
    if not issubclass(buffer_meta.cls, DTensor):
        raise NotImplementedError(
            "GraphTrainer in-graph gradient accumulation only supports plain "
            f"tensors and DTensor subclasses, got {buffer_meta.cls.__name__} "
            f"for {fqn!r}"
        )

    flat_offset = 0
    local_tensor_offset = None
    device_mesh_offset = None
    for attr in buffer_meta.attrs:
        num_tensors, inner_meta = buffer_meta.inner_metas[attr]
        gradient_num_tensors, gradient_inner_meta = gradient_meta.inner_metas[attr]
        if num_tensors != gradient_num_tensors:
            raise ValueError(
                f"Gradient tensor subclass leaves do not match buffer for {fqn!r}"
            )
        if attr == "_local_tensor":
            if (
                num_tensors != 1
                or inner_meta is not None
                or gradient_inner_meta is not None
            ):
                raise NotImplementedError(
                    "GraphTrainer in-graph gradient accumulation requires plain "
                    f"DTensor local tensors for {fqn!r}"
                )
            local_tensor_offset = flat_offset
        elif attr == "device_mesh":
            if (
                num_tensors != 1
                or inner_meta is not None
                or gradient_inner_meta is not None
            ):
                raise NotImplementedError(
                    "GraphTrainer in-graph gradient accumulation requires a plain "
                    f"DTensor device mesh for {fqn!r}"
                )
            device_mesh_offset = flat_offset
        else:
            raise NotImplementedError(
                "GraphTrainer in-graph gradient accumulation does not support "
                f"DTensor wrapper attribute {attr!r} for {fqn!r}"
            )
        flat_offset += num_tensors
    if local_tensor_offset is None:
        raise ValueError(f"DTensor gradient state {fqn!r} has no local tensor")
    if device_mesh_offset is None:
        raise ValueError(f"DTensor gradient state {fqn!r} has no device mesh")
    return (local_tensor_offset,), device_mesh_offset


def _validate_device_mesh_leaf(
    fqn: str,
    buffer: torch.fx.Node,
    gradient: torch.fx.Node,
) -> None:
    buffer_mesh = buffer.meta.get("val")
    gradient_mesh = gradient.meta.get("val")
    if (
        not isinstance(buffer_mesh, DeviceMesh)
        or not isinstance(gradient_mesh, DeviceMesh)
        or buffer_mesh != gradient_mesh
    ):
        raise ValueError(f"DTensor device mesh does not match buffer for {fqn!r}")


def _validate_tensor_leaf(
    fqn: str,
    buffer: torch.fx.Node,
    gradient: torch.fx.Node,
) -> None:
    buffer_value = buffer.meta.get("val")
    gradient_value = gradient.meta.get("val")
    if not isinstance(buffer_value, torch.Tensor) or not isinstance(
        gradient_value, torch.Tensor
    ):
        raise ValueError(f"Missing tensor metadata for gradient state {fqn!r}")
    if (
        buffer_value.shape != gradient_value.shape
        or buffer_value.dtype != gradient_value.dtype
        or buffer_value.device != gradient_value.device
    ):
        raise ValueError(
            "Gradient shape, dtype, and device must match its buffer for "
            f"{fqn!r}; got gradient "
            f"{tuple(gradient_value.shape)}, {gradient_value.dtype}, "
            f"{gradient_value.device} and buffer "
            f"{tuple(buffer_value.shape)}, {buffer_value.dtype}, "
            f"{buffer_value.device}"
        )
    if gradient_value.layout != torch.strided:
        raise NotImplementedError(
            "GraphTrainer in-graph gradient accumulation does not support "
            f"{gradient_value.layout} gradient {fqn!r}"
        )


def _insert_graph_gradient_sink(
    gm: torch.fx.GraphModule,
    *,
    fqn: str,
    buffer: torch.fx.Node,
    gradient: torch.fx.Node,
) -> torch.fx.Node:
    _validate_tensor_leaf(fqn, buffer, gradient)
    sink = gm.graph.call_function(
        torch.ops.aten.add_.Tensor,
        args=(buffer, gradient),
    )
    sink.meta = copy.copy(gradient.meta)
    for key in ("custom", "unbacked_bindings"):
        if isinstance(value := sink.meta.get(key), dict):
            sink.meta[key] = copy.copy(value)
    if "val" in buffer.meta:
        sink.meta["val"] = buffer.meta["val"]
    sink.meta["graph_gradient_fqn"] = fqn
    return sink


def finalize_graph_gradient_accumulation(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    traced_result: TracedResult,
) -> torch.fx.GraphModule:
    """Replace ``[loss, *grads]`` with in-place graph-state accumulation."""
    del example_inputs
    graph_state = traced_result.graph_state
    if graph_state.grad_sink_active:
        return gm
    if not graph_state.mappings:
        raise ValueError("Graph gradient accumulation requires non-empty graph state")
    if not graph_state.has_output_grads:
        raise ValueError("Graph gradient state has no output gradient mappings")
    output = next(node for node in gm.graph.nodes if node.op == "output")
    output_leaves = pytree.tree_leaves(output.args[0])
    if any(not isinstance(leaf, torch.fx.Node) for leaf in output_leaves):
        raise NotImplementedError(
            "GraphTrainer in-graph gradient accumulation requires every "
            "loss and gradient output to be a tensor"
        )
    flat_outputs = cast(list[torch.fx.Node], output_leaves)
    output_ranges = _flat_tensor_ranges(
        traced_result.num_flat_outputs,
        traced_result.output_subclass_layouts,
    )
    expected_num_output_leaves = sum(len(indices) for indices in output_ranges)
    if len(flat_outputs) != expected_num_output_leaves:
        raise ValueError(
            "Graph output metadata changed before gradient finalization: "
            f"expected {expected_num_output_leaves} tensor leaves, got "
            f"{len(flat_outputs)}"
        )

    gradient_output_index_set = {
        mapping.output_grad_index for mapping in graph_state.mappings
    }
    loss_output_indices = [
        output_index
        for output_index in range(traced_result.num_flat_outputs)
        if output_index not in gradient_output_index_set
    ]
    if len(loss_output_indices) != 1:
        raise ValueError(
            "GraphTrainer gradient finalization requires exactly one non-gradient "
            f"loss output, got {len(loss_output_indices)}"
        )

    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    with gm.graph.inserting_before(output):
        for state_index, mapping in enumerate(graph_state.mappings):
            fqn = mapping.fqn
            buffer_indices = mapping.input_indices
            gradient_output_index = mapping.output_grad_index
            assert gradient_output_index is not None
            buffer_logical_index = len(traced_result.state_fqns) + state_index
            gradient_indices = output_ranges[gradient_output_index]
            leaf_offsets, device_mesh_offset = _graph_state_leaf_offsets(
                fqn,
                traced_result.input_subclass_layouts.get(buffer_logical_index),
                traced_result.output_subclass_layouts.get(gradient_output_index),
            )
            if device_mesh_offset is not None:
                _validate_device_mesh_leaf(
                    fqn,
                    placeholders[buffer_indices[device_mesh_offset]],
                    flat_outputs[gradient_indices[device_mesh_offset]],
                )
            for leaf_offset in leaf_offsets:
                buffer = placeholders[buffer_indices[leaf_offset]]
                gradient = flat_outputs[gradient_indices[leaf_offset]]
                _insert_graph_gradient_sink(
                    gm,
                    fqn=fqn,
                    buffer=buffer,
                    gradient=gradient,
                )

    loss_output_index = loss_output_indices[0]
    loss_indices = output_ranges[loss_output_index]
    output.args = ([flat_outputs[index] for index in loss_indices],)
    gm.graph.lint()
    gm.recompile()

    traced_result.num_flat_outputs = 1
    loss_layout = traced_result.output_subclass_layouts.get(loss_output_index)
    traced_result.output_subclass_layouts = (
        {0: loss_layout} if loss_layout is not None else {}
    )
    traced_result.output_spec = pytree.tree_flatten([0])[1]
    traced_result.graph_state = graph_state.activate_grad_sink()
    return gm
