# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    SubclassLayout,
    TracedResult,
)


StorageKey = tuple[torch.device, int, int]


def _storage_keys(tensor: torch.Tensor) -> tuple[StorageKey, ...]:
    if isinstance(tensor, DTensor):
        return _storage_keys(tensor.to_local())
    storage = tensor.untyped_storage()
    return ((tensor.device, storage._cdata, storage.data_ptr()),)


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
        graph_state: Ordered parameter-name-to-gradient-buffer mapping.
        buffer_storage_keys: Buffer storage identities used to detect replacement.
        buffers: Tuple view of the gradient buffers in parameter order.
    """

    parameter_fqns: tuple[str, ...]
    parameters: tuple[torch.Tensor, ...]
    parameter_storage_keys: tuple[tuple[StorageKey, ...], ...]
    graph_state: dict[str, torch.Tensor]
    buffer_storage_keys: tuple[tuple[StorageKey, ...], ...]

    @classmethod
    def create(
        cls,
        model: nn.Module,
        optimizers: Iterable[torch.optim.Optimizer],
    ) -> GraphGradientState:
        named_parameters = [
            (fqn, parameter)
            for fqn, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ]
        cls._validate_unique_parameters(named_parameters)
        parameters = tuple(parameter for _, parameter in named_parameters)
        cls._validate_optimizer_params_membership(parameters, optimizers)

        graph_state = {
            fqn: torch.zeros_like(parameter) for fqn, parameter in named_parameters
        }
        state = cls(
            parameter_fqns=tuple(fqn for fqn, _ in named_parameters),
            parameters=parameters,
            parameter_storage_keys=tuple(
                _storage_keys(parameter) for parameter in parameters
            ),
            graph_state=graph_state,
            buffer_storage_keys=tuple(
                _storage_keys(buffer) for buffer in graph_state.values()
            ),
        )
        state.bind_buffers_to_params_grads()
        return state

    @property
    def buffers(self) -> tuple[torch.Tensor, ...]:
        """Return gradient buffers in parameter order."""
        return tuple(self.graph_state.values())

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
        for parameter, buffer in zip(
            self.parameters, self.graph_state.values(), strict=True
        ):
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
            self.graph_state.values(),
            self.buffer_storage_keys,
            strict=True,
        ):
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


def _flat_tensor_ranges(
    num_values: int,
    layouts: dict[int, SubclassLayout],
) -> tuple[tuple[int, ...], ...]:
    ranges = []
    flat_index = 0
    for logical_index in range(num_values):
        num_tensors = (
            layouts[logical_index].num_tensors if logical_index in layouts else 1
        )
        ranges.append(tuple(range(flat_index, flat_index + num_tensors)))
        flat_index += num_tensors
    return tuple(ranges)


def finalize_graph_gradient_accumulation(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    traced_result: TracedResult,
) -> torch.fx.GraphModule:
    """Replace ``[loss, *grads]`` with in-place graph-state accumulation."""
    del example_inputs
    if not traced_result.graph_state_fqns:
        raise ValueError("Graph gradient accumulation requires non-empty graph state")
    expected_outputs = len(traced_result.graph_state_fqns) + 1
    if traced_result.num_flat_outputs != expected_outputs:
        raise ValueError(
            "GraphTrainer gradient accumulation requires one loss followed by one "
            f"gradient per buffer, got {traced_result.num_flat_outputs} outputs for "
            f"{len(traced_result.graph_state_fqns)} buffers"
        )

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

    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    input_ranges = _flat_tensor_ranges(
        traced_result.num_flat_inputs,
        traced_result.input_subclass_layouts,
    )
    with gm.graph.inserting_before(output):
        for state_index, fqn in enumerate(traced_result.graph_state_fqns):
            buffer_logical_index = len(traced_result.state_fqns) + state_index
            buffer_indices = input_ranges[buffer_logical_index]
            gradient_output_index = state_index + 1
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
                _validate_tensor_leaf(fqn, buffer, gradient)
                inplace_grad_accumulation = gm.graph.call_function(
                    torch.ops.aten.add_.Tensor,
                    args=(buffer, gradient),
                )
                if "val" in buffer.meta:
                    inplace_grad_accumulation.meta["val"] = buffer.meta["val"]

    loss_indices = output_ranges[0]
    output.args = ([flat_outputs[index] for index in loss_indices],)
    gm.graph.lint()
    gm.recompile()

    traced_result.num_flat_outputs = 1
    loss_layout = traced_result.output_subclass_layouts.get(0)
    traced_result.output_subclass_layouts = (
        {0: loss_layout} if loss_layout is not None else {}
    )
    traced_result.output_spec = pytree.tree_flatten([0])[1]
    return gm
