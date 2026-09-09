# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


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
        parameters = tuple(parameter for _, parameter in named_parameters)
        parameter_storage_keys = tuple(
            _storage_keys(parameter) for parameter in parameters
        )
        cls._validate_unique_parameters(named_parameters, parameter_storage_keys)
        cls._validate_optimizer_params_membership(parameters, optimizers)

        buffers = tuple(torch.zeros_like(parameter) for parameter in parameters)
        state = cls(
            parameter_fqns=tuple(fqn for fqn, _ in named_parameters),
            parameters=parameters,
            parameter_storage_keys=parameter_storage_keys,
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
        parameter_storage_keys: Sequence[tuple[StorageKey, ...]],
    ) -> None:
        names_by_identity: dict[int, str] = {}
        names_by_storage: dict[StorageKey, str] = {}
        for (fqn, parameter), storage_keys in zip(
            named_parameters, parameter_storage_keys, strict=True
        ):
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

            for storage_key in storage_keys:
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

    def prepare_for_backward(self) -> None:
        """Clear and rebind retained buffers after ``zero_grad(set_to_none=True)``."""
        if not self.parameters or self.parameters[0].grad is self.buffers[0]:
            return

        for fqn, parameter in zip(self.parameter_fqns, self.parameters, strict=True):
            if parameter.grad is not None:
                raise RuntimeError(
                    "GraphTrainer expected parameter gradients to be cleared "
                    f"before rebinding, but {fqn!r} has a gradient"
                )

        for buffer in self.buffers:
            buffer.zero_()
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
        for fqn, buffer, storage_key in zip(
            self.parameter_fqns,
            self.buffers,
            self.buffer_storage_keys,
            strict=True,
        ):
            if self.graph_state[fqn] is not buffer:
                raise RuntimeError(
                    "GraphTrainer requires a stable graph-state buffer for "
                    f"{fqn!r}; the mapping value was replaced"
                )
            if _storage_keys(buffer) != storage_key:
                raise RuntimeError(
                    "GraphTrainer requires stable gradient-buffer storage for "
                    f"{fqn!r}; its data pointer changed"
                )
        self.validate_grad_bindings()

    def validate_grad_bindings(self) -> None:
        """Validate that parameter gradients still use trainer-owned buffers."""
        for fqn, parameter, buffer in zip(
            self.parameter_fqns,
            self.parameters,
            self.buffers,
            strict=True,
        ):
            if parameter.grad is not buffer:
                raise RuntimeError(
                    "GraphTrainer requires a stable parameter.grad buffer for "
                    f"{fqn!r}; an optimizer or hook replaced it"
                )
