# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.utils._pytree import tree_map_only

from torchtitan.components.checkpointer.base import OPTIMIZER
from torchtitan.components.checkpointer.utils import canonical_fqn

if TYPE_CHECKING:
    import torch.nn as nn


class OptimizerState(Protocol):
    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...


class DCPStateDictAdapter(ABC):
    """Convert runtime state into and out of its DCP representation."""

    @abstractmethod
    def to_dcp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def from_dcp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        pass


def _logical_state_shapes(model_parts: list[nn.Module]) -> dict[str, tuple[int, ...]]:
    tracked_states = []
    for model in model_parts:
        for module_fqn, module in model.named_modules():
            local_shapes = getattr(module, "_spmd_logical_state_shapes", None)
            if local_shapes is not None:
                tracked_states.append((module_fqn, local_shapes))
    if not tracked_states:
        return {}

    logical_shapes: dict[str, tuple[int, ...]] = {}
    for module_fqn, local_shapes in tracked_states:
        prefix = f"{module_fqn}." if module_fqn else ""
        for state_name, logical_shape in local_shapes.items():
            fqn = canonical_fqn(f"{prefix}{state_name}")
            previous = logical_shapes.setdefault(fqn, logical_shape)
            if previous != logical_shape:
                raise ValueError(
                    f"Conflicting logical shapes for state {fqn!r}: "
                    f"{previous} and {logical_shape}."
                )
    return logical_shapes


class PaddedDTensorStateDictAdapter(DCPStateDictAdapter):
    """Exclude padded SPMD parameter regions from native DCP checkpoints."""

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizers: OptimizerState | None,
    ) -> None:
        self.logical_shapes = _logical_state_shapes(model_parts)
        self.optimizers = optimizers

    @property
    def is_needed(self) -> bool:
        return bool(self.logical_shapes)

    @staticmethod
    def _checkpoint_tensor(
        tensor: DTensor,
        logical_shape: tuple[int, ...],
    ) -> torch.Tensor:
        local_tensor = tensor.to_local().detach()
        local_shape, global_offset = compute_local_shape_and_global_offset(
            tensor.shape,
            tensor.device_mesh,
            tensor.placements,
        )
        if tuple(local_tensor.shape) != tuple(local_shape):
            raise ValueError(
                "DCP cannot map padded DTensor storage with local shape "
                f"{tuple(local_tensor.shape)} to its runtime shard shape "
                f"{tuple(local_shape)}."
            )
        if len(logical_shape) != tensor.ndim or any(
            logical_dim > padded_dim
            for logical_dim, padded_dim in zip(logical_shape, tensor.shape, strict=True)
        ):
            raise ValueError(
                f"Logical shape {logical_shape} is incompatible with padded "
                f"DTensor shape {tuple(tensor.shape)}."
            )

        valid_shape = tuple(
            max(0, min(offset + size, logical_dim) - offset)
            for offset, size, logical_dim in zip(
                global_offset,
                local_shape,
                logical_shape,
                strict=True,
            )
        )
        with torch.no_grad():
            if any(size == 0 for size in valid_shape):
                local_tensor.zero_()
            else:
                for dim, (valid_size, padded_size) in enumerate(
                    zip(valid_shape, local_shape, strict=True)
                ):
                    if valid_size == padded_size:
                        continue
                    index = [slice(None)] * local_tensor.ndim
                    index[dim] = slice(valid_size, None)
                    local_tensor[tuple(index)].zero_()

        # Preserve the padded DTensor's physical chunk offset. Rebuilding a
        # DTensor with the logical shape would recompute FSDP shard boundaries
        # and could assign the existing local values to different offsets.
        checkpoint_tensor = local_tensor.as_strided(
            local_tensor.shape,
            local_tensor.stride(),
        )
        checkpoint_tensor_with_metadata: Any = checkpoint_tensor
        checkpoint_tensor_with_metadata.global_shape = logical_shape
        if any(size == 0 for size in valid_shape):
            checkpoint_tensor_with_metadata.global_offsets = ()
            checkpoint_tensor_with_metadata.local_offsets = ()
            checkpoint_tensor_with_metadata.local_sizes = ()
        else:
            checkpoint_tensor_with_metadata.global_offsets = (tuple(global_offset),)
            checkpoint_tensor_with_metadata.local_offsets = ((0,) * tensor.ndim,)
            checkpoint_tensor_with_metadata.local_sizes = (valid_shape,)
        checkpoint_tensor_with_metadata._torchtitan_runtime_dtensor = tensor
        return checkpoint_tensor

    def _convert_tensor(
        self,
        tensor: Any,
        logical_shape: tuple[int, ...],
    ) -> Any:
        if not isinstance(tensor, DTensor) or tuple(tensor.shape) == logical_shape:
            return tensor
        return self._checkpoint_tensor(tensor, logical_shape)

    def _optimizer_logical_shape(self, key: str) -> tuple[int, ...] | None:
        for fqn, logical_shape in self.logical_shapes.items():
            if key.startswith(f"state.{fqn}."):
                return logical_shape
        return None

    def to_dcp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        converted = dict(state_dict)
        missing = self.logical_shapes.keys() - converted.keys()
        if missing:
            raise ValueError(
                "Unevenly sharded states are not exposed by the model state dict: "
                f"{sorted(missing)}. Their state-dict hooks must also expose "
                "logical-shape metadata."
            )
        for fqn, logical_shape in self.logical_shapes.items():
            tensor = converted[fqn]
            if not isinstance(tensor, DTensor) and tuple(tensor.shape) != logical_shape:
                raise ValueError(
                    f"Unevenly sharded state {fqn!r} must be a DTensor before "
                    f"DCP conversion, got {type(tensor).__name__}."
                )
            converted[fqn] = self._convert_tensor(tensor, logical_shape)

        optimizer = converted.get(OPTIMIZER)
        if self.optimizers is not None and optimizer is self.optimizers:
            optimizer_state = self.optimizers.state_dict()
            converted_optimizer = dict(optimizer_state)
            for key, value in optimizer_state.items():
                logical_shape = self._optimizer_logical_shape(key)
                if logical_shape is not None:
                    converted_optimizer[key] = self._convert_tensor(
                        value, logical_shape
                    )
            converted[OPTIMIZER] = converted_optimizer
        return converted

    def from_dcp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        def restore_runtime_dtensor(tensor: torch.Tensor) -> torch.Tensor:
            return getattr(tensor, "_torchtitan_runtime_dtensor", tensor)

        converted = tree_map_only(
            torch.Tensor,
            restore_runtime_dtensor,
            state_dict,
        )
        if (
            self.optimizers is not None
            and OPTIMIZER in converted
            and isinstance(converted[OPTIMIZER], dict)
        ):
            self.optimizers.load_state_dict(converted[OPTIMIZER])
            converted[OPTIMIZER] = self.optimizers
        return converted
