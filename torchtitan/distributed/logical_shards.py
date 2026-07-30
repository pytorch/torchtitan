# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from operator import index as operator_index
from typing import cast

import torch
from torch.distributed.checkpoint import CheckpointableTensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Replicate,
    Shard,
)


__all__ = [
    "LocalLogicalTensor",
    "LogicalShardLayout",
    "normalize_logical_tensor",
]


@dataclass(frozen=True, slots=True)
class LogicalShardLayout:
    """Mapping from local tensor rectangles to one logical global tensor."""

    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class LocalLogicalTensor:
    """A physical local tensor and its logical global shard layout."""

    local_tensor: torch.Tensor
    layout: LogicalShardLayout


def _as_int_tuple(values, *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(operator_index(value) for value in values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of integers") from error
    return result


def _as_rectangles(values, *, name: str) -> tuple[tuple[int, ...], ...]:
    try:
        return tuple(
            _as_int_tuple(rectangle, name=f"{name}[{index}]")
            for index, rectangle in enumerate(values)
        )
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of integer sequences") from error


def _rectangles_overlap(
    first_offset: tuple[int, ...],
    first_size: tuple[int, ...],
    second_offset: tuple[int, ...],
    second_size: tuple[int, ...],
) -> bool:
    return all(
        first_start < second_start + second_length
        and second_start < first_start + first_length
        for first_start, first_length, second_start, second_length in zip(
            first_offset,
            first_size,
            second_offset,
            second_size,
            strict=True,
        )
    )


def _normalize_layout(
    local_tensor: torch.Tensor,
    *,
    global_shape,
    global_offsets,
    local_offsets,
    local_sizes,
) -> LogicalShardLayout:
    normalized_global_shape = _as_int_tuple(global_shape, name="global_shape")
    normalized_global_offsets = _as_rectangles(global_offsets, name="global_offsets")
    normalized_local_offsets = _as_rectangles(local_offsets, name="local_offsets")
    normalized_local_sizes = _as_rectangles(local_sizes, name="local_sizes")
    local_shape = tuple(local_tensor.shape)
    ndim = len(normalized_global_shape)

    if any(size < 0 for size in normalized_global_shape):
        raise ValueError("global_shape dimensions must be non-negative")
    if len(local_shape) != ndim:
        raise ValueError(
            "logical global shape and physical local tensor must have the same "
            f"number of dimensions, but got {ndim} and {len(local_shape)}"
        )

    num_rectangles = len(normalized_global_offsets)
    if len(normalized_local_offsets) != num_rectangles:
        raise ValueError("global_offsets and local_offsets must have the same length")
    if len(normalized_local_sizes) != num_rectangles:
        raise ValueError("global_offsets and local_sizes must have the same length")

    rectangles: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    for index, (global_offset, local_offset, local_size) in enumerate(
        zip(
            normalized_global_offsets,
            normalized_local_offsets,
            normalized_local_sizes,
            strict=True,
        )
    ):
        for field_name, value in (
            ("global_offsets", global_offset),
            ("local_offsets", local_offset),
            ("local_sizes", local_size),
        ):
            if len(value) != ndim:
                raise ValueError(f"{field_name}[{index}] must have {ndim} dimensions")

        for dim, (offset, size, global_dim) in enumerate(
            zip(global_offset, local_size, normalized_global_shape, strict=True)
        ):
            if offset < 0 or size < 0 or offset + size > global_dim:
                raise ValueError(
                    f"global rectangle {index} dimension {dim} is outside "
                    "global_shape"
                )
        for dim, (offset, size, local_dim) in enumerate(
            zip(local_offset, local_size, local_shape, strict=True)
        ):
            if offset < 0 or offset + size > local_dim:
                raise ValueError(
                    f"local rectangle {index} dimension {dim} is outside the "
                    "physical local tensor"
                )

        # A rectangle with an empty dimension owns no logical elements. Removing
        # it gives every empty owner one canonical representation: zero rectangles.
        if any(size == 0 for size in local_size):
            continue
        rectangles.append((global_offset, local_offset, local_size))

    for first in range(len(rectangles)):
        first_global, first_local, first_size = rectangles[first]
        for second in range(first + 1, len(rectangles)):
            second_global, second_local, second_size = rectangles[second]
            if _rectangles_overlap(
                first_global, first_size, second_global, second_size
            ):
                raise ValueError(
                    f"global rectangles {first} and {second} must not overlap"
                )
            if _rectangles_overlap(first_local, first_size, second_local, second_size):
                raise ValueError(
                    f"local rectangles {first} and {second} must not overlap"
                )

    return LogicalShardLayout(
        global_shape=normalized_global_shape,
        global_offsets=tuple(rectangle[0] for rectangle in rectangles),
        local_offsets=tuple(rectangle[1] for rectangle in rectangles),
        local_sizes=tuple(rectangle[2] for rectangle in rectangles),
    )


def _normalize_dtensor(tensor: DTensor) -> LocalLogicalTensor:
    for placement in tensor.placements:
        if isinstance(placement, Partial):
            raise ValueError(
                "logical shard metadata does not support DTensor Partial placements"
            )
        if isinstance(placement, _StridedShard):
            raise ValueError(
                "logical shard metadata does not yet support DTensor "
                "_StridedShard placements"
            )
        if not isinstance(placement, (Shard, Replicate)):
            raise ValueError(
                "logical shard metadata only supports DTensor Shard and "
                f"Replicate placements, but got {placement!r}"
            )

    if tensor.device_mesh.get_coordinate() is None:
        raise ValueError(
            "logical shard metadata requires the current rank to belong to the "
            "DTensor device mesh"
        )

    local_tensor = tensor.to_local()
    global_shape = tuple(tensor.shape)
    local_shape, global_offset = compute_local_shape_and_global_offset(
        global_shape,
        tensor.device_mesh,
        tensor.placements,
    )
    local_shape = tuple(local_shape)
    if tuple(local_tensor.shape) != local_shape:
        raise ValueError(
            "DTensor physical local shape does not match its unpadded logical "
            f"shard shape: got {tuple(local_tensor.shape)} and {local_shape}"
        )

    owns_elements = all(size > 0 for size in local_shape)
    zeros = (0,) * len(global_shape)
    layout = _normalize_layout(
        local_tensor,
        global_shape=global_shape,
        global_offsets=(tuple(global_offset),) if owns_elements else (),
        local_offsets=(zeros,) if owns_elements else (),
        local_sizes=(local_shape,) if owns_elements else (),
    )
    return LocalLogicalTensor(local_tensor=local_tensor, layout=layout)


def _normalize_checkpointable_tensor(
    tensor: torch.Tensor,
) -> LocalLogicalTensor:
    metadata = cast(CheckpointableTensor, tensor)
    layout = _normalize_layout(
        tensor,
        global_shape=metadata.global_shape,
        global_offsets=metadata.global_offsets,
        local_offsets=metadata.local_offsets,
        local_sizes=metadata.local_sizes,
    )
    return LocalLogicalTensor(local_tensor=tensor, layout=layout)


def normalize_logical_tensor(tensor: torch.Tensor) -> LocalLogicalTensor:
    """Normalize a dense, DTensor, or CheckpointableTensor shard layout."""

    if isinstance(tensor, DTensor):
        return _normalize_dtensor(tensor)

    checkpoint_fields = (
        "global_shape",
        "global_offsets",
        "local_offsets",
        "local_sizes",
    )
    present_fields = tuple(
        field for field in checkpoint_fields if hasattr(tensor, field)
    )
    if present_fields and len(present_fields) != len(checkpoint_fields):
        missing_fields = tuple(
            field for field in checkpoint_fields if field not in present_fields
        )
        raise ValueError(
            "tensor has incomplete CheckpointableTensor metadata; missing "
            f"{missing_fields}"
        )
    if len(present_fields) == len(checkpoint_fields):
        return _normalize_checkpointable_tensor(tensor)

    if hasattr(tensor, "_placements") or hasattr(tensor, "_global_shape"):
        raise ValueError(
            "distributed local tensors must expose public CheckpointableTensor "
            "rectangle metadata; private placement metadata is insufficient"
        )

    shape = tuple(tensor.shape)
    zeros = (0,) * tensor.ndim
    owns_elements = all(size > 0 for size in shape)
    layout = _normalize_layout(
        tensor,
        global_shape=shape,
        global_offsets=(zeros,) if owns_elements else (),
        local_offsets=(zeros,) if owns_elements else (),
        local_sizes=(shape,) if owns_elements else (),
    )
    return LocalLogicalTensor(local_tensor=tensor, layout=layout)
