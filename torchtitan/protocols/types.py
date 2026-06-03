# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared type primitives for torchtitan protocols.

Houses small types reused across the protocols package (and its callers)
so individual protocol modules don't grow their own copies.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import spmd_types as spmd
from torch.distributed.tensor import Placement


class StrEnum(str, Enum):
    """str + Enum for Python < 3.11 compatibility."""

    pass


class MeshAxisName(StrEnum):
    """Names for axes of a ``DeviceMesh``.

    Naming convention: throughout torchtitan code, comments, and docstrings
    we say ``axis`` for a ``DeviceMesh`` axis and ``dim`` for a tensor
    dimension. This avoids the ambiguity of ``dim`` referring to both.

    Note that PyTorch upstream's ``DeviceMesh`` API still uses the older
    ``mesh_dim_names`` attribute and ``mesh_dim`` parameter names; we keep
    those exact spellings when calling into PyTorch APIs (we cannot rename
    upstream surface), but use ``axis`` for any name we own.
    """

    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    FSDP = "fsdp"
    TP = "tp"
    CP = "cp"
    PP = "pp"
    EP = "ep"
    EFSDP = "efsdp"


@dataclass(frozen=True, slots=True)
class NamedPlacement:
    """Placement annotations keyed by logical mesh axis name.

    ``placements`` may contain DTensor ``Placement`` values or local-SPMD
    per-axis types. ``partition_spec`` is only used by local-SPMD typechecking
    to disambiguate multi-axis sharding of one tensor dimension.
    """

    placements: dict[MeshAxisName, Placement | spmd.PerMeshAxisSpmdType]
    partition_spec: spmd.PartitionSpec | tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if not self.placements:
            raise ValueError("NamedPlacement requires at least one mesh axis.")
        if self.partition_spec is not None and not isinstance(
            self.partition_spec, tuple
        ):
            raise TypeError(
                f"Expected partition_spec to be a tuple, got "
                f"{self.partition_spec!r}."
            )
        for axis_name, axis_type in self.placements.items():
            if not isinstance(axis_name, MeshAxisName):
                raise TypeError(f"Expected MeshAxisName key, got {axis_name!r}.")
            if not isinstance(axis_type, Placement) and not isinstance(
                axis_type, spmd.PerMeshAxisSpmdType
            ):
                raise TypeError(
                    f"Expected DTensor Placement or SPMD axis type for "
                    f"{axis_name.value!r}, got {axis_type!r}."
                )

    def axes(self) -> tuple[MeshAxisName, ...]:
        return tuple(self.placements)

    def is_spmd(self) -> bool:
        return all(
            isinstance(axis_type, spmd.PerMeshAxisSpmdType)
            for axis_type in self.placements.values()
        )
