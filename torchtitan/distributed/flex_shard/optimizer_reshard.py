# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public optimizer-reshard configuration and runtime mixin."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast, overload, TypeVar

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.optim import Optimizer

from torchtitan.distributed.parallel_dims import MeshAxisName


__all__ = ["BucketConfig", "ComputeLayout", "FlexOptimizer"]


class _FrozenAxisPlacements(Mapping[MeshAxisName, Replicate | Shard]):
    """Small immutable mapping that remains safe to copy with configs."""

    __slots__ = ("_items",)

    def __init__(
        self,
        placements: Mapping[MeshAxisName, Replicate | Shard],
    ) -> None:
        self._items = tuple(placements.items())

    def __getitem__(self, axis_name: MeshAxisName) -> Replicate | Shard:
        for candidate_axis_name, placement in self._items:
            if candidate_axis_name == axis_name:
                return placement
        raise KeyError(axis_name)

    def __iter__(self) -> Iterator[MeshAxisName]:
        return (axis_name for axis_name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenAxisPlacements:
        return self


@dataclass(frozen=True, slots=True)
class ComputeLayout:
    """Describe a temporary compute layout on named storage-mesh axes.

    Placements override the named axes; omitted axes preserve their storage
    placement. Ownership axes assign each complete logical matrix to one
    participant in their Cartesian group. Extra declarations may target mesh
    variants not used by a particular parameter, but at least one declaration
    must apply when the layout is resolved.
    """

    axis_placements: Mapping[MeshAxisName, Replicate | Shard] = field(
        default_factory=dict
    )
    owner_mesh_axis_names: tuple[MeshAxisName, ...] = ()

    def __post_init__(self) -> None:
        axis_placements = dict(self.axis_placements)
        owner_mesh_axis_names = tuple(self.owner_mesh_axis_names)
        if not axis_placements and not owner_mesh_axis_names:
            raise ValueError("ComputeLayout must declare a placement or owner axis")
        for axis_name, placement in axis_placements.items():
            if type(axis_name) is not MeshAxisName:
                raise ValueError("ComputeLayout placement axes must use MeshAxisName")
            if type(placement) not in (Replicate, Shard):
                raise ValueError("ComputeLayout placements must be Replicate or Shard")
        if any(
            type(axis_name) is not MeshAxisName for axis_name in owner_mesh_axis_names
        ):
            raise ValueError("ComputeLayout owner axes must use MeshAxisName")
        if len(set(owner_mesh_axis_names)) != len(owner_mesh_axis_names):
            raise ValueError("ComputeLayout owner axes must be unique")
        overlapping_axes = set(axis_placements).intersection(owner_mesh_axis_names)
        if overlapping_axes:
            names = sorted(axis_name.value for axis_name in overlapping_axes)
            raise ValueError(
                "ComputeLayout axes cannot have both placement and ownership: "
                f"{names}"
            )

        normalized_axis_placements = dict(
            sorted(axis_placements.items(), key=lambda item: item[0].value)
        )
        normalized_owner_axes = tuple(
            sorted(owner_mesh_axis_names, key=lambda axis_name: axis_name.value)
        )
        object.__setattr__(
            self,
            "axis_placements",
            _FrozenAxisPlacements(normalized_axis_placements),
        )
        object.__setattr__(self, "owner_mesh_axis_names", normalized_owner_axes)


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """Ordered optimizer-work bucket selected by canonical FQN patterns.

    Compute layouts determine communication topology. A bucket controls only
    scheduling order and overlap. FlexShard internally splits its selected
    parameters into adjacent homogeneous physical transport buckets.
    """

    patterns: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", tuple(self.patterns))

    def _bind(
        self,
        mesh: DeviceMesh | None,
        fqns: tuple[str, ...],
    ) -> _BucketSpec:
        return _BucketSpec(
            fqns=fqns,
            mesh=mesh,
            name=self.name,
        )


class FlexOptimizer:
    """Mixin added in-place by ``flex_optimizer_reshard``.

    The dynamically composed class places this mixin before the original
    optimizer class, so FlexShard owns the persistent step lifecycle while the
    original optimizer supplies its compute-specific callbacks. Checkpoint
    through ``state_dict``; whole-optimizer pickling is not supported.
    """

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        optimizer = cast(Any, self)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        optimizer._preflight_step()
        optimizer._redistribution_runtime.run(
            optimizer._bucket_plans,
            local_tensor_spec=optimizer._local_tensor_spec,
            prepare=optimizer._prepare_local,
            compute=optimizer._compute_update,
            finalize=optimizer._apply_update,
            local_bucket_executor=optimizer,
        )
        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        raise RuntimeError("FlexOptimizer parameter groups are frozen")


@dataclass(frozen=True, slots=True)
class _BucketSpec:
    """One resolved physical optimizer-work bucket.

    ``fqns`` are bound from one public ``BucketConfig`` after compute layouts
    select concrete transport groups. ``mesh`` is the bucket's exact
    one-dimensional communication mesh, or ``None`` when every parameter is
    already compute-ready. ``name`` is diagnostic metadata only.
    """

    fqns: tuple[str, ...]
    mesh: DeviceMesh | None
    name: str = ""

    def __post_init__(self) -> None:
        if self.mesh is not None and self.mesh.ndim != 1:
            raise ValueError("bucket mesh must be one-dimensional")
        object.__setattr__(self, "fqns", tuple(self.fqns))


_OptimizerT = TypeVar("_OptimizerT", bound=Optimizer)
_flex_optimizer_classes: dict[type, type] = {}


def _attach_flex_optimizer(optimizer: _OptimizerT) -> _OptimizerT:
    """Add the FlexOptimizer mixin without replacing the optimizer object."""
    if isinstance(optimizer, FlexOptimizer):
        raise ValueError("flex_optimizer_reshard cannot be applied more than once")

    original_class = optimizer.__class__
    flex_optimizer_class = _flex_optimizer_classes.get(original_class)
    if flex_optimizer_class is None:
        flex_optimizer_class = type(
            f"FlexOptimizer{original_class.__name__}",
            (FlexOptimizer, original_class),
            {},
        )
        _flex_optimizer_classes[original_class] = flex_optimizer_class
    optimizer.__class__ = flex_optimizer_class
    # Optimizer patches step on the class during construction. The dynamic
    # class is created afterward, so install the standard profiling/hooks
    # wrapper around FlexOptimizer.step now.
    optimizer._patch_step_function()
    return optimizer
