# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DistMuon configuration, construction, and optimizer implementation."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast, NoReturn, overload

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.optim import Optimizer

from ._optimizer_reshard_runtime import _BucketedRedistributionRuntime

from ._optimizer_reshard_schedule import (
    _bind_bucket_configs,
    _BucketPlanningContext,
    _build_bucket_plans,
    _build_dim0_shard_redistribution_plan,
    _build_owned_redistribution_plan,
    _device_mesh_ranks,
    _dtensor_storage_region_for_participant,
    _dtensor_storage_regions,
    _LocalBucketPlan,
    _ParticipantPartition,
    _RedistributionGroup,
    _RedistributionPlan,
    _require_valid_plan,
    _RouteEndpoint,
    _StorageRegionMapping,
    _TensorRegion,
    _TensorRegionRoute,
    _validate_bucket_plans_across_ranks,
)
from .optimizer_reshard import (
    _BucketSpec,
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)


__all__ = [
    "build_dist_muon",
]


def build_dist_muon(
    params: Iterable[dict[str, Any]],
    *,
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    bucket_configs: Sequence[BucketConfig],
    **kwargs: Any,
) -> DistMuon:
    """Construct a DistMuon optimizer with FlexShard redistribution.

    DistMuon's ``BlockShard`` path accepts only a 2D parameter stored as
    contiguous row-concatenated matrices. The placement must target tensor
    dimension 0. ``block_sizes=(R,)`` describes ``[M * R, C]`` storage;
    longer tuples repeat independently shardable matrix row counts. The leading
    dimension must be nonzero and contain whole repetitions. A native matrix
    batch ``[..., R, C]`` uses ``Shard(0)`` to distribute its
    outermost batch dimension, or ``Owned`` to assign the complete batch to one
    rank.
    Replicated storage and storage shards along either matrix dimension can
    redistribute to ``Shard(0)`` compute on one mesh axis.
    A single 2D matrix without ``BlockShard`` uses whole-matrix compute such as
    ``Owned``.
    """
    return DistMuon(
        _normalize_param_groups(params),
        compute_sharding_by_fqn=compute_sharding_by_fqn,
        bucket_configs=bucket_configs,
        **kwargs,
    )


def _normalize_param_groups(
    params: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy parameter groups and materialize their aligned values."""
    normalized_param_groups = []
    for param_group in params:
        normalized_group = dict(param_group)
        params_value = normalized_group.get("params", ())
        normalized_params = (
            (params_value,) if isinstance(params_value, Tensor) else tuple(params_value)
        )
        param_names_value = normalized_group.get("param_names")
        normalized_param_names = (
            () if param_names_value is None else tuple(param_names_value)
        )
        if param_names_value is None or len(normalized_params) != len(
            normalized_param_names
        ):
            raise ValueError("params and param_names must be aligned")
        normalized_group["params"] = normalized_params
        normalized_group["param_names"] = normalized_param_names
        normalized_param_groups.append(normalized_group)

    return normalized_param_groups


def _validate_compute_sharding_configuration(
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
) -> None:
    """Validate per-parameter compute sharding configuration."""
    for fqn, compute_layout in compute_sharding_by_fqn.items():
        if not isinstance(fqn, str):
            raise ValueError("compute_sharding_by_fqn keys must be strings")
        if type(compute_layout) is not ComputeLayout:
            raise ValueError("compute_sharding_by_fqn values must be ComputeLayout")


def _initialize_dist_muon(
    optimizer: DistMuon,
    *,
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    bucket_configs: Sequence[BucketConfig],
) -> None:
    """Initialize FlexShard for one newly constructed DistMuon.

    Every group must provide aligned ``params`` and ``param_names``. Every
    local parameter FQN must have one entry in ``compute_sharding_by_fqn``;
    extra compute-sharding entries for parameters on other pipeline stages are
    ignored. Parameter groups, compute layouts, and bucket configuration are
    frozen because optimizer state and collectives depend on them.
    """
    _validate_compute_sharding_configuration(compute_sharding_by_fqn)

    for param_group in optimizer.param_groups:
        group_params = tuple(param_group["params"])
        raw_param_names = param_group.get("param_names")
        param_names = () if raw_param_names is None else tuple(raw_param_names)
        if raw_param_names is None or len(group_params) != len(param_names):
            raise ValueError("params and param_names must be aligned")
        for param, fqn in zip(group_params, param_names, strict=True):
            if fqn not in compute_sharding_by_fqn:
                raise ValueError(f"missing compute sharding for Muon parameter {fqn!r}")

    tensor_device = optimizer._validate_parameter_storage()
    compute_layouts = optimizer._build_parameter_compute_layouts(
        compute_sharding_by_fqn
    )
    optimizer._specs = _bind_bucket_configs(
        tuple(bucket_configs),
        compute_layouts,
        get_fqn=lambda layout: layout.fqn,
        get_storage_dtensor=lambda layout: layout.param,
        requires_redistribution=lambda layout: (not layout.storage_is_compute_ready),
        get_redistribution_storage_mesh_axis=lambda layout: (
            layout.redistribution_storage_mesh_axis
        ),
    )
    optimizer._initialize_plan(compute_layouts)
    optimizer._validate_plan_across_ranks()
    optimizer._redistribution_runtime = _BucketedRedistributionRuntime[
        _ParameterComputeLayout
    ](tensor_device)
    optimizer._redistribution_runtime.reserve_buffers(
        optimizer._bucket_plans,
        local_tensor_spec=optimizer._local_tensor_spec,
    )
    optimizer.register_load_state_dict_post_hook(_after_load_state_dict, prepend=True)


class DistMuon(Optimizer):
    """Muon optimizer constructed by ``build_dist_muon``.

    Parameter groups, FQNs, storage layouts, compute layouts, and bucket plans
    are frozen after resharding is applied. Every configured parameter must
    have a layout-compatible DTensor gradient before each rank enters
    ``step()``.

    Matrix-batch compute views use batched BF16 kernels. They implement the
    same mathematical update as ``torch.optim.Muon`` running one matrix at a
    time, but bitwise equality across the two kernel schedules is not part of
    the contract.
    """

    _specs: tuple[_BucketSpec, ...]
    _matrix_views_by_fqn: dict[str, tuple[_MatrixBatchView, ...]]
    _redistribution_runtime: _BucketedRedistributionRuntime[_ParameterComputeLayout]
    _param_groups_frozen: bool

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        compute_sharding_by_fqn: Mapping[str, ComputeLayout],
        bucket_configs: Sequence[BucketConfig],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        self._first_step_validated = False
        self._param_groups_frozen = False
        super().__init__(params, defaults)
        self._validate_groups()
        self._param_groups_frozen = True
        _initialize_dist_muon(
            self,
            compute_sharding_by_fqn=compute_sharding_by_fqn,
            bucket_configs=bucket_configs,
        )

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._preflight_step()
        self._redistribution_runtime.run(
            self._bucket_plans,
            local_tensor_spec=self._local_tensor_spec,
            prepare=self._prepare_local,
            compute=self._compute_update,
            finalize=self._apply_update,
        )
        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._param_groups_frozen:
            raise RuntimeError("DistMuon parameter groups are frozen")
        super().add_param_group(param_group)

    def _validate_groups(self) -> None:
        if len(self.param_groups) != 1:
            raise ValueError("DistMuon requires exactly one parameter group")
        for group_index, group in enumerate(self.param_groups):
            ns_steps = group["ns_steps"]
            coefficients = group["ns_coefficients"]
            if (
                group.get("fused")
                or group.get("foreach")
                or any(
                    not 0 <= group[name]
                    for name in ("lr", "weight_decay", "momentum", "eps")
                )
                or not isinstance(ns_steps, int)
                or not 0 <= ns_steps < 100
                or len(coefficients) != 3
                or not all(isinstance(value, (int, float)) for value in coefficients)
                or group["adjust_lr_fn"]
                not in (None, "original", "match_rms_adamw", "spectral_unclamped")
            ):
                raise ValueError(f"unsupported DistMuon group {group_index}")

    def _validate_parameter_storage(self) -> torch.device:
        local_devices = set()
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, DTensor):
                    raise TypeError("DistMuon requires DTensor parameters")
                local_device = param.to_local().device
                local_devices.add(local_device)
        if len(local_devices) != 1:
            raise ValueError("DistMuon requires one device per process")
        return local_devices.pop()

    def _build_parameter_compute_layouts(
        self,
        compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    ) -> tuple[_ParameterComputeLayout, ...]:
        parameters = []
        seen_names = set()
        seen_params = set()
        for group_index, group in enumerate(self.param_groups):
            params = group["params"]
            names = group["param_names"]
            for fqn, param in zip(names, params, strict=True):
                if fqn in seen_names or id(param) in seen_params:
                    raise ValueError(f"duplicate Muon parameter {fqn!r}")
                seen_names.add(fqn)
                seen_params.add(id(param))
                parameters.append((group_index, fqn, param))

        compute_layouts = []
        for group_index, fqn, param in parameters:
            resolved_transition = _resolve_storage_to_compute_transition(
                fqn,
                param,
                compute_sharding_by_fqn[fqn],
            )
            compute_layouts.append(
                _ParameterComputeLayout(
                    fqn=fqn,
                    param=param,
                    group_index=group_index,
                    storage_mesh_ranks=_device_mesh_ranks(param.device_mesh),
                    storage_layout_signature=_storage_layout_signature(param),
                    local_storage_signature=_local_storage_signature(param.to_local()),
                    compute_sharding=resolved_transition.compute_sharding,
                    storage_to_compute_transition=resolved_transition.storage_to_compute_transition,
                    resolved_compute_layout_signature=(
                        resolved_transition.resolved_compute_layout_signature
                    ),
                    redistribution_storage_mesh_axis=(
                        resolved_transition.redistribution_storage_mesh_axis
                    ),
                )
            )
        return tuple(compute_layouts)

    def _initialize_plan(
        self,
        compute_layouts: Sequence[_ParameterComputeLayout],
    ) -> None:
        ns_steps_by_group = tuple(group["ns_steps"] for group in self.param_groups)
        result = _build_bucket_plans(
            compute_layouts,
            self._specs,
            get_fqn=lambda item: item.fqn,
            get_storage_dtensor=lambda item: item.param,
            requires_redistribution=lambda item: (not item.storage_is_compute_ready),
            resolve_redistribution_plans=partial(
                _resolve_muon_redistribution_plans,
                ns_steps_by_group=ns_steps_by_group,
            ),
        )

        def build_views(
            layout: _ParameterComputeLayout,
            compute_partition: _ParticipantPartition | None,
        ) -> tuple[_MatrixBatchView, ...]:
            param = layout.param
            compute_shape = (
                param.to_local().shape
                if compute_partition is None
                else torch.Size(compute_partition.tensor_shape)
            )
            compute_sharding = layout.compute_sharding
            if type(compute_sharding) is BlockShard:
                if compute_partition is None:
                    region = _dtensor_storage_region_for_participant(
                        param, param.device_mesh.get_rank()
                    )
                else:
                    (region,) = compute_partition.logical_regions
                assert region.shape == tuple(compute_shape) and region.offsets[1] == 0
                return _matrix_batch_views_from_shape(
                    compute_shape,
                    matrix_row_sizes=compute_sharding.block_sizes,
                    logical_row_start=region.offsets[0],
                )
            if not compute_shape.numel():
                return ()
            return (
                _MatrixBatchView(
                    shape=compute_shape,
                    strides=tuple(
                        math.prod(compute_shape[dim + 1 :])
                        for dim in range(len(compute_shape))
                    ),
                    offset=0,
                ),
            )

        matrix_views_by_fqn = {}
        for bucket in result.plans:
            if isinstance(bucket, _LocalBucketPlan):
                for item in bucket.items:
                    matrix_views_by_fqn[item.fqn] = build_views(item, None)
            else:
                for item in bucket.unredistributed_items:
                    matrix_views_by_fqn[item.fqn] = build_views(item, None)
                for item, plan in zip(
                    bucket.redistributed_items,
                    bucket.redistribution_plans,
                    strict=True,
                ):
                    matrix_views_by_fqn[item.fqn] = build_views(
                        item, plan.compute_partition(bucket.group.local_participant)
                    )
        self._bucket_plans = result.plans
        self._parameter_compute_layouts = result.ordered_items
        self._matrix_views_by_fqn = matrix_views_by_fqn

    def _validate_plan_across_ranks(self) -> None:
        _validate_bucket_plans_across_ranks(
            self._bucket_plans,
            item_signature=self._plan_item_signature,
        )

    def _plan_item_signature(
        self, compute_layout: _ParameterComputeLayout
    ) -> tuple[Any, ...]:
        return (
            compute_layout.fqn,
            tuple(compute_layout.param.shape),
            tuple(compute_layout.param.stride()),
            str(compute_layout.param.dtype),
            compute_layout.param.to_local().device.type,
            compute_layout.storage_is_compute_ready,
            compute_layout.compute_sharding,
            compute_layout.resolved_compute_layout_signature,
            _device_mesh_ranks(compute_layout.param.device_mesh),
            tuple(map(str, compute_layout.param.placements)),
            self._group_signature(compute_layout),
        )

    def _group(self, compute_layout: _ParameterComputeLayout) -> dict[str, Any]:
        return self.param_groups[compute_layout.group_index]

    def _group_signature(
        self, compute_layout: _ParameterComputeLayout
    ) -> tuple[Any, ...]:
        group = self._group(compute_layout)
        return tuple(
            group[key]
            for key in (
                "lr",
                "weight_decay",
                "momentum",
                "nesterov",
                "ns_coefficients",
                "eps",
                "ns_steps",
                "adjust_lr_fn",
            )
        )

    def _preflight_step(self) -> None:
        """Fail the local worker before bucket communication on invalid input.

        TorchTitan's elastic launcher terminates peer workers after this error
        escapes. Do not add a validation collective to the optimizer hot path.
        """
        initialize_state = not self._first_step_validated
        missing_gradients = []
        changed_parameter_storage_fqn = None
        changed_gradient_storage_fqn = None
        gradients = [] if initialize_state else None
        for compute_layout in self._parameter_compute_layouts:
            if (
                changed_parameter_storage_fqn is None
                and compute_layout.storage_is_compute_ready
                and _local_storage_signature(compute_layout.param.to_local())
                != compute_layout.local_storage_signature
            ):
                changed_parameter_storage_fqn = compute_layout.fqn

            grad = compute_layout.param.grad
            if grad is None:
                missing_gradients.append(compute_layout.fqn)
            elif not isinstance(grad, DTensor) or not self._has_storage_layout(
                grad, compute_layout
            ):
                if changed_gradient_storage_fqn is None:
                    changed_gradient_storage_fqn = compute_layout.fqn
            elif gradients is not None:
                gradients.append((compute_layout, grad))

        if missing_gradients:
            raise RuntimeError(
                "DistMuon requires every configured gradient before "
                f"step(); missing gradients: {missing_gradients}"
            )
        if changed_parameter_storage_fqn is not None:
            raise RuntimeError(
                f"parameter local storage changed for "
                f"{changed_parameter_storage_fqn!r}; "
                "rebuild DistMuon"
            )
        if changed_gradient_storage_fqn is not None:
            raise RuntimeError(
                f"gradient storage layout changed for {changed_gradient_storage_fqn!r}"
            )

        if gradients is not None:
            for compute_layout, _grad in gradients:
                self._validate_momentum(compute_layout)

        # State creation happens only after every gradient and existing state
        # tensor has passed validation, so a deterministic input error cannot
        # partially update an earlier bucket.
        if gradients is not None:
            for compute_layout, grad in gradients:
                self._momentum(compute_layout, grad)
            self._first_step_validated = True

    @staticmethod
    def _has_storage_layout(
        tensor: DTensor, compute_layout: _ParameterComputeLayout
    ) -> bool:
        mesh_matches = (
            tensor.device_mesh is compute_layout.param.device_mesh
            or _device_mesh_ranks(tensor.device_mesh)
            == compute_layout.storage_mesh_ranks
        )
        return mesh_matches and (
            _storage_layout_signature(tensor) == compute_layout.storage_layout_signature
        )

    def _validate_momentum(self, compute_layout: _ParameterComputeLayout) -> None:
        momentum = self.state.get(compute_layout.param, {}).get("momentum_buffer")
        if momentum is None:
            return
        if not isinstance(momentum, DTensor) or not self._has_storage_layout(
            momentum, compute_layout
        ):
            raise RuntimeError(
                f"momentum storage layout changed for {compute_layout.fqn!r}"
            )

    def _momentum(
        self, compute_layout: _ParameterComputeLayout, grad: DTensor
    ) -> DTensor:
        state = self.state[compute_layout.param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(
                grad, memory_format=torch.preserve_format
            )
        return state["momentum_buffer"]

    def _local_gradient_and_momentum(
        self, compute_layout: _ParameterComputeLayout
    ) -> tuple[Tensor, Tensor, DTensor, dict[str, Any]]:
        grad = cast(DTensor, compute_layout.param.grad)
        momentum_state = cast(
            DTensor,
            self.state[compute_layout.param]["momentum_buffer"],
        )
        local_reference = compute_layout.param.to_local()
        if compute_layout.storage_is_compute_ready:
            local_reference = local_reference.detach()
        local_grad = grad.to_local().view_as(local_reference)
        local_momentum = momentum_state.to_local().view_as(local_reference)
        group = self._group(compute_layout)
        return local_grad, local_momentum, momentum_state, group

    def _prepare_local(
        self, compute_layout: _ParameterComputeLayout, out: Tensor
    ) -> None:
        grad, momentum, momentum_state, group = self._local_gradient_and_momentum(
            compute_layout
        )
        _prepare_muon_input(
            grad,
            momentum,
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            out=out,
        )
        torch.autograd.graph.increment_version(momentum_state)

    def _compute_update(
        self, compute_layout: _ParameterComputeLayout, compute: Tensor
    ) -> None:
        group = self._group(compute_layout)
        _compute_muon_direction(
            compute,
            matrix_views=self._matrix_views_by_fqn[compute_layout.fqn],
            lr_reference_shape=compute_layout.lr_reference_shape,
            adjust_lr_fn=group["adjust_lr_fn"],
            ns_coefficients=group["ns_coefficients"],
            ns_steps=group["ns_steps"],
            eps=group["eps"],
        )

    def _apply_update(
        self, compute_layout: _ParameterComputeLayout, direction: Tensor
    ) -> None:
        group = self._group(compute_layout)
        local_param = compute_layout.param.to_local()
        if compute_layout.storage_is_compute_ready:
            local_param = local_param.detach()
        _apply_muon_update(
            local_param,
            direction,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            adjust_lr_fn=group["adjust_lr_fn"],
            compute_matrix_shape=compute_layout.lr_reference_shape,
        )
        torch.autograd.graph.increment_version(compute_layout.param)

    @staticmethod
    def _local_tensor_spec(
        compute_layout: _ParameterComputeLayout,
    ) -> tuple[torch.Size, torch.dtype, torch.device]:
        assert compute_layout.storage_is_compute_ready
        tensor = compute_layout.param.to_local().detach()
        return tensor.shape, tensor.dtype, tensor.device


@dataclass(frozen=True, slots=True)
class _MatrixBatchView:
    """A strided view into a contiguous compute tensor.

    Strides and offset are measured in elements, relative to the supplied tensor.
    """

    shape: torch.Size
    strides: tuple[int, ...]
    offset: int

    def view_as_matrix_batch(self, compute_tensor: Tensor) -> Tensor:
        """Return a zero-copy matrix-batch view of the compute tensor."""
        if not compute_tensor.is_contiguous():
            raise RuntimeError("matrix views require a contiguous compute tensor")
        end = self.offset
        if self.shape.numel():
            end += 1 + sum(
                (size - 1) * stride
                for size, stride in zip(self.shape, self.strides, strict=True)
            )
        if self.offset < 0 or end > compute_tensor.numel():
            raise RuntimeError(
                "compute tensor shape is inconsistent with its matrix view"
            )
        return compute_tensor.as_strided(
            self.shape,
            self.strides,
            storage_offset=compute_tensor.storage_offset() + self.offset,
        )


def _matrix_batch_views_from_shape(
    compute_shape: torch.Size,
    *,
    matrix_row_sizes: tuple[int, ...],
    logical_row_start: int,
) -> tuple[_MatrixBatchView, ...]:
    """Describe complete matrix batches in a logical parameter's row slice."""
    num_rows, matrix_columns = compute_shape
    matrix_row_stride = sum(matrix_row_sizes)
    position_row_offset = 0
    views = []
    for matrix_rows in matrix_row_sizes:
        # First occurrence of this pattern position within the local row slice.
        first_local_row = (position_row_offset - logical_row_start) % matrix_row_stride
        num_matrices = max(
            0, 1 + (num_rows - first_local_row - matrix_rows) // matrix_row_stride
        )
        if num_matrices:
            views.append(
                _MatrixBatchView(
                    shape=torch.Size((num_matrices, matrix_rows, matrix_columns)),
                    strides=(matrix_row_stride * matrix_columns, matrix_columns, 1),
                    offset=first_local_row * matrix_columns,
                )
            )
        position_row_offset += matrix_rows
    assert (
        sum(view.shape[0] * view.shape[1] for view in views) == num_rows
    ), "compute row boundaries must preserve complete matrices"
    return tuple(sorted(views, key=lambda view: view.offset))


def _validate_matrix_batch_storage_placements(
    fqn: str,
    param: DTensor,
) -> None:
    """Validate placements supported by matrix-batch storage."""
    for mesh_axis_size, placement in zip(
        param.device_mesh.shape,
        param.placements,
        strict=True,
    ):
        if mesh_axis_size == 1 or type(placement) is Replicate:
            continue
        if type(placement) is not Shard:
            raise ValueError(
                f"Muon parameter {fqn!r} with matrix-batch storage "
                "requires exact Shard or Replicate placements"
            )
        if placement.dim % param.ndim != 0:
            raise ValueError(
                f"Muon parameter {fqn!r} with matrix-batch storage "
                "requires shards along tensor dimension 0"
            )


def _row_intervals_by_mesh_axis_coordinate(
    num_rows: int,
    sharding: _AxisComputeSharding,
    *,
    mesh_axis_size: int,
) -> tuple[tuple[int, int], ...]:
    """Resolve the global rows owned by each mesh-axis coordinate.

    Placement names alone do not show whether communication is needed:
    ``Shard(0)`` and ``BlockShard(0, (R,))`` may own the same rows when shard
    boundaries align, or different rows when a storage shard splits a block.
    Comparing these intervals distinguishes a local view from redistribution.
    """
    if type(sharding) is Replicate:
        return ((0, num_rows),) * mesh_axis_size

    if type(sharding) is BlockShard:
        assert sharding.dim == 0
        num_sharding_units = sharding.num_blocks(num_rows)
    else:
        assert type(sharding) is Shard and sharding.dim == 0
        num_sharding_units = num_rows

    intervals = []
    for axis_coordinate in range(mesh_axis_size):
        local_num_units, unit_offset = Shard.local_shard_size_and_offset(
            num_sharding_units,
            mesh_axis_size,
            axis_coordinate,
        )
        if type(sharding) is BlockShard:
            start = sharding.block_start(unit_offset)
            end = sharding.block_start(unit_offset + local_num_units)
        else:
            start, end = unit_offset, unit_offset + local_num_units
        intervals.append((start, end))
    return tuple(intervals)


def _resolve_storage_to_compute_redistribution_requirement(
    fqn: str,
    param: DTensor,
    block_shard: BlockShard | None,
    target_sharding_by_storage_mesh_axis: Mapping[int, _AxisComputeSharding],
    declared_storage_mesh_axes: Sequence[int],
) -> tuple[int, ...]:
    """Compare actual storage with the target compute tensor."""
    if block_shard is None:
        changed_storage_mesh_axes = []
        mesh_axis_names = param.device_mesh.mesh_dim_names
        assert mesh_axis_names is not None
        for storage_mesh_axis in sorted(declared_storage_mesh_axes):
            source_sharding = _normalize_storage_placement(
                param.placements[storage_mesh_axis],
                ndim=param.ndim,
                mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
            )
            if type(source_sharding) is _UnsupportedStoragePlacement:
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} has unsupported storage placement "
                    f"{source_sharding.type_name!r} "
                    f"({source_sharding.representation}) on mesh axis "
                    f"{mesh_axis_names[storage_mesh_axis]!r}"
                )
            target_sharding = target_sharding_by_storage_mesh_axis.get(
                storage_mesh_axis
            )
            if target_sharding is not None and source_sharding != target_sharding:
                changed_storage_mesh_axes.append(storage_mesh_axis)
        redistribution_storage_mesh_axes = tuple(changed_storage_mesh_axes)
    else:
        mesh_axis_names = param.device_mesh.mesh_dim_names
        assert mesh_axis_names is not None
        active_block_shard_mesh_axes = tuple(
            storage_mesh_axis
            for storage_mesh_axis, target_sharding in (
                target_sharding_by_storage_mesh_axis.items()
            )
            if type(target_sharding) is BlockShard
            and param.device_mesh.size(storage_mesh_axis) > 1
        )
        if len(active_block_shard_mesh_axes) > 1:
            axis_names = [
                mesh_axis_names[axis] for axis in active_block_shard_mesh_axes
            ]
            raise NotImplementedError(
                f"Muon parameter {fqn!r} requests matrix-batch compute on "
                f"multiple active mesh axes {axis_names}; only one active "
                "BlockShard axis is supported"
            )

        redistribution_storage_mesh_axis = (
            active_block_shard_mesh_axes[0] if active_block_shard_mesh_axes else None
        )
        nonreplicated_other_mesh_axes = []
        for storage_mesh_axis, placement in enumerate(param.placements):
            if storage_mesh_axis == redistribution_storage_mesh_axis:
                continue
            storage_sharding = _normalize_storage_placement(
                placement,
                ndim=param.ndim,
                mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
            )
            if type(storage_sharding) is not Replicate:
                nonreplicated_other_mesh_axes.append(mesh_axis_names[storage_mesh_axis])
        if nonreplicated_other_mesh_axes:
            if redistribution_storage_mesh_axis is None:
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} matrix-batch compute requires an "
                    "active BlockShard target for every non-replicated storage "
                    f"mesh axis; non-replicated axes: {nonreplicated_other_mesh_axes}"
                )
            raise NotImplementedError(
                f"Muon parameter {fqn!r} matrix-batch compute along mesh "
                f"axis {mesh_axis_names[redistribution_storage_mesh_axis]!r} "
                "requires every other storage mesh axis to be replicated; "
                f"non-replicated axes: {nonreplicated_other_mesh_axes}"
            )

        if redistribution_storage_mesh_axis is None:
            redistribution_storage_mesh_axes = ()
        else:
            storage_sharding = cast(
                Replicate | Shard,
                _normalize_storage_placement(
                    param.placements[redistribution_storage_mesh_axis],
                    ndim=param.ndim,
                    mesh_axis_size=param.device_mesh.size(
                        redistribution_storage_mesh_axis
                    ),
                ),
            )
            compute_sharding = target_sharding_by_storage_mesh_axis[
                redistribution_storage_mesh_axis
            ]
            assert type(compute_sharding) is BlockShard
            mesh_axis_size = param.device_mesh.size(redistribution_storage_mesh_axis)
            storage_intervals = _row_intervals_by_mesh_axis_coordinate(
                param.shape[0],
                storage_sharding,
                mesh_axis_size=mesh_axis_size,
            )
            compute_intervals = _row_intervals_by_mesh_axis_coordinate(
                param.shape[0],
                compute_sharding,
                mesh_axis_size=mesh_axis_size,
            )
            redistribution_storage_mesh_axes = (
                ()
                if storage_intervals == compute_intervals
                else (redistribution_storage_mesh_axis,)
            )

    return redistribution_storage_mesh_axes


@dataclass(frozen=True, slots=True)
class _ParameterComputeLayout:
    fqn: str
    param: DTensor
    group_index: int
    storage_mesh_ranks: tuple[int, ...]
    storage_layout_signature: tuple[Any, ...]
    local_storage_signature: tuple[Any, ...]
    compute_sharding: _ResolvedComputeSharding
    storage_to_compute_transition: _StorageToComputeTransition
    resolved_compute_layout_signature: tuple[Any, ...]
    redistribution_storage_mesh_axis: int | None

    @property
    def lr_reference_shape(self) -> tuple[int, ...]:
        """Return the same LR reference shape on every rank."""
        if type(self.compute_sharding) is BlockShard:
            return (sum(self.compute_sharding.block_sizes), self.param.shape[-1])
        return self.param.shape[-2:]

    @property
    def storage_is_compute_ready(self) -> bool:
        return isinstance(
            self.storage_to_compute_transition, _NoRedistributionTransition
        )


@dataclass(frozen=True, slots=True)
class _NoRedistributionTransition:
    pass


@dataclass(frozen=True, slots=True)
class _RedistributionTransition:
    pass


_StorageToComputeTransition = _NoRedistributionTransition | _RedistributionTransition


# A per-mesh-axis compute sharding takes three successive forms while
# ``ComputeLayout`` resolves. The layout itself declares ``Owned``,
# ``Replicate``, ``Shard``, or ``BlockShard`` on each axis it names.

# Lowered: ``shard_order_by_tensor_dim`` has become DTensor placements, so an
# axis the layout applies later than storage-mesh order is now
# ``_StridedShard``. ``Owned`` passes through that lowering untouched.
_LoweredComputeSharding = Owned | Replicate | Shard | _StridedShard | BlockShard

# Per axis: ``Owned`` axes have been split out and are tracked separately, so
# every remaining axis carries a tensor sharding. ``BlockShard`` stays explicit.
_AxisComputeSharding = Replicate | Shard | _StridedShard | BlockShard

# Resolved: ``BlockShard`` retains its matrix boundaries for the planner.
_ResolvedComputeSharding = Owned | Replicate | Shard | BlockShard


@dataclass(frozen=True, slots=True)
class _UnsupportedStoragePlacement:
    type_name: str
    representation: str


@dataclass(frozen=True, slots=True)
class _ResolvedStorageToComputeTransition:
    compute_sharding: _ResolvedComputeSharding
    storage_to_compute_transition: _StorageToComputeTransition
    resolved_compute_layout_signature: tuple[Any, ...]
    redistribution_storage_mesh_axis: int | None = None


def _resolve_muon_redistribution_plans(
    contexts: tuple[_BucketPlanningContext[_ParameterComputeLayout], ...],
    *,
    ns_steps_by_group: Sequence[int],
) -> tuple[tuple[_RedistributionPlan | None, ...], ...]:
    """Resolve Muon compute shardings directly into transport plans."""
    cumulative_loads_by_participants: dict[tuple[int, ...], tuple[int, ...]] = {}
    specs_by_bucket = []
    unique_specs = {}
    for context in contexts:
        participants = context.group.participants
        initial_loads = cumulative_loads_by_participants.setdefault(
            participants,
            (0,) * len(participants),
        )
        owner_ranks, cumulative_loads = _assign_balanced_owner_ranks(
            context.items,
            participants=participants,
            cumulative_loads=initial_loads,
            ns_steps_by_group=ns_steps_by_group,
        )
        cumulative_loads_by_participants[participants] = cumulative_loads
        bucket_specs = []
        for layout, owner_rank in zip(context.items, owner_ranks, strict=True):
            if layout.storage_is_compute_ready:
                bucket_specs.append(None)
                continue
            spec = (
                layout.storage_layout_signature,
                tuple(layout.param.device_mesh.shape),
                layout.storage_mesh_ranks,
                layout.redistribution_storage_mesh_axis,
                context.group.participants,
                context.group.mesh_axis_participants,
                layout.compute_sharding,
                owner_rank,
            )
            bucket_specs.append(spec)
            unique_specs.setdefault(spec, (layout, context.group, owner_rank))
        specs_by_bucket.append(bucket_specs)

    plans_by_spec = {
        spec: _build_parameter_redistribution_plan(*args)
        for spec, args in unique_specs.items()
    }
    return tuple(
        tuple(None if spec is None else plans_by_spec[spec] for spec in bucket)
        for bucket in specs_by_bucket
    )


def _assign_balanced_owner_ranks(
    compute_layouts: Sequence[_ParameterComputeLayout],
    *,
    participants: tuple[int, ...],
    cumulative_loads: Sequence[int],
    ns_steps_by_group: Sequence[int],
) -> tuple[tuple[int | None, ...], tuple[int, ...]]:
    """Balance temporary compute ownership within and across ordered buckets."""
    assignments: list[int | None] = [None] * len(compute_layouts)
    candidates = tuple(
        (index, layout)
        for index, layout in enumerate(compute_layouts)
        if type(layout.compute_sharding) is Owned
    )
    candidate_partitions, updated_cumulative_loads = _balance_loads_across_partitions(
        tuple(
            (
                _estimate_muon_compute_cost(
                    layout.param.shape,
                    ns_steps_by_group[layout.group_index],
                ),
                layout.param.numel() * layout.param.element_size(),
                layout.fqn,
            )
            for _index, layout in candidates
        ),
        initial_cumulative_primary_loads=cumulative_loads,
    )
    for (index, _layout), partition in zip(
        candidates,
        candidate_partitions,
        strict=True,
    ):
        assignments[index] = participants[partition]
    return tuple(assignments), updated_cumulative_loads


def _estimate_muon_compute_cost(
    matrix_shape: torch.Size,
    ns_steps: int,
) -> int:
    *batch_shape, rows, columns = matrix_shape
    num_matrices = math.prod(batch_shape)
    short_dim, long_dim = sorted((rows, columns))
    # Each NS step has two s^2 * l matmuls and one s^3 matmul.
    return num_matrices * ns_steps * short_dim * short_dim * (2 * long_dim + short_dim)


def _balance_loads_across_partitions(
    loads: Sequence[tuple[int, int, str]],
    *,
    initial_cumulative_primary_loads: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Balance keyed loads with a deterministic LPT heuristic.

    Each load is ``(primary, secondary, stable_key)``. Assignments are
    partition indices aligned with those loads. Each call balances its primary
    and then secondary loads before cumulative primary load. Stable keys make
    ordering deterministic. This is not an exact partition optimum.
    """
    num_partitions = len(initial_cumulative_primary_loads)
    assignments = [0] * len(loads)
    partition_loads = [
        (0, 0, cumulative_load, partition)
        for partition, cumulative_load in enumerate(initial_cumulative_primary_loads)
    ]
    heapq.heapify(partition_loads)
    ordered_loads = sorted(
        enumerate(loads),
        key=lambda indexed_load: (
            -indexed_load[1][0],
            -indexed_load[1][1],
            indexed_load[1][2],
        ),
    )
    for load_index, (primary, secondary, _stable_key) in ordered_loads:
        (
            current_primary,
            current_secondary,
            cumulative_primary,
            partition,
        ) = heapq.heappop(partition_loads)
        assignments[load_index] = partition
        heapq.heappush(
            partition_loads,
            (
                current_primary + primary,
                current_secondary + secondary,
                cumulative_primary + primary,
                partition,
            ),
        )

    updated_cumulative_primary_loads = [0] * num_partitions
    for _primary, _secondary, cumulative_primary, partition in partition_loads:
        updated_cumulative_primary_loads[partition] = cumulative_primary
    return tuple(assignments), tuple(updated_cumulative_primary_loads)


def _build_parameter_redistribution_plan(
    compute_layout: _ParameterComputeLayout,
    group: _RedistributionGroup,
    owner_rank: int | None,
) -> _RedistributionPlan | None:
    transition = compute_layout.storage_to_compute_transition
    if isinstance(transition, _NoRedistributionTransition):
        return None
    assert isinstance(transition, _RedistributionTransition)

    group_local_storage_shape, storage_regions = _dtensor_storage_regions(
        compute_layout.param,
        group.participants,
        required_storage_mesh_axis=(compute_layout.redistribution_storage_mesh_axis),
    )
    compute_sharding = compute_layout.compute_sharding
    if type(compute_sharding) is Owned:
        assert owner_rank is not None
        assert owner_rank in group.participants
        return _build_owned_redistribution_plan(
            storage_regions,
            participants=group.participants,
            owner_rank=owner_rank,
            logical_shape=tuple(compute_layout.param.shape),
        )

    assert owner_rank is None
    if type(compute_sharding) is Shard:
        return _build_dim0_shard_redistribution_plan(
            storage_regions,
            participants=group.participants,
            shard_participants=group.mesh_axis_participants,
            logical_shape=group_local_storage_shape,
        )

    assert type(compute_sharding) is BlockShard
    return _build_batched_matrix_redistribution_plan(
        storage_regions,
        participants=group.participants,
        shard_participants=group.mesh_axis_participants,
        storage_shape=tuple(compute_layout.param.shape),
        block_shard=compute_sharding,
    )


def _build_batched_matrix_redistribution_plan(
    storage_regions: Sequence[_StorageRegionMapping],
    *,
    participants: tuple[int, ...],
    shard_participants: tuple[int, ...],
    storage_shape: tuple[int, ...],
    block_shard: BlockShard,
) -> _RedistributionPlan:
    """Build flat 2D routes whose destination shards contain complete matrices."""
    _require_valid_plan(
        len(storage_shape) == 2 and block_shard.dim == 0,
        "block redistribution requires flat 2D storage and row blocks",
    )
    _require_valid_plan(
        set(shard_participants) == set(participants)
        and len(shard_participants) == len(participants),
        "block shard participants must match the redistribution group",
    )

    num_matrices = block_shard.num_blocks(storage_shape[0])
    matrix_columns = storage_shape[1]
    storage_by_participant = {}
    storage_endpoints = []
    for holders, logical_region in storage_regions:
        _require_valid_plan(
            bool(holders)
            and len(logical_region.shape) == 2
            and logical_region.offsets[1] == 0
            and logical_region.shape[1] == matrix_columns,
            "matrix-batch redistribution requires row-sharded 2D storage",
        )
        tensor_region = _TensorRegion(
            offsets=(0, 0),
            shape=logical_region.shape,
        )
        storage_endpoints.append((holders, logical_region, tensor_region))
        for participant in holders:
            _require_valid_plan(
                participant in participants
                and participant not in storage_by_participant,
                "matrix-batch storage holders must partition participants",
            )
            storage_by_participant[participant] = (logical_region, tensor_region)

    _require_valid_plan(
        set(storage_by_participant) == set(participants),
        "matrix-batch storage must cover every participant",
    )

    storage_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=storage_by_participant[participant][1].shape,
            logical_regions=(storage_by_participant[participant][0],),
        )
        for participant in participants
    )

    compute_endpoints = []
    compute_partitions_list = []
    shard_index_by_participant = {
        participant: index for index, participant in enumerate(shard_participants)
    }
    for participant in participants:
        local_num_matrices, matrix_offset = Shard.local_shard_size_and_offset(
            num_matrices,
            len(participants),
            shard_index_by_participant[participant],
        )
        logical_region = _TensorRegion(
            offsets=(block_shard.block_start(matrix_offset), 0),
            shape=(
                block_shard.block_start(matrix_offset + local_num_matrices)
                - block_shard.block_start(matrix_offset),
                matrix_columns,
            ),
        )
        compute_endpoints.append(((participant,), matrix_offset, local_num_matrices))
        compute_partitions_list.append(
            _ParticipantPartition(
                participant=participant,
                tensor_shape=logical_region.shape,
                logical_regions=(logical_region,),
            )
        )
    compute_partitions = tuple(compute_partitions_list)

    storage_to_compute_routes = []
    for source_holders, storage_region, storage_tensor_base in storage_endpoints:
        storage_row_offset = storage_region.offsets[0]
        storage_row_end = storage_row_offset + storage_region.shape[0]
        for (
            destination_holders,
            matrix_offset,
            local_num_matrices,
        ) in compute_endpoints:
            for local_matrix_index in range(local_num_matrices):
                matrix_index = matrix_offset + local_matrix_index
                matrix_row_offset = block_shard.block_start(matrix_index)
                route_row_offset = max(storage_row_offset, matrix_row_offset)
                route_row_end = min(
                    storage_row_end,
                    block_shard.block_start(matrix_index + 1),
                )
                route_rows = route_row_end - route_row_offset
                if route_rows <= 0:
                    continue

                logical_region = _TensorRegion(
                    offsets=(route_row_offset, 0),
                    shape=(route_rows, matrix_columns),
                )
                storage_tensor_region = _TensorRegion(
                    offsets=(
                        storage_tensor_base.offsets[0]
                        + route_row_offset
                        - storage_row_offset,
                        0,
                    ),
                    shape=(route_rows, matrix_columns),
                )
                compute_tensor_region = _TensorRegion(
                    offsets=(
                        route_row_offset - block_shard.block_start(matrix_offset),
                        0,
                    ),
                    shape=(route_rows, matrix_columns),
                )
                storage_to_compute_routes.append(
                    _TensorRegionRoute(
                        logical_region=logical_region,
                        source=_RouteEndpoint(
                            storage_tensor_region,
                            source_holders,
                        ),
                        destination=_RouteEndpoint(
                            compute_tensor_region,
                            destination_holders,
                        ),
                    )
                )

    return _RedistributionPlan(
        participants=participants,
        logical_shape=storage_shape,
        storage_partitions=storage_partitions,
        compute_partitions=compute_partitions,
        storage_to_compute_routes=tuple(storage_to_compute_routes),
    )


def _lower_shard_order_to_strided_shards(
    param: DTensor,
    compute_layout: ComputeLayout,
    storage_axis_by_name: Mapping[str, int],
    shardings_by_storage_mesh_axis: Mapping[int, _LoweredComputeSharding],
) -> dict[int, _LoweredComputeSharding]:
    """Encode a declared shard order as DTensor placements.

    DTensor applies same-dimension shard axes in storage-mesh order, so an axis
    that the compute layout applies later than that default becomes a
    ``_StridedShard`` whose split factor is the product of the mesh sizes of the
    axes it now follows but that sit to its right in the storage mesh. Axes the
    layout declares for other mesh variants are absent here and drop out of the
    order.
    """
    lowered_shardings = dict(shardings_by_storage_mesh_axis)
    for tensor_dim, axis_names in compute_layout.shard_order_by_tensor_dim.items():
        ordered_mesh_axes = [
            storage_axis_by_name[axis_name]
            for axis_name in axis_names
            if axis_name in storage_axis_by_name
        ]
        for order_index, storage_mesh_axis in enumerate(ordered_mesh_axes):
            split_factor = math.prod(
                param.device_mesh.size(preceding_mesh_axis)
                for preceding_mesh_axis in ordered_mesh_axes[:order_index]
                if preceding_mesh_axis > storage_mesh_axis
            )
            if split_factor > 1:
                lowered_shardings[storage_mesh_axis] = _StridedShard(
                    tensor_dim, split_factor=split_factor
                )
    return lowered_shardings


def _validate_shard_order_compute_targets(
    fqn: str,
    param: DTensor,
    target_sharding_by_storage_mesh_axis: Mapping[int, _AxisComputeSharding],
    owned_storage_mesh_axes: Sequence[int],
) -> None:
    """Reject declared shard orders that DistMuon cannot lower to a transport plan.

    DistMuon supports reordering a mesh axis behind exactly one later mesh axis
    that shards the same tensor dimension, because that axis is the one whose
    storage ownership the redistribution preserves.
    """
    mesh_axis_names = param.device_mesh.mesh_dim_names
    assert mesh_axis_names is not None
    owned_axis_set = set(owned_storage_mesh_axes)
    for (
        storage_mesh_axis,
        target_sharding,
    ) in target_sharding_by_storage_mesh_axis.items():
        if type(target_sharding) is not _StridedShard:
            continue
        target_dim = _normalize_dim(target_sharding.dim, param.ndim)
        rightward_shard_axes = []
        for rightward_mesh_axis in range(
            storage_mesh_axis + 1,
            param.device_mesh.ndim,
        ):
            if rightward_mesh_axis in owned_axis_set:
                continue
            rightward_sharding = target_sharding_by_storage_mesh_axis.get(
                rightward_mesh_axis
            )
            if rightward_sharding is None:
                rightward_sharding = _normalize_storage_placement(
                    param.placements[rightward_mesh_axis],
                    ndim=param.ndim,
                    mesh_axis_size=param.device_mesh.size(rightward_mesh_axis),
                )
            if (
                type(rightward_sharding) is Shard
                and _normalize_dim(rightward_sharding.dim, param.ndim) == target_dim
            ):
                rightward_shard_axes.append(rightward_mesh_axis)

        if len(rightward_shard_axes) != 1:
            raise ValueError(
                f"Muon parameter {fqn!r} orders mesh axis "
                f"{mesh_axis_names[storage_mesh_axis]!r} after another axis on "
                f"tensor dimension {target_dim}; DistMuon requires exactly one "
                "later mesh axis to shard that dimension"
            )
        preserved_mesh_axis_size = param.device_mesh.size(rightward_shard_axes[0])
        if target_sharding.split_factor != preserved_mesh_axis_size:
            raise ValueError(
                f"Muon parameter {fqn!r} must order mesh axis "
                f"{mesh_axis_names[storage_mesh_axis]!r} directly after "
                f"{mesh_axis_names[rightward_shard_axes[0]]!r} on tensor "
                f"dimension {target_dim}; DistMuon does not support ordering it "
                "after further mesh axes"
            )


def _is_supported_orthogonal_dim0_shard_redistribution(
    *,
    ndim: int,
    block_shard: BlockShard | None,
    source_storage_placement: object,
    target_compute_sharding: _AxisComputeSharding | None,
    preserved_storage_placement: object,
) -> bool:
    if block_shard is not None or ndim < 3:
        return False
    if (
        type(source_storage_placement) is not Shard
        or type(target_compute_sharding) is not _StridedShard
        or type(preserved_storage_placement) is not Shard
    ):
        return False

    storage_dim = _normalize_dim(source_storage_placement.dim, ndim)
    compute_dim = _normalize_dim(target_compute_sharding.dim, ndim)
    preserved_dim = _normalize_dim(preserved_storage_placement.dim, ndim)
    return storage_dim == ndim - 2 and compute_dim == preserved_dim == 0


def _resolve_storage_to_compute_transition(
    fqn: str,
    param: DTensor,
    compute_layout: ComputeLayout,
) -> _ResolvedStorageToComputeTransition:
    """Validate one storage layout and resolve its concrete compute transition."""
    local = param.to_local()
    if param.ndim < 2:
        raise ValueError(
            f"Muon parameter {fqn!r} compute shape "
            f"{tuple(param.shape)} must be a matrix or matrix batch"
        )
    if torch.is_complex(param) or not local.is_contiguous():
        _raise_unsupported_layout(fqn)

    mesh_axis_names = param.device_mesh.mesh_dim_names
    if mesh_axis_names is None:
        raise ValueError(
            f"Muon parameter {fqn!r} requires a storage mesh with named axes"
        )
    storage_axis_by_name = {
        axis_name: storage_mesh_axis
        for storage_mesh_axis, axis_name in enumerate(mesh_axis_names)
    }
    applicable_compute_shardings_by_storage_mesh_axis: dict[
        int, _LoweredComputeSharding
    ] = {
        storage_axis_by_name[axis_name]: sharding
        for axis_name, sharding in compute_layout.shardings_by_mesh_axis.items()
        if axis_name in storage_axis_by_name
    }
    if not applicable_compute_shardings_by_storage_mesh_axis:
        declared_axes = sorted(compute_layout.shardings_by_mesh_axis)
        raise ValueError(
            f"Muon compute layout for parameter {fqn!r} declares no axis in "
            f"storage mesh {list(mesh_axis_names)}; declared axes: {declared_axes}"
        )
    applicable_compute_shardings_by_storage_mesh_axis = (
        _lower_shard_order_to_strided_shards(
            param,
            compute_layout,
            storage_axis_by_name,
            applicable_compute_shardings_by_storage_mesh_axis,
        )
    )

    applicable_owned_storage_mesh_axes = tuple(
        storage_mesh_axis
        for storage_mesh_axis, sharding in (
            applicable_compute_shardings_by_storage_mesh_axis.items()
        )
        if type(sharding) is Owned
    )

    # Unit mesh axes normalize to Replicate but retain their declared blocks.
    block_shards = tuple(
        sharding
        for sharding in applicable_compute_shardings_by_storage_mesh_axis.values()
        if type(sharding) is BlockShard
    )
    block_shard = None
    if block_shards:
        if param.ndim != 2:
            raise ValueError(
                f"Muon parameter {fqn!r} BlockShard currently requires a "
                f"2D row-concatenated parameter; got shape {tuple(param.shape)}"
            )
        normalized_dims = tuple(
            _normalize_dim(sharding.dim, param.ndim) for sharding in block_shards
        )
        if any(dim != 0 for dim in normalized_dims):
            raise ValueError(
                f"Muon parameter {fqn!r} matrix-batch BlockShard must shard "
                "tensor dimension 0"
            )
        block_sizes = block_shards[0].block_sizes
        if any(sharding.block_sizes != block_sizes for sharding in block_shards):
            raise ValueError(
                f"Muon parameter {fqn!r} must use the same BlockShard block sizes "
                "across mesh axes"
            )
        block_shard = BlockShard(dim=0, block_sizes=block_sizes)
        if block_shard.num_blocks(param.shape[0]) == 0:
            raise ValueError(
                f"Muon parameter {fqn!r} requires at least one matrix block"
            )
        _validate_matrix_batch_storage_placements(fqn, param)
        if applicable_owned_storage_mesh_axes:
            raise ValueError(
                f"Muon owned compute for parameter {fqn!r} requires a 2D matrix"
            )
        shard_axes = [
            mesh_axis_names[storage_mesh_axis]
            for storage_mesh_axis, sharding in (
                applicable_compute_shardings_by_storage_mesh_axis.items()
            )
            if type(sharding) in (Shard, _StridedShard)
        ]
        if shard_axes:
            raise ValueError(
                f"Muon parameter {fqn!r} with matrix-batch compute requires "
                f"BlockShard instead of Shard on mesh axes {shard_axes}"
            )

    replicated_axes = [
        mesh_axis_names[storage_mesh_axis]
        for storage_mesh_axis, sharding in (
            applicable_compute_shardings_by_storage_mesh_axis.items()
        )
        if type(sharding) is Replicate
    ]
    if replicated_axes:
        raise NotImplementedError(
            f"Muon parameter {fqn!r} requests explicit replicated compute on "
            f"mesh axes {replicated_axes}; replicated compute is not implemented"
        )

    normalized_target_sharding_by_storage_mesh_axis: dict[
        int, _AxisComputeSharding
    ] = {}
    declared_shard_dims = []
    for (
        storage_mesh_axis,
        sharding,
    ) in applicable_compute_shardings_by_storage_mesh_axis.items():
        if type(sharding) is Owned:
            continue
        placement = cast(_AxisComputeSharding, sharding)
        if type(placement) is Shard:
            declared_shard_dims.append(_normalize_dim(placement.dim, param.ndim))
        elif type(placement) is _StridedShard:
            declared_shard_dims.append(_normalize_dim(placement.dim, param.ndim))
        elif type(placement) is BlockShard:
            declared_shard_dims.append(0)
        target_sharding = _normalize_compute_placement(
            placement,
            ndim=param.ndim,
            mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
        )
        normalized_target_sharding_by_storage_mesh_axis[
            storage_mesh_axis
        ] = target_sharding

    _validate_shard_order_compute_targets(
        fqn,
        param,
        normalized_target_sharding_by_storage_mesh_axis,
        applicable_owned_storage_mesh_axes,
    )
    changed_storage_mesh_axes = _resolve_storage_to_compute_redistribution_requirement(
        fqn,
        param,
        block_shard,
        normalized_target_sharding_by_storage_mesh_axis,
        tuple(applicable_compute_shardings_by_storage_mesh_axis),
    )
    active_owned_storage_mesh_axes = tuple(
        storage_mesh_axis
        for storage_mesh_axis in applicable_owned_storage_mesh_axes
        if param.device_mesh.size(storage_mesh_axis) > 1
    )
    transport_mesh_axes = tuple(
        sorted(set(changed_storage_mesh_axes).union(active_owned_storage_mesh_axes))
    )
    if len(transport_mesh_axes) > 1:
        axis_names = [mesh_axis_names[axis] for axis in transport_mesh_axes]
        raise NotImplementedError(
            f"Muon parameter {fqn!r} requires compute redistribution or "
            f"owned compute on multiple mesh axes {axis_names}; multi-axis "
            "transport is not implemented"
        )

    redistribution_storage_mesh_axis = (
        transport_mesh_axes[0] if transport_mesh_axes else None
    )
    uses_supported_orthogonal_shard_redistribution = False
    if redistribution_storage_mesh_axis is not None:
        redistribution_axis_name = mesh_axis_names[redistribution_storage_mesh_axis]
        for storage_mesh_axis, placement in enumerate(param.placements):
            if storage_mesh_axis == redistribution_storage_mesh_axis:
                if type(placement) not in (Replicate, Shard):
                    raise NotImplementedError(
                        f"Muon parameter {fqn!r} cannot redistribute "
                        f"{type(placement).__name__} storage on mesh axis "
                        f"{redistribution_axis_name!r}"
                    )
            else:
                preserved_storage_sharding = _normalize_storage_placement(
                    placement,
                    ndim=param.ndim,
                    mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
                )
                if type(preserved_storage_sharding) is Replicate:
                    continue
                redistribution_storage_placement = param.placements[
                    redistribution_storage_mesh_axis
                ]
                redistribution_compute_sharding = (
                    normalized_target_sharding_by_storage_mesh_axis.get(
                        redistribution_storage_mesh_axis
                    )
                )
                if (
                    not uses_supported_orthogonal_shard_redistribution
                    and _is_supported_orthogonal_dim0_shard_redistribution(
                        ndim=param.ndim,
                        block_shard=block_shard,
                        source_storage_placement=redistribution_storage_placement,
                        target_compute_sharding=redistribution_compute_sharding,
                        preserved_storage_placement=placement,
                    )
                ):
                    uses_supported_orthogonal_shard_redistribution = True
                    continue
                if (
                    type(redistribution_storage_placement) is Shard
                    and type(redistribution_compute_sharding)
                    in (Shard, _StridedShard, BlockShard)
                    and type(placement) is Shard
                ):
                    target_compute_shard = cast(
                        Shard | _StridedShard | BlockShard,
                        redistribution_compute_sharding,
                    )
                    storage_dim = _normalize_dim(
                        redistribution_storage_placement.dim, param.ndim
                    )
                    target_dim = _normalize_dim(target_compute_shard.dim, param.ndim)
                    preserved_dim = _normalize_dim(placement.dim, param.ndim)
                    if (
                        storage_dim == param.ndim - 2
                        and target_dim == preserved_dim == 0
                        and type(redistribution_compute_sharding) is Shard
                        and redistribution_storage_mesh_axis < storage_mesh_axis
                    ):
                        preserved_axis_name = mesh_axis_names[storage_mesh_axis]
                        raise ValueError(
                            f"Muon parameter {fqn!r} must declare "
                            "shard_order_by_tensor_dim={0: "
                            f"({preserved_axis_name!r}, "
                            f"{redistribution_axis_name!r})}} when preserving "
                            f"Shard(0) on mesh axis {preserved_axis_name!r}"
                        )
                    raise NotImplementedError(
                        f"Muon parameter {fqn!r} cannot redistribute storage on "
                        f"mesh axis {redistribution_axis_name!r} from "
                        f"Shard({storage_dim}) to "
                        f"{redistribution_compute_sharding!r} while "
                        f"preserving Shard({preserved_dim}) storage on mesh axis "
                        f"{mesh_axis_names[storage_mesh_axis]!r}; orthogonal-shard "
                        "redistribution is not implemented"
                    )
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} cannot redistribute mesh axis "
                    f"{redistribution_axis_name!r} while storage mesh axis "
                    f"{mesh_axis_names[storage_mesh_axis]!r} has non-replicated "
                    f"placement {placement}; this implementation requires every "
                    "other storage mesh axis to be replicated"
                )

    reordered_axis_names = [
        mesh_axis_names[storage_mesh_axis]
        for storage_mesh_axis, target_sharding in (
            normalized_target_sharding_by_storage_mesh_axis.items()
        )
        if type(target_sharding) is _StridedShard
    ]
    if (
        reordered_axis_names
        and redistribution_storage_mesh_axis is not None
        and not uses_supported_orthogonal_shard_redistribution
    ):
        raise ValueError(
            f"Muon parameter {fqn!r} has an unsupported shard order on mesh "
            f"axes {reordered_axis_names}; DistMuon currently reorders a mesh "
            "axis only when a preceding redistribution axis preserves one "
            "rightward Shard axis on the same tensor dimension"
        )

    resolved_target_signature = []
    resolved_shard_dims = []
    owned_axis_set = set(applicable_owned_storage_mesh_axes)
    for storage_mesh_axis, axis_name in enumerate(mesh_axis_names):
        if storage_mesh_axis in owned_axis_set:
            target_sharding: (
                Owned | _AxisComputeSharding | _UnsupportedStoragePlacement
            ) = Owned()
        elif storage_mesh_axis in normalized_target_sharding_by_storage_mesh_axis:
            target_sharding = normalized_target_sharding_by_storage_mesh_axis[
                storage_mesh_axis
            ]
        else:
            target_sharding = _normalize_storage_placement(
                param.placements[storage_mesh_axis],
                ndim=param.ndim,
                mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
            )
        if type(target_sharding) is _UnsupportedStoragePlacement:
            raise NotImplementedError(
                f"Muon parameter {fqn!r} has unsupported compute placement "
                f"{target_sharding.type_name!r} "
                f"({target_sharding.representation}) on mesh axis {axis_name!r}"
            )
        resolved_target_signature.append((axis_name, target_sharding))
        if type(target_sharding) is Shard:
            resolved_shard_dims.append(target_sharding.dim)
        elif type(target_sharding) is _StridedShard:
            resolved_shard_dims.append(target_sharding.dim)
        elif type(target_sharding) is BlockShard:
            resolved_shard_dims.append(0)

    resolved_compute_layout_signature = tuple(resolved_target_signature)
    compute_shard_dims = [*resolved_shard_dims, *declared_shard_dims]
    if applicable_owned_storage_mesh_axes and block_shard is not None:
        raise ValueError(
            f"Muon owned compute for parameter {fqn!r} requires a native "
            "matrix or matrix-batch tensor"
        )
    if active_owned_storage_mesh_axes:
        compute_sharding: _ResolvedComputeSharding = Owned()
    elif compute_shard_dims:
        if block_shard is None and param.ndim == 2:
            raise ValueError(
                f"Muon parameter {fqn!r}: 2D Muon compute cannot use Shard; "
                "use Owned() for one matrix or "
                "BlockShard(dim=0, block_sizes=(R,)) for row-concatenated matrices"
            )
        if (block_shard is None and param.ndim < 3) or any(
            shard_dim != 0 for shard_dim in compute_shard_dims
        ):
            raise ValueError(
                f"Muon sharded compute for parameter {fqn!r} requires a native "
                "matrix batch sharded only on tensor dimension 0"
            )
        compute_sharding = block_shard if block_shard is not None else Shard(0)
    elif applicable_owned_storage_mesh_axes:
        compute_sharding = Owned()
    else:
        raise ValueError(f"unsupported storage-to-compute layout for {fqn!r}")

    if redistribution_storage_mesh_axis is None:
        return _ResolvedStorageToComputeTransition(
            compute_sharding=compute_sharding,
            storage_to_compute_transition=_NoRedistributionTransition(),
            resolved_compute_layout_signature=resolved_compute_layout_signature,
        )

    return _ResolvedStorageToComputeTransition(
        compute_sharding=compute_sharding,
        storage_to_compute_transition=_RedistributionTransition(),
        resolved_compute_layout_signature=resolved_compute_layout_signature,
        redistribution_storage_mesh_axis=redistribution_storage_mesh_axis,
    )


def _raise_unsupported_layout(fqn: str) -> NoReturn:
    raise ValueError(f"unsupported storage-to-compute layout for {fqn!r}")


def _normalize_compute_placement(
    placement: _AxisComputeSharding,
    *,
    ndim: int,
    mesh_axis_size: int,
) -> _AxisComputeSharding:
    if type(placement) is Replicate:
        return Replicate()
    if type(placement) is Shard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if mesh_axis_size == 1:
            return Replicate()
        return Shard(normalized_dim)
    if type(placement) is _StridedShard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if mesh_axis_size == 1:
            return Replicate()
        if placement.split_factor == 1:
            return Shard(normalized_dim)
        return _StridedShard(
            normalized_dim,
            split_factor=placement.split_factor,
        )
    assert type(placement) is BlockShard
    normalized_dim = _normalize_dim(placement.dim, ndim)
    if mesh_axis_size == 1:
        return Replicate()
    return BlockShard(normalized_dim, placement.block_sizes)


def _normalize_storage_placement(
    placement: object,
    *,
    ndim: int,
    mesh_axis_size: int,
) -> Replicate | Shard | _StridedShard | _UnsupportedStoragePlacement:
    if mesh_axis_size == 1 or type(placement) is Replicate:
        return Replicate()
    if type(placement) is Shard:
        return Shard(_normalize_dim(placement.dim, ndim))
    if type(placement) is _StridedShard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if placement.split_factor == 1:
            return Shard(normalized_dim)
        return _StridedShard(
            normalized_dim,
            split_factor=placement.split_factor,
        )
    return _UnsupportedStoragePlacement(
        type_name=type(placement).__name__,
        representation=repr(placement),
    )


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim if dim >= 0 else dim + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dimension {dim} is invalid for a rank-{ndim} tensor")
    return normalized


def _adjust_muon_learning_rate(
    lr: float,
    adjust_lr_fn: str | None,
    compute_matrix_shape: torch.Size | tuple[int, ...],
) -> float:
    """Adjust Muon's learning rate for the matrix aspect ratio."""
    rows, columns = compute_matrix_shape[-2:]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, rows / columns))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(rows, columns))
    elif adjust_lr_fn == "spectral_unclamped":
        ratio = math.sqrt(rows / columns)
    else:
        raise ValueError(f"unsupported adjust_lr_fn {adjust_lr_fn!r}")
    return lr * ratio


def _prepare_muon_input(
    gradient: Tensor,
    momentum_buffer: Tensor,
    *,
    momentum: float,
    nesterov: bool,
    out: Tensor,
) -> Tensor:
    """Update momentum and prepare the Tensor passed to Muon computation."""
    momentum_buffer.lerp_(gradient, 1 - momentum)
    if nesterov:
        torch.lerp(
            gradient,
            momentum_buffer,
            momentum,
            out=out,
        )
    else:
        out.copy_(momentum_buffer)
    return out


def _compute_muon_direction(
    prepared: Tensor,
    *,
    matrix_views: Sequence[_MatrixBatchView],
    lr_reference_shape: torch.Size | tuple[int, ...],
    adjust_lr_fn: str | None,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Compute independent matrix directions with relative shape scaling."""
    reference_ratio = _adjust_muon_learning_rate(1.0, adjust_lr_fn, lr_reference_shape)
    for view in matrix_views:
        matrices = view.view_as_matrix_batch(prepared)
        matrices.copy_(
            _zeropower_via_newtonschulz(
                matrices,
                ns_coefficients=ns_coefficients,
                ns_steps=ns_steps,
                eps=eps,
            )
        )
        ratio = _adjust_muon_learning_rate(1.0, adjust_lr_fn, matrices.shape[-2:])
        # Preserve the original update arithmetic when the shape factors match.
        if ratio != reference_ratio:
            matrices.mul_(ratio / reference_ratio)
    return prepared


def _apply_muon_update(
    parameter: Tensor,
    direction: Tensor,
    *,
    lr: float,
    weight_decay: float,
    adjust_lr_fn: str | None,
    compute_matrix_shape: torch.Size | tuple[int, ...],
) -> Tensor:
    """Apply decoupled weight decay and a computed Muon direction."""
    adjusted_lr = _adjust_muon_learning_rate(
        lr,
        adjust_lr_fn,
        compute_matrix_shape,
    )
    parameter.mul_(1 - lr * weight_decay)
    parameter.add_(direction, alpha=-adjusted_lr)
    return parameter


def _zeropower_via_newtonschulz(
    update: Tensor,
    *,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Compute Muon's approximate polar factor without optimizer state."""
    a, b, c = ns_coefficients
    result = update.to(dtype=torch.bfloat16, copy=True)
    transposed = result.shape[-2] > result.shape[-1]
    if transposed:
        result = result.transpose(-2, -1)
    result.div_(result.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))

    if result.ndim == 2:
        for _ in range(ns_steps):
            gram = result @ result.T
            gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            result = torch.addmm(result, gram_update, result, beta=a)
    else:
        original_shape = result.shape
        matrices = result.reshape(-1, *original_shape[-2:])
        # Batched kernels and independent matrix calls can use different BF16
        # reduction orders.
        for _ in range(ns_steps):
            gram = matrices @ matrices.transpose(-2, -1)
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            matrices = torch.baddbmm(matrices, gram_update, matrices, beta=a)
        result = matrices.reshape(original_shape)

    return result.transpose(-2, -1) if transposed else result


def _after_load_state_dict(optimizer: Optimizer) -> None:
    muon = cast(DistMuon, optimizer)
    # Optimizer.load_state_dict restores group values such as ns_steps after
    # construction, and those values affect compute planning and buffer sizes.
    muon._validate_groups()
    muon._initialize_plan(muon._parameter_compute_layouts)
    muon._validate_plan_across_ranks()
    muon._redistribution_runtime.reserve_buffers(
        muon._bucket_plans,
        local_tensor_spec=muon._local_tensor_spec,
    )
    # init_optim_state may have validated placeholder state before the load.
    muon._first_step_validated = False


def _local_storage_signature(tensor: Tensor) -> tuple[Any, ...]:
    return (
        tensor.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _storage_layout_signature(tensor: DTensor) -> tuple[Any, ...]:
    local = tensor.to_local()
    # Empty shards address no elements, and autograd can choose different strides.
    local_strides = tuple(local.stride()) if local.numel() else ()
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.placements,
        tuple(local.shape),
        local_strides,
        local.dtype,
        local.device,
        local.is_contiguous(),
    )
