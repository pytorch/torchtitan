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
    _build_owned_redistribution_plan,
    _build_replicated_to_dim0_shard_plan,
    _device_mesh_ranks,
    _dtensor_storage_regions,
    _ParticipantPartition,
    _RedistributionGroup,
    _RedistributionPlan,
    _require_valid_plan,
    _RouteEndpoint,
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
    ``[M * R, C]`` with contiguous local DTensor storage. The placement must
    target tensor dimension 0 with ``block_size=R``; the leading dimension
    must be nonzero and divisible by ``R``. Each consecutive ``R`` rows forms
    one independent ``[R, C]`` matrix for local Muon compute. A native
    batch-first 3D ``[M, R, C]`` parameter uses ``Shard(0)`` to distribute
    complete matrices. A single 2D matrix without ``BlockShard`` uses
    whole-matrix compute such as ``Owned``.
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


def _matrix_batch_view_from_compute_layout(
    fqn: str,
    param: Tensor,
    compute_layout: ComputeLayout,
) -> _MatrixBatchView | None:
    storage_shape = torch.Size(param.shape)
    applicable_axis_names = None
    if isinstance(param, DTensor):
        mesh_axis_names = param.device_mesh.mesh_dim_names
        if mesh_axis_names is None:
            raise ValueError(
                f"Muon parameter {fqn!r} requires a storage mesh with named axes"
            )
        applicable_axis_names = set(mesh_axis_names)
    block_shards = tuple(
        cast(BlockShard, sharding)
        for axis_name, sharding in compute_layout.shardings_by_mesh_axis.items()
        if type(sharding) is BlockShard
        and (applicable_axis_names is None or axis_name in applicable_axis_names)
    )
    if not block_shards:
        return None

    if len(storage_shape) != 2:
        raise ValueError(
            f"Muon parameter {fqn!r} BlockShard currently requires a "
            f"2D [M * R, C] parameter; got shape {tuple(storage_shape)}"
        )
    normalized_dims = tuple(
        _normalize_dim(block_shard.dim, len(storage_shape))
        for block_shard in block_shards
    )
    if any(dim != 0 for dim in normalized_dims):
        raise ValueError(
            f"Muon parameter {fqn!r} matrix-batch BlockShard must shard "
            "tensor dimension 0"
        )
    block_size = block_shards[0].block_size
    if any(block_shard.block_size != block_size for block_shard in block_shards):
        raise ValueError(
            f"Muon parameter {fqn!r} must use one BlockShard block size "
            "across mesh axes"
        )
    return _MatrixBatchView.from_storage_shape(
        storage_shape,
        matrix_rows=block_size,
    )


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

    parameters_to_prepare = []
    for param_group in optimizer.param_groups:
        group_params = tuple(param_group["params"])
        raw_param_names = param_group.get("param_names")
        param_names = () if raw_param_names is None else tuple(raw_param_names)
        if raw_param_names is None or len(group_params) != len(param_names):
            raise ValueError("params and param_names must be aligned")
        for param, fqn in zip(group_params, param_names, strict=True):
            if fqn not in compute_sharding_by_fqn:
                raise ValueError(f"missing compute sharding for Muon parameter {fqn!r}")
            compute_layout = compute_sharding_by_fqn[fqn]
            parameters_to_prepare.append((param, fqn, compute_layout))

    prepared_compute_layouts = {}
    for param, fqn, compute_layout in parameters_to_prepare:
        global_storage_shape = torch.Size(param.shape)
        compute_view = _matrix_batch_view_from_compute_layout(
            fqn,
            param,
            compute_layout,
        )
        if compute_view is not None:
            if isinstance(param, DTensor):
                _validate_matrix_batch_storage_placements(fqn, param)
        if compute_view is None:
            compute_view_key = ("identity",)
            global_compute_shape = global_storage_shape
        else:
            compute_view_key = ("matrix_batch", compute_view.matrix_rows)
            global_compute_shape = compute_view.matrix_batch_shape(global_storage_shape)
        if len(global_compute_shape) not in (2, 3):
            raise ValueError(
                f"Muon parameter {fqn!r} compute shape "
                f"{tuple(global_compute_shape)} must be 2D or batch-first 3D"
            )
        prepared_compute_layouts[fqn] = _PreparedParameterComputeLayout(
            compute_view_key=compute_view_key,
            compute_layout=compute_layout,
            global_compute_shape=global_compute_shape,
            compute_view=compute_view,
        )
    optimizer._prepared_compute_layouts = prepared_compute_layouts
    tensor_device = optimizer._validate_parameter_storage()
    compute_layouts = optimizer._build_parameter_compute_layouts()
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

    _prepared_compute_layouts: dict[str, _PreparedParameterComputeLayout]
    _specs: tuple[_BucketSpec, ...]
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
        if len(local_devices) != 1 or next(iter(local_devices)).type != "cuda":
            raise ValueError("DistMuon requires one CUDA device per process")
        return local_devices.pop()

    def _build_parameter_compute_layouts(
        self,
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
            prepared = self._prepared_compute_layouts[fqn]
            global_compute_shape = torch.Size(prepared.global_compute_shape)
            resolved_transition = _resolve_storage_to_compute_transition(
                fqn,
                param,
                global_compute_shape,
                prepared.compute_view,
                prepared.compute_layout,
            )
            compute_layouts.append(
                _ParameterComputeLayout(
                    fqn=fqn,
                    param=param,
                    group_index=group_index,
                    compute_view_key=prepared.compute_view_key,
                    global_compute_shape=global_compute_shape,
                    compute_view=prepared.compute_view,
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
        self._bucket_plans = result.plans
        self._parameter_compute_layouts = result.ordered_items

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
            tuple(compute_layout.global_compute_shape),
            compute_layout.storage_is_compute_ready,
            compute_layout.compute_view_key,
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
        if compute_layout.compute_view is not None:
            compute = compute_layout.compute_view.view_as_matrix_batch(compute)
        _compute_muon_direction(
            compute,
            ns_coefficients=group["ns_coefficients"],
            ns_steps=group["ns_steps"],
            eps=group["eps"],
            out=compute,
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
            compute_matrix_shape=compute_layout.global_compute_shape,
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
    matrix_rows: int
    matrix_columns: int

    @classmethod
    def from_storage_shape(
        cls,
        storage_shape: torch.Size,
        *,
        matrix_rows: int,
    ) -> _MatrixBatchView:
        if (
            len(storage_shape) != 2
            or storage_shape[0] == 0
            or storage_shape[0] % matrix_rows
        ):
            raise ValueError(
                f"storage shape {tuple(storage_shape)} cannot be partitioned "
                f"into {matrix_rows}-row Muon matrices"
            )
        return cls(
            matrix_rows=matrix_rows,
            matrix_columns=storage_shape[1],
        )

    def matrix_batch_shape(self, compute_tensor_shape: torch.Size) -> torch.Size:
        if not (
            len(compute_tensor_shape) == 2
            and not compute_tensor_shape[0] % self.matrix_rows
            and compute_tensor_shape[1] == self.matrix_columns
        ):
            raise RuntimeError(
                "compute tensor shape is inconsistent with the prepared "
                "matrix-batch view"
            )
        return torch.Size(
            (
                compute_tensor_shape[0] // self.matrix_rows,
                self.matrix_rows,
                self.matrix_columns,
            )
        )

    def view_as_matrix_batch(self, compute_tensor: Tensor) -> Tensor:
        """Return a zero-copy matrix-batch view of the compute tensor."""
        matrix_batch_shape = self.matrix_batch_shape(torch.Size(compute_tensor.shape))
        return compute_tensor.unflatten(0, matrix_batch_shape[:2])


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
    ``Shard(0)`` and ``BlockShard(0, R)`` may own the same rows when shard
    boundaries align, or different rows when a storage shard splits a block.
    Comparing these intervals distinguishes a local view from redistribution.
    """
    if type(sharding) is Replicate:
        return ((0, num_rows),) * mesh_axis_size

    if type(sharding) is BlockShard:
        assert sharding.dim == 0 and not num_rows % sharding.block_size
        num_sharding_units = num_rows // sharding.block_size
        rows_per_unit = sharding.block_size
    else:
        assert type(sharding) is Shard and sharding.dim == 0
        num_sharding_units = num_rows
        rows_per_unit = 1

    intervals = []
    for axis_coordinate in range(mesh_axis_size):
        local_num_units, unit_offset = Shard.local_shard_size_and_offset(
            num_sharding_units,
            mesh_axis_size,
            axis_coordinate,
        )
        start = unit_offset * rows_per_unit
        intervals.append((start, start + local_num_units * rows_per_unit))
    return tuple(intervals)


def _resolve_storage_to_compute_redistribution_requirement(
    fqn: str,
    param: DTensor,
    compute_view: _MatrixBatchView | None,
    target_sharding_by_storage_mesh_axis: Mapping[int, _AxisComputeSharding],
    declared_storage_mesh_axes: Sequence[int],
) -> tuple[int, ...]:
    """Compare actual storage with the target compute tensor."""
    if compute_view is None:
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
class _PreparedParameterComputeLayout:
    compute_view_key: tuple[Any, ...]
    compute_layout: ComputeLayout
    global_compute_shape: torch.Size
    compute_view: _MatrixBatchView | None


@dataclass(frozen=True, slots=True)
class _ParameterComputeLayout:
    fqn: str
    param: DTensor
    group_index: int
    compute_view_key: tuple[Any, ...]
    global_compute_shape: torch.Size
    compute_view: _MatrixBatchView | None
    storage_mesh_ranks: tuple[int, ...]
    storage_layout_signature: tuple[Any, ...]
    local_storage_signature: tuple[Any, ...]
    compute_sharding: _ResolvedComputeSharding
    storage_to_compute_transition: _StorageToComputeTransition
    resolved_compute_layout_signature: tuple[Any, ...]
    redistribution_storage_mesh_axis: int | None

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


# Per-mesh-axis tensor sharding while resolving ``ComputeLayout``. ``BlockShard``
# remains explicit here, while ``Owned`` is handled separately.
_AxisComputeSharding = Replicate | Shard | BlockShard

# Executor strategy after per-axis shardings are resolved. ``BlockShard`` has
# become ``Shard(0)`` plus ``_MatrixBatchView``; ``Owned`` remains valid here.
_ResolvedComputeSharding = Owned | Replicate | Shard


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
    plans_by_bucket = []
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
        plans_by_bucket.append(
            tuple(
                _build_parameter_redistribution_plan(
                    layout,
                    context.group,
                    owner_rank,
                )
                for layout, owner_rank in zip(
                    context.items,
                    owner_ranks,
                    strict=True,
                )
            )
        )
    return tuple(plans_by_bucket)


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
                    layout.global_compute_shape,
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
    rows, columns = matrix_shape
    short_dim, long_dim = sorted((rows, columns))
    # Each NS step has two s^2 * l matmuls and one s^3 matmul.
    return ns_steps * short_dim * short_dim * (2 * long_dim + short_dim)


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

    storage_regions = _dtensor_storage_regions(
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
    assert type(compute_sharding) is Shard
    if tuple(compute_layout.global_compute_shape) == tuple(compute_layout.param.shape):
        return _build_replicated_to_dim0_shard_plan(
            storage_regions,
            participants=group.participants,
            logical_shape=tuple(compute_layout.global_compute_shape),
        )

    return _build_batched_matrix_redistribution_plan(
        storage_regions,
        participants=group.participants,
        storage_shape=tuple(compute_layout.param.shape),
        compute_shape=tuple(compute_layout.global_compute_shape),
    )


def _build_batched_matrix_redistribution_plan(
    storage_regions: Sequence[tuple[tuple[int, ...], _TensorRegion]],
    *,
    participants: tuple[int, ...],
    storage_shape: tuple[int, ...],
    compute_shape: tuple[int, ...],
) -> _RedistributionPlan:
    """Build flat 2D routes whose destination shards contain complete matrices."""
    _require_valid_plan(
        len(storage_shape) == 2
        and len(compute_shape) == 3
        and storage_shape[0] == compute_shape[0] * compute_shape[1]
        and storage_shape[1] == compute_shape[2],
        "matrix-batch redistribution requires flat 2D storage plus a 3D "
        "matrix-batch compute shape",
    )

    num_matrices, matrix_rows, matrix_columns = compute_shape
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
    for mesh_rank, participant in enumerate(participants):
        local_num_matrices, matrix_offset = Shard.local_shard_size_and_offset(
            num_matrices,
            len(participants),
            mesh_rank,
        )
        logical_region = _TensorRegion(
            offsets=(matrix_offset * matrix_rows, 0),
            shape=(local_num_matrices * matrix_rows, matrix_columns),
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
                matrix_row_offset = matrix_index * matrix_rows
                route_row_offset = max(storage_row_offset, matrix_row_offset)
                route_row_end = min(
                    storage_row_end,
                    matrix_row_offset + matrix_rows,
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
                        local_matrix_index * matrix_rows
                        + route_row_offset
                        - matrix_row_offset,
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


def _resolve_storage_to_compute_transition(
    fqn: str,
    param: DTensor,
    global_compute_shape: torch.Size,
    compute_view: _MatrixBatchView | None,
    compute_layout: ComputeLayout,
) -> _ResolvedStorageToComputeTransition:
    """Validate one storage layout and resolve its concrete compute transition."""
    local = param.to_local()
    if (
        len(global_compute_shape) not in (2, 3)
        or torch.is_complex(param)
        or param.ndim < 2
        or not local.is_contiguous()
    ):
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
    applicable_compute_shardings_by_storage_mesh_axis = {
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

    applicable_owned_storage_mesh_axes = tuple(
        storage_mesh_axis
        for storage_mesh_axis, sharding in (
            applicable_compute_shardings_by_storage_mesh_axis.items()
        )
        if type(sharding) is Owned
    )

    if compute_view is not None:
        if applicable_owned_storage_mesh_axes:
            raise ValueError(
                f"Muon owned compute for parameter {fqn!r} requires a 2D matrix"
            )
        shard_axes = [
            mesh_axis_names[storage_mesh_axis]
            for storage_mesh_axis, sharding in (
                applicable_compute_shardings_by_storage_mesh_axis.items()
            )
            if type(sharding) is Shard
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

    changed_storage_mesh_axes = _resolve_storage_to_compute_redistribution_requirement(
        fqn,
        param,
        compute_view,
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
                    type(redistribution_storage_placement) is Shard
                    and type(redistribution_compute_sharding) in (Shard, BlockShard)
                    and type(placement) is Shard
                ):
                    storage_dim = _normalize_dim(
                        redistribution_storage_placement.dim, param.ndim
                    )
                    preserved_dim = _normalize_dim(placement.dim, param.ndim)
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
        elif type(target_sharding) is BlockShard:
            resolved_shard_dims.append(0)

    resolved_compute_layout_signature = tuple(resolved_target_signature)
    compute_shard_dims = [*resolved_shard_dims, *declared_shard_dims]
    if applicable_owned_storage_mesh_axes and (
        len(global_compute_shape) != 2 or param.ndim != 2
    ):
        raise ValueError(
            f"Muon owned compute for parameter {fqn!r} requires a 2D matrix"
        )
    if active_owned_storage_mesh_axes:
        compute_sharding: _ResolvedComputeSharding = Owned()
    elif compute_shard_dims:
        if compute_view is None and len(global_compute_shape) == 2:
            raise ValueError(
                f"Muon parameter {fqn!r}: 2D Muon compute cannot use Shard; "
                "use Owned() for one matrix or "
                "BlockShard(dim=0, block_size=R) for row-concatenated matrices"
            )
        if len(global_compute_shape) != 3 or any(
            shard_dim != 0 for shard_dim in compute_shard_dims
        ):
            raise ValueError(
                f"Muon sharded compute for parameter {fqn!r} requires a 3D "
                "batch-first tensor sharded only on tensor dimension 0"
            )
        compute_sharding = Shard(0)
    elif applicable_owned_storage_mesh_axes:
        compute_sharding = Owned()
    else:
        raise ValueError(f"unsupported storage-to-compute layout for {fqn!r}")

    if redistribution_storage_mesh_axis is not None and type(compute_sharding) is Shard:
        source_sharding = _normalize_storage_placement(
            param.placements[redistribution_storage_mesh_axis],
            ndim=param.ndim,
            mesh_axis_size=param.device_mesh.size(redistribution_storage_mesh_axis),
        )
        target_sharding = normalized_target_sharding_by_storage_mesh_axis[
            redistribution_storage_mesh_axis
        ]
        if (
            type(source_sharding) is not Replicate
            and type(target_sharding) is not BlockShard
            and source_sharding != target_sharding
        ):
            axis_name = mesh_axis_names[redistribution_storage_mesh_axis]
            raise NotImplementedError(
                f"Muon parameter {fqn!r} cannot yet change tensor sharding "
                f"from {source_sharding} to {target_sharding} on mesh axis "
                f"{axis_name!r}"
            )

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
        placement = cast(Shard, placement)
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if mesh_axis_size == 1:
            return Replicate()
        return Shard(normalized_dim)
    assert type(placement) is BlockShard
    placement = cast(BlockShard, placement)
    normalized_dim = _normalize_dim(placement.dim, ndim)
    if mesh_axis_size == 1:
        return Replicate()
    return BlockShard(normalized_dim, placement.block_size)


def _normalize_storage_placement(
    placement: object,
    *,
    ndim: int,
    mesh_axis_size: int,
) -> Replicate | Shard | _UnsupportedStoragePlacement:
    if mesh_axis_size == 1 or type(placement) is Replicate:
        return Replicate()
    if type(placement) in (Shard, _StridedShard):
        shard = cast(Shard | _StridedShard, placement)
        return Shard(_normalize_dim(shard.dim, ndim))
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
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    out: Tensor,
) -> Tensor:
    """Compute Muon's approximate orthogonal update direction."""
    direction = _zeropower_via_newtonschulz(
        prepared,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )
    out.copy_(direction)
    return out


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
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.placements,
        tuple(local.shape),
        tuple(local.stride()),
        local.dtype,
        local.device,
        local.is_contiguous(),
    )
