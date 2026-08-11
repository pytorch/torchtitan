# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed Muon configuration, construction, and optimizer implementation."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Any, cast, NoReturn, overload

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard
from torch.optim import Optimizer
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
)

from torchtitan.distributed.flex_shard._optimizer_reshard_schedule import (
    _bind_bucket_configs,
    _BucketPlanningContext,
    _build_bucket_plans,
    _build_replicated_to_dim0_shard_plan,
    _build_single_participant_redistribution_plan,
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
from torchtitan.distributed.flex_shard.optimizer_reshard import BucketConfig


__all__ = [
    "BatchedMatrixComputeView",
    "build_distributed_muon",
    "DistributedMuon",
    "MuonComputeShardingConfig",
    "Owned",
]


@dataclass(frozen=True, slots=True)
class Owned:
    """Require complete 2D matrix compute on one balanced participant."""


@dataclass(frozen=True, slots=True)
class BatchedMatrixComputeView:
    """View 2D storage as matrices with batch and rows flattened into dim 0."""

    num_matrices: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_matrices, bool)
            or not isinstance(self.num_matrices, int)
            or self.num_matrices <= 0
        ):
            raise ValueError("num_matrices must be a positive integer")

    def _resolve(self, storage_shape: torch.Size) -> _ResolvedBatchedMatrixView:
        if (
            len(storage_shape) != 2
            or storage_shape[0] == 0
            or storage_shape[0] % self.num_matrices
        ):
            raise ValueError(
                f"storage shape {tuple(storage_shape)} cannot be viewed as "
                f"{self.num_matrices} matrices"
            )
        return _ResolvedBatchedMatrixView(
            matrix_rows=storage_shape[0] // self.num_matrices,
            matrix_columns=storage_shape[1],
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class MuonComputeShardingConfig:
    """Define the logical Muon tensor and its compute placement.

    ``Owned`` balances complete 2D matrices across bucket participants.
    ``Shard(0)`` partitions a 3D batch-first tensor ``[B, R, C]`` along the
    batch dimension, where each ``[R, C]`` slice is one matrix.
    Storage is redistributed when it does not already match the requested
    compute placement.
    """

    placement: Owned | Shard

    # Applied before compute placement, so placement dimensions refer to the
    # viewed tensor. A future view_after_placement mode can apply a local view
    # after redistribution; that ordering is not supported yet.
    view_before_placement: BatchedMatrixComputeView | None = None

    def __post_init__(self) -> None:
        if type(self.placement) not in (Owned, Shard):
            raise ValueError(
                "MuonComputeShardingConfig.placement must be Owned or Shard"
            )
        if (
            self.view_before_placement is not None
            and type(self.view_before_placement) is not BatchedMatrixComputeView
        ):
            raise ValueError(
                "MuonComputeShardingConfig.view_before_placement must be "
                "BatchedMatrixComputeView or None"
            )

    def to_dict(self) -> dict:
        """Serialize for JSON logging. Placements become repr strings."""
        return {"repr": repr(self)}


def build_distributed_muon(
    params: Iterable[dict[str, Any]],
    *,
    bucket_configs: Sequence[BucketConfig],
    **kwargs: Any,
) -> DistributedMuon:
    """Prepare named DTensor parameter groups and construct DistributedMuon.

    Every group must provide aligned ``params`` and ``param_names`` plus one
    ``compute_sharding`` contract. Parameter groups, bucket configuration, and
    layouts are frozen after construction because optimizer state and
    collectives depend on them.
    """
    prepared_params = []
    parameters_to_prepare = []
    for param_group in params:
        group = dict(param_group)
        compute_sharding = group.pop("compute_sharding")
        compute_view = compute_sharding.view_before_placement
        group["_compute_placement"] = compute_sharding.placement
        raw_params = group.get("params", ())
        group_params = (
            (raw_params,) if isinstance(raw_params, Tensor) else tuple(raw_params)
        )
        raw_param_names = group.get("param_names")
        param_names = () if raw_param_names is None else tuple(raw_param_names)
        if raw_param_names is None or len(group_params) != len(param_names):
            raise ValueError("params and param_names must be aligned")
        group["params"] = group_params
        group["param_names"] = param_names

        for param, fqn in zip(group_params, param_names, strict=True):
            parameters_to_prepare.append((param, fqn, compute_view))
        prepared_params.append(group)

    prepared_compute_views = {}
    for param, fqn, compute_view in parameters_to_prepare:
        global_storage_shape = torch.Size(param.shape)
        resolved_view = None
        if compute_view is not None and any(
            type(placement) not in (Shard, Replicate)
            for placement in getattr(param, "placements", ())
        ):
            raise ValueError(
                f"batched-matrix Muon parameter {fqn!r} requires exact "
                "Shard or Replicate storage placements"
            )
        if compute_view is not None:
            resolved_view = compute_view._resolve(global_storage_shape)
            if isinstance(param, DTensor):
                _validate_batched_matrix_storage_alignment(
                    fqn,
                    param,
                    resolved_view,
                )
        local_storage = param.to_local() if isinstance(param, DTensor) else param
        local_storage_for_compute_view = (
            local_storage.detach() if isinstance(param, DTensor) else local_storage
        )
        local_storage_shape = torch.Size(local_storage.shape)
        if compute_view is None:
            compute_view_key = ("identity",)
            global_compute_shape = global_storage_shape
            local_storage_view = local_storage_for_compute_view
        else:
            compute_view_key = (
                "batched_matrix",
                compute_view.num_matrices,
                0,
            )
            assert resolved_view is not None
            global_compute_shape = torch.Size(
                (
                    compute_view.num_matrices,
                    resolved_view.matrix_rows,
                    resolved_view.matrix_columns,
                )
            )
            local_storage_view = local_storage_for_compute_view.view(
                resolved_view.compute_shape(local_storage_shape)
            )
        if len(global_compute_shape) not in (2, 3):
            raise ValueError(
                f"Muon parameter {fqn!r} compute shape "
                f"{tuple(global_compute_shape)} must be 2D or batch-first 3D"
            )
        prepared_compute_views[fqn] = _PreparedParameterComputeView(
            compute_view_key=compute_view_key,
            global_compute_shape=global_compute_shape,
            local_storage_view=local_storage_view,
        )
    return DistributedMuon(
        prepared_params,
        _bucket_configs=tuple(bucket_configs),
        _prepared_compute_views=prepared_compute_views,
        **kwargs,
    )


class DistributedMuon(Optimizer):
    """Internal runtime constructed through ``build_distributed_muon``.

    Parameter groups, FQNs, storage layouts, compute layouts, and bucket plans
    are frozen after construction. Every configured parameter must have a
    layout-compatible DTensor gradient before each rank enters ``step()``.

    Batched matrix compute views use batched BF16 kernels. They implement the
    same mathematical update as ``torch.optim.Muon`` running one matrix at a
    time, but bitwise equality across the two kernel schedules is not part of
    the contract.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        _bucket_configs: Sequence[BucketConfig],
        _prepared_compute_views: Mapping[str, _PreparedParameterComputeView],
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
        self._prepared_compute_views = dict(_prepared_compute_views)
        super().__init__(params, defaults)
        tensor_device = self._validate_parameter_storage()
        group_compute_placements = []
        for group in self.param_groups:
            compute_placement = group.pop("_compute_placement", None)
            group_compute_placements.append(compute_placement)
        self._group_compute_placements = tuple(group_compute_placements)

        self._validate_groups()
        compute_layouts = self._build_parameter_compute_layouts()
        self._specs = _bind_bucket_configs(
            _bucket_configs,
            compute_layouts,
            get_fqn=lambda layout: layout.fqn,
            get_storage_dtensor=lambda layout: layout.param,
            requires_redistribution=lambda layout: (
                not layout.storage_is_compute_ready
            ),
            get_required_storage_mesh_axis=lambda layout: (
                layout.redistribution_storage_mesh_axis
            ),
        )
        self._initialize_plan(compute_layouts)
        self._validate_plan_across_ranks()
        self._redistribution_runtime = _BucketedRedistributionRuntime[
            _ParameterComputeLayout
        ](tensor_device)
        self._redistribution_runtime.reserve_buffers(
            self._plans,
            local_tensor_spec=self._local_tensor_spec,
        )
        self.register_load_state_dict_post_hook(_after_load_state_dict)

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
            self._plans,
            local_tensor_spec=self._local_tensor_spec,
            prepare=self._prepare_local,
            compute=self._compute_update,
            finalize=self._apply_update,
        )
        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if hasattr(self, "_plans"):
            raise RuntimeError("DistributedMuon parameter groups are frozen")
        super().add_param_group(param_group)

    def _validate_groups(self) -> None:
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
                raise ValueError(f"unsupported DistributedMuon group {group_index}")

    def _validate_parameter_storage(self) -> torch.device:
        local_devices = set()
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, DTensor):
                    raise TypeError("DistributedMuon requires DTensor parameters")
                local_device = param.to_local().device
                local_devices.add(local_device)
        if len(local_devices) != 1 or next(iter(local_devices)).type != "cuda":
            raise ValueError("DistributedMuon requires one CUDA device per process")
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
            compute_placement = self._group_compute_placements[group_index]
            prepared = self._prepared_compute_views[fqn]
            global_compute_shape = torch.Size(prepared.global_compute_shape)
            local_storage_view = prepared.local_storage_view
            resolved_transition = _resolve_storage_to_compute_transition(
                fqn,
                param,
                global_compute_shape,
                local_storage_view,
                compute_placement,
            )
            compute_layouts.append(
                _ParameterComputeLayout(
                    fqn=fqn,
                    param=param,
                    group_index=group_index,
                    compute_view_key=prepared.compute_view_key,
                    global_compute_shape=global_compute_shape,
                    local_storage_view=local_storage_view,
                    storage_mesh_ranks=_device_mesh_ranks(param.device_mesh),
                    storage_layout_signature=_storage_layout_signature(param),
                    local_storage_signature=_local_storage_signature(param.to_local()),
                    compute_distribution=resolved_transition.compute_distribution,
                    storage_to_compute_transition=resolved_transition.storage_to_compute_transition,
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
        self._plans = result.plans
        self._parameter_compute_layouts = result.ordered_items

    def _validate_plan_across_ranks(self) -> None:
        _validate_bucket_plans_across_ranks(
            self._plans,
            item_signature=self._plan_item_signature,
        )

    def _plan_item_signature(
        self, compute_layout: _ParameterComputeLayout
    ) -> tuple[Any, ...]:
        return (
            compute_layout.fqn,
            compute_layout.group_index,
            tuple(compute_layout.param.shape),
            tuple(compute_layout.param.stride()),
            str(compute_layout.param.dtype),
            compute_layout.param.to_local().device.type,
            tuple(compute_layout.global_compute_shape),
            type(compute_layout.storage_to_compute_transition).__name__,
            compute_layout.compute_view_key,
            _compute_distribution_key(compute_layout.compute_distribution),
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
                "DistributedMuon requires every configured gradient before "
                f"step(); missing gradients: {missing_gradients}"
            )
        if changed_parameter_storage_fqn is not None:
            raise RuntimeError(
                f"parameter local storage changed for "
                f"{changed_parameter_storage_fqn!r}; "
                "rebuild DistributedMuon"
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
        local_reference = (
            compute_layout.local_storage_view
            if compute_layout.storage_is_compute_ready
            else compute_layout.param.to_local()
        )
        assert local_reference is not None
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
            ns_coefficients=group["ns_coefficients"],
            ns_steps=group["ns_steps"],
            eps=group["eps"],
            out=compute,
        )

    def _apply_update(
        self, compute_layout: _ParameterComputeLayout, direction: Tensor
    ) -> None:
        group = self._group(compute_layout)
        local_param = (
            compute_layout.local_storage_view
            if compute_layout.storage_is_compute_ready
            else compute_layout.param.to_local()
        )
        assert local_param is not None
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
        tensor = compute_layout.local_storage_view
        assert tensor is not None
        return tensor.shape, tensor.dtype, tensor.device


@dataclass(frozen=True, slots=True)
class _ResolvedBatchedMatrixView:
    matrix_rows: int
    matrix_columns: int

    def compute_shape(self, storage_shape: torch.Size) -> torch.Size:
        if not (
            len(storage_shape) == 2
            and not storage_shape[0] % self.matrix_rows
            and storage_shape[1] == self.matrix_columns
        ):
            raise RuntimeError(
                "prepared batched-matrix view is internally inconsistent"
            )
        return torch.Size(
            (
                storage_shape[0] // self.matrix_rows,
                self.matrix_rows,
                self.matrix_columns,
            )
        )


def _validate_batched_matrix_storage_alignment(
    fqn: str,
    param: DTensor,
    resolved_view: _ResolvedBatchedMatrixView,
) -> None:
    """Validate every storage shard from globally identical DTensor metadata."""
    for placement in param.placements:
        if type(placement) is Replicate:
            continue
        assert type(placement) is Shard
        if placement.dim % param.ndim != 0:
            raise ValueError(
                f"batched-matrix Muon parameter {fqn!r} requires storage "
                "shards along tensor dimension 0"
            )

    matrix_rows = resolved_view.matrix_rows
    # Every rank must validate all coordinates before DistributedMuon performs
    # collectives; checking only the local shard could strand its peers.
    coordinates = product(
        *(range(mesh_axis_size) for mesh_axis_size in param.device_mesh.shape)
    )
    for coordinate in coordinates:
        local_shape, global_offset = _compute_local_shape_and_global_offset(
            param.shape,
            param.device_mesh.shape,
            list(coordinate),
            param.placements,
        )
        if local_shape[0] and (
            local_shape[0] % matrix_rows or global_offset[0] % matrix_rows
        ):
            raise ValueError(
                f"batched-matrix Muon parameter {fqn!r} storage shards are not "
                f"aligned to matrix rows of size {matrix_rows}"
            )


@dataclass(frozen=True, slots=True)
class _PreparedParameterComputeView:
    compute_view_key: tuple[Any, ...]
    global_compute_shape: torch.Size
    local_storage_view: Tensor | None


@dataclass(frozen=True, slots=True)
class _ParameterComputeLayout:
    fqn: str
    param: DTensor
    group_index: int
    compute_view_key: tuple[Any, ...]
    global_compute_shape: torch.Size
    local_storage_view: Tensor | None
    storage_mesh_ranks: tuple[int, ...]
    storage_layout_signature: tuple[Any, ...]
    local_storage_signature: tuple[Any, ...]
    compute_distribution: _ComputeDistribution
    storage_to_compute_transition: _StorageToComputeTransition
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
class _OwnedRedistributionTransition:
    pass


@dataclass(frozen=True, slots=True)
class _ShardedRedistributionTransition:
    pass


_StorageToComputeTransition = (
    _NoRedistributionTransition
    | _OwnedRedistributionTransition
    | _ShardedRedistributionTransition
)


@dataclass(frozen=True, slots=True)
class _ShardedCompute:
    dim: int


@dataclass(frozen=True, slots=True)
class _SingleRankCompute:
    pass


_ComputeDistribution = _ShardedCompute | _SingleRankCompute


def _compute_distribution_key(
    distribution: _ComputeDistribution,
) -> tuple[str, ...] | tuple[str, int]:
    if isinstance(distribution, _ShardedCompute):
        return ("shard", distribution.dim)
    assert isinstance(distribution, _SingleRankCompute)
    return ("single_rank",)


@dataclass(frozen=True, slots=True)
class _ResolvedStorageToComputeTransition:
    compute_distribution: _ComputeDistribution
    storage_to_compute_transition: _StorageToComputeTransition
    redistribution_storage_mesh_axis: int | None = None


def _resolve_muon_redistribution_plans(
    contexts: tuple[_BucketPlanningContext[_ParameterComputeLayout], ...],
    *,
    ns_steps_by_group: Sequence[int],
) -> tuple[tuple[_RedistributionPlan | None, ...], ...]:
    """Resolve Muon compute placements directly into transport plans."""
    cumulative_loads_by_participants: dict[tuple[int, ...], tuple[int, ...]] = {}
    plans_by_bucket = []
    for context in contexts:
        participants = context.group.participants
        initial_loads = cumulative_loads_by_participants.setdefault(
            participants,
            (0,) * len(participants),
        )
        compute_participants, cumulative_loads = _assign_balanced_single_participants(
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
                    compute_participant,
                )
                for layout, compute_participant in zip(
                    context.items,
                    compute_participants,
                    strict=True,
                )
            )
        )
    return tuple(plans_by_bucket)


def _assign_balanced_single_participants(
    compute_layouts: Sequence[_ParameterComputeLayout],
    *,
    participants: tuple[int, ...],
    cumulative_loads: Sequence[int],
    ns_steps_by_group: Sequence[int],
) -> tuple[tuple[int | None, ...], tuple[int, ...]]:
    """Balance single-participant compute within and across ordered buckets."""
    assignments: list[int | None] = [None] * len(compute_layouts)
    candidates = tuple(
        (index, layout)
        for index, layout in enumerate(compute_layouts)
        if isinstance(layout.compute_distribution, _SingleRankCompute)
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
    compute_participant: int | None,
) -> _RedistributionPlan | None:
    transition = compute_layout.storage_to_compute_transition
    if isinstance(transition, _NoRedistributionTransition):
        return None

    storage_regions = _dtensor_storage_regions(
        compute_layout.param,
        group.participants,
        required_storage_mesh_axis=(compute_layout.redistribution_storage_mesh_axis),
    )
    if isinstance(transition, _OwnedRedistributionTransition):
        assert compute_participant is not None
        assert compute_participant in group.participants
        return _build_single_participant_redistribution_plan(
            storage_regions,
            participants=group.participants,
            compute_participant=compute_participant,
            logical_shape=tuple(compute_layout.param.shape),
        )

    assert compute_participant is None
    assert isinstance(transition, _ShardedRedistributionTransition)
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
    """Map flat row storage to sharded matrix batches."""
    _require_valid_plan(
        len(storage_shape) == 2
        and len(compute_shape) == 3
        and storage_shape[0] == compute_shape[0] * compute_shape[1]
        and storage_shape[1] == compute_shape[2],
        "matrix-batch redistribution requires a flattened 2D view",
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
                tensor_shape=(local_num_matrices, matrix_rows, matrix_columns),
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
                        local_matrix_index,
                        route_row_offset - matrix_row_offset,
                        0,
                    ),
                    shape=(1, route_rows, matrix_columns),
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
    local_storage_view: Tensor | None,
    compute_placement: object,
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

    replicated_storage = _has_replicated_storage(param)
    storage_can_redistribute, storage_shard_axis = _redistribution_storage_shard_axis(
        param
    )
    mesh_size = param.device_mesh.size()
    if isinstance(compute_placement, Shard):
        if len(global_compute_shape) == 3 and (
            _normalize_dim(compute_placement.dim, len(global_compute_shape)) == 0
        ):
            if replicated_storage:
                return _ResolvedStorageToComputeTransition(
                    compute_distribution=_ShardedCompute(0),
                    storage_to_compute_transition=(
                        _NoRedistributionTransition()
                        if mesh_size == 1
                        else _ShardedRedistributionTransition()
                    ),
                )
            if (
                local_storage_view is not None
                and local_storage_view.shape[1:] == global_compute_shape[1:]
                and _has_dim0_sharded_storage(param)
            ):
                return _ResolvedStorageToComputeTransition(
                    compute_distribution=_ShardedCompute(0),
                    storage_to_compute_transition=_NoRedistributionTransition(),
                )
    elif (
        isinstance(compute_placement, Owned)
        and len(global_compute_shape) == 2
        and param.ndim == 2
    ):
        if storage_can_redistribute:
            return _ResolvedStorageToComputeTransition(
                compute_distribution=_SingleRankCompute(),
                storage_to_compute_transition=(
                    _NoRedistributionTransition()
                    if mesh_size == 1
                    else _OwnedRedistributionTransition()
                ),
                redistribution_storage_mesh_axis=storage_shard_axis,
            )
    _raise_unsupported_layout(fqn)


def _raise_unsupported_layout(fqn: str) -> NoReturn:
    raise ValueError(f"unsupported storage-to-compute layout for {fqn!r}")


def _has_replicated_storage(param: DTensor) -> bool:
    return all(type(placement) is Replicate for placement in param.placements)


def _has_dim0_sharded_storage(param: DTensor) -> bool:
    has_shard = False
    for placement in param.placements:
        # FSDP2 emits _StridedShard when a later TP/EP axis already shards
        # this dimension. Keep the allowlist exact so new placements fail closed.
        if type(placement) in (Shard, _StridedShard):
            shard = cast(Shard | _StridedShard, placement)
            if shard.dim % param.ndim != 0:
                return False
            has_shard = True
        elif type(placement) is not Replicate:
            return False
    return has_shard


def _redistribution_storage_shard_axis(param: DTensor) -> tuple[bool, int | None]:
    """Recognize storage replicated outside at most one exact Shard axis."""
    storage_shard_axis = None
    for mesh_axis, placement in enumerate(param.placements):
        if type(placement) is Replicate:
            continue
        if type(placement) is not Shard or storage_shard_axis is not None:
            return False, None
        storage_shard_axis = mesh_axis
    return True, storage_shard_axis


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
    muon = cast(DistributedMuon, optimizer)
    # Optimizer.load_state_dict restores group values such as ns_steps after
    # construction, and those values affect compute planning and buffer sizes.
    muon._validate_groups()
    muon._initialize_plan(muon._parameter_compute_layouts)
    muon._validate_plan_across_ranks()
    muon._redistribution_runtime.reserve_buffers(
        muon._plans,
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
