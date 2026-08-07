# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muon compute placement and the internal DistributedMuon runtime."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, overload

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.optim import Optimizer

from .flex_optimizer_reshard import (
    _BucketedRedistributionRuntime,
    _build_bucket_plans,
    _build_owned_redistribution_plan,
    _device_mesh_ranks,
    _dtensor_storage_blocks,
    _RedistributionGroup,
    _RedistributionPlan,
    _validate_bucket_plans_across_ranks,
    assign_balanced_owners,
    BucketSpec,
)


__all__ = ["BucketSpec", "assign_balanced_owners", "Owned"]


@dataclass(frozen=True, slots=True)
class Owned:
    """Require complete 2D matrix compute.

    This is a Muon compute placement, not a DTensor storage placement.
    Replicated storage computes locally; sharded storage uses the parameter's
    mesh-local owner from ``BucketSpec.owner_rank_by_fqn``.
    """


class DistributedMuon(Optimizer):
    """Internal runtime constructed through ``build_distributed_muon``.

    Parameter groups, FQNs, storage layouts, compute layouts, and bucket plans
    are frozen after construction. Every configured parameter must have a
    layout-compatible DTensor gradient before each rank enters ``step()``.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        bucket_spec: Sequence[BucketSpec],
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

        self._specs = tuple(bucket_spec)
        self._validate_groups()
        self._initialize_plan()
        self._validate_plan_across_ranks()
        self._redistribution_runtime = _BucketedRedistributionRuntime[
            _ParameterComputeLayout
        ](tensor_device)
        self._set_checkpoint_layout_fingerprints()

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

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        for saved_group, current_group in zip(
            state_dict["param_groups"], self.param_groups, strict=True
        ):
            for param_id, fqn in zip(
                saved_group["params"], current_group["param_names"], strict=True
            ):
                state_dict["state"].setdefault(param_id, {})[
                    _LAYOUT_FINGERPRINT_KEY
                ] = self._layout_fingerprints_by_fqn[fqn]
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_groups = state_dict.get("param_groups", ())
        saved_state = state_dict.get("state", {})
        for saved_group, current_group in zip(
            saved_groups, self.param_groups, strict=True
        ):
            for param_id, fqn in zip(
                saved_group["params"], current_group["param_names"], strict=True
            ):
                fingerprint = saved_state.get(param_id, {}).get(_LAYOUT_FINGERPRINT_KEY)
                if fingerprint != self._layout_fingerprints_by_fqn[fqn]:
                    raise ValueError(
                        "checkpoint changed DistributedMuon's compute layout"
                    )
        super().load_state_dict(state_dict)
        self._validate_plan_across_ranks()
        self._first_step_validated = False

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
            local_compute_tensor = prepared.local_compute_tensor
            resolved_compute_placement = _resolve_compute_placement(
                fqn,
                param,
                global_compute_shape,
                local_compute_tensor,
                compute_placement,
            )
            compute_layouts.append(
                _ParameterComputeLayout(
                    fqn=fqn,
                    param=param,
                    group_index=group_index,
                    compute_view_key=prepared.compute_view_key,
                    global_compute_shape=global_compute_shape,
                    local_compute_tensor=local_compute_tensor,
                    local_storage_signature=_local_storage_signature(param.to_local()),
                    compute_placement_key=resolved_compute_placement.fingerprint_key,
                    compute_transition=resolved_compute_placement.compute_transition,
                )
            )
        return tuple(compute_layouts)

    def _initialize_plan(self) -> None:
        compute_layouts = self._build_parameter_compute_layouts()
        result = _build_bucket_plans(
            compute_layouts,
            self._specs,
            fqn=lambda item: item.fqn,
            requires_owner=lambda item: item.requires_owner,
            storage_dtensor=lambda item: item.param,
            redistribution_plan=_build_compute_redistribution_plan,
        )
        self._plans = result.plans
        self._parameter_compute_layouts = result.ordered_items

    def _set_checkpoint_layout_fingerprints(self) -> None:
        self._layout_fingerprints_by_fqn = {}
        for layout in self._parameter_compute_layouts:
            descriptor = (
                layout.fqn,
                tuple(layout.param.shape),
                layout.compute_view_key,
                tuple(layout.global_compute_shape),
                layout.compute_placement_key,
            )
            self._layout_fingerprints_by_fqn[layout.fqn] = (
                _LAYOUT_FINGERPRINT_VERSION,
                # Optimizer.load_state_dict rebuilds iterable state values via
                # type(value)(generator), which round-trips bytes but not strings.
                hashlib.sha256(repr(descriptor).encode()).digest(),
            )

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
            type(compute_layout.compute_transition).__name__,
            compute_layout.compute_view_key,
            compute_layout.compute_placement_key,
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
        missing_gradients = [
            compute_layout.fqn
            for compute_layout in self._parameter_compute_layouts
            if compute_layout.param.grad is None
        ]
        if missing_gradients:
            raise RuntimeError(
                "DistributedMuon requires every configured gradient before "
                f"step(); missing gradients: {missing_gradients}"
            )

        for compute_layout in self._parameter_compute_layouts:
            if (
                compute_layout.compute_locally
                and _local_storage_signature(compute_layout.param.to_local())
                != compute_layout.local_storage_signature
            ):
                raise RuntimeError(
                    f"parameter local storage changed for {compute_layout.fqn!r}; "
                    "rebuild DistributedMuon"
                )
        gradients = []
        for compute_layout in self._parameter_compute_layouts:
            grad = self._gradient(compute_layout)
            gradients.append((compute_layout, grad))
            if initialize_state:
                self._validate_momentum(compute_layout)

        # State creation happens only after every gradient and existing state
        # tensor has passed validation, so a deterministic input error cannot
        # partially update an earlier bucket.
        if initialize_state:
            for compute_layout, grad in gradients:
                self._momentum(compute_layout, grad)
            self._first_step_validated = True

    @staticmethod
    def _has_storage_layout(
        tensor: DTensor, compute_layout: _ParameterComputeLayout
    ) -> bool:
        local = tensor.to_local()
        param_local = compute_layout.param.to_local()
        return (
            tensor.shape == compute_layout.param.shape
            and tensor.stride() == compute_layout.param.stride()
            and _device_mesh_ranks(tensor.device_mesh)
            == _device_mesh_ranks(compute_layout.param.device_mesh)
            and tensor.placements == compute_layout.param.placements
            and local.shape == param_local.shape
            and local.stride() == param_local.stride()
            and local.dtype == param_local.dtype
            and local.device == param_local.device
            and local.is_contiguous()
        )

    def _gradient(self, compute_layout: _ParameterComputeLayout) -> DTensor:
        grad = compute_layout.param.grad
        if not isinstance(grad, DTensor) or not self._has_storage_layout(
            grad, compute_layout
        ):
            raise RuntimeError(
                f"gradient storage layout changed for {compute_layout.fqn!r}"
            )
        return grad

    def _validate_momentum(self, compute_layout: _ParameterComputeLayout) -> None:
        state = self.state.get(compute_layout.param, {})
        fingerprint = state.get(_LAYOUT_FINGERPRINT_KEY)
        if (
            fingerprint is not None
            and fingerprint != self._layout_fingerprints_by_fqn[compute_layout.fqn]
        ):
            raise RuntimeError(
                f"optimizer state layout changed for {compute_layout.fqn!r}"
            )
        momentum = state.get("momentum_buffer")
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
        state[_LAYOUT_FINGERPRINT_KEY] = self._layout_fingerprints_by_fqn[
            compute_layout.fqn
        ]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(
                grad, memory_format=torch.preserve_format
            )
        return state["momentum_buffer"]

    def _update_local_momentum(
        self, compute_layout: _ParameterComputeLayout
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        grad = cast(DTensor, compute_layout.param.grad)
        momentum = cast(DTensor, self.state[compute_layout.param]["momentum_buffer"])
        local_grad = grad.to_local().view_as(compute_layout.local_compute_tensor)
        local_momentum = momentum.to_local().view_as(
            compute_layout.local_compute_tensor
        )
        group = self._group(compute_layout)
        local_momentum.lerp_(local_grad, 1 - group["momentum"])
        torch.autograd.graph.increment_version(momentum)
        return local_grad, local_momentum, group

    @staticmethod
    def _write_prepared(
        group: dict[str, Any], grad: Tensor, momentum: Tensor, out: Tensor
    ) -> None:
        if group["nesterov"]:
            torch.lerp(
                grad,
                momentum,
                group["momentum"],
                out=out,
            )
        else:
            out.copy_(momentum)

    def _prepare_local(
        self, compute_layout: _ParameterComputeLayout, out: Tensor
    ) -> None:
        grad, momentum, group = self._update_local_momentum(compute_layout)
        self._write_prepared(group, grad, momentum, out)

    def _compute_update(
        self, compute_layout: _ParameterComputeLayout, compute: Tensor
    ) -> None:
        group = self._group(compute_layout)
        _compute_muon_update(
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
            compute_layout.local_compute_tensor
            if compute_layout.compute_locally
            else compute_layout.param.to_local()
        )
        adjusted_lr = _adjust_learning_rate(
            group["lr"],
            group["adjust_lr_fn"],
            compute_layout.global_compute_shape,
        )
        local_param.mul_(1 - group["lr"] * group["weight_decay"])
        local_param.add_(direction, alpha=-adjusted_lr)
        torch.autograd.graph.increment_version(compute_layout.param)

    @staticmethod
    def _local_tensor_spec(
        compute_layout: _ParameterComputeLayout,
    ) -> tuple[torch.Size, torch.dtype, torch.device]:
        tensor = compute_layout.local_compute_tensor
        return tensor.shape, tensor.dtype, tensor.device


_LAYOUT_FINGERPRINT_KEY = "_distributed_muon_layout_fingerprint"
_LAYOUT_FINGERPRINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class _PreparedParameterComputeView:
    compute_view_key: tuple[Any, ...]
    global_compute_shape: torch.Size
    local_compute_tensor: Tensor


@dataclass(frozen=True, slots=True)
class _ParameterComputeLayout:
    fqn: str
    param: DTensor
    group_index: int
    compute_view_key: tuple[Any, ...]
    global_compute_shape: torch.Size
    local_compute_tensor: Tensor
    local_storage_signature: tuple[Any, ...]
    compute_placement_key: tuple[Any, ...]
    compute_transition: _ComputeTransition

    @property
    def compute_locally(self) -> bool:
        return isinstance(self.compute_transition, _LocalCompute)

    @property
    def requires_owner(self) -> bool:
        return isinstance(self.compute_transition, _OwnedCompute)


@dataclass(frozen=True, slots=True)
class _LocalCompute:
    pass


@dataclass(frozen=True, slots=True)
class _OwnedCompute:
    pass


_ComputeTransition = _LocalCompute | _OwnedCompute


@dataclass(frozen=True, slots=True)
class _ResolvedComputePlacement:
    fingerprint_key: tuple[Any, ...]
    compute_transition: _ComputeTransition


def _build_compute_redistribution_plan(
    compute_layout: _ParameterComputeLayout,
    group: _RedistributionGroup,
    owner_rank: int | None,
) -> _RedistributionPlan | None:
    transition = compute_layout.compute_transition
    if isinstance(transition, _LocalCompute):
        return None

    assert isinstance(transition, _OwnedCompute)
    assert owner_rank is not None
    return _build_owned_redistribution_plan(
        _dtensor_storage_blocks(compute_layout.param, group.participants),
        participants=group.participants,
        owner=group.participants[owner_rank],
        logical_shape=tuple(compute_layout.param.shape),
    )


def _resolve_compute_placement(
    fqn: str,
    param: DTensor,
    global_compute_shape: torch.Size,
    local_compute_tensor: Tensor,
    compute_placement: object,
) -> _ResolvedComputePlacement:
    """Validate and canonicalize one storage-to-compute transition.

    ``Owned`` accepts a 2D matrix stored as exact ``Shard`` on a 1D mesh and
    redistributes it to its configured owner. ``Shard(0)`` accepts matrix
    batches whose storage shards keep each matrix whole. Fully replicated
    storage computes locally under either compatible declaration, including
    when a size-one mesh axis has normalized sharded storage to replication.
    """
    local = param.to_local()
    if torch.is_complex(param) or param.ndim < 2 or not local.is_contiguous():
        raise ValueError(f"Muon parameter {fqn!r} has unsupported shape or storage")

    replicated_storage = _has_replicated_storage(param)
    if isinstance(compute_placement, Shard):
        if (
            len(global_compute_shape) >= 3
            and _normalize_dim(compute_placement.dim, len(global_compute_shape)) == 0
            and (
                (
                    replicated_storage
                    and local_compute_tensor.shape == global_compute_shape
                )
                or (
                    local_compute_tensor.shape[1:] == global_compute_shape[1:]
                    and _has_dim0_sharded_storage(param)
                )
            )
        ):
            return _ResolvedComputePlacement(
                fingerprint_key=("shard", 0),
                compute_transition=_LocalCompute(),
            )
    elif (
        isinstance(compute_placement, Owned)
        and len(global_compute_shape) == 2
        and param.ndim == 2
    ):
        if replicated_storage:
            return _ResolvedComputePlacement(
                fingerprint_key=("owned",),
                compute_transition=_LocalCompute(),
            )
        if _has_owned_sharded_storage(param):
            return _ResolvedComputePlacement(
                fingerprint_key=("owned",),
                compute_transition=_OwnedCompute(),
            )
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


def _has_owned_sharded_storage(param: DTensor) -> bool:
    return (
        param.device_mesh.ndim == 1
        and len(param.placements) == 1
        and type(param.placements[0]) is Shard
    )


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim if dim >= 0 else dim + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dimension {dim} is invalid for a rank-{ndim} tensor")
    return normalized


def _local_storage_signature(tensor: Tensor) -> tuple[Any, ...]:
    return (
        tensor.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


# Keep the functional math aligned with torch.optim.Muon while owning the
# implementation here so the distributed runtime has no Muon dependency.
def _adjust_learning_rate(
    lr: float,
    adjust_lr_fn: str | None,
    compute_matrix_shape: torch.Size,
) -> float:
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


def _compute_muon_update(
    prepared: Tensor,
    *,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    out: Tensor,
) -> Tensor:
    direction = _zeropower_via_newtonschulz(
        prepared,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )
    out.copy_(direction)
    return out


def _zeropower_via_newtonschulz(
    update: Tensor,
    *,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Compute Muon's approximate polar factor without using torch.optim.Muon."""
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
        for _ in range(ns_steps):
            gram = matrices @ matrices.transpose(-2, -1)
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            matrices = torch.baddbmm(matrices, gram_update, matrices, beta=a)
        result = matrices.reshape(original_shape)

    return result.transpose(-2, -1) if transposed else result
