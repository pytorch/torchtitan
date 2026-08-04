# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone bucketed Distributed Muon optimizer."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.optim import Optimizer
from .bucketed_redistribution import (
    _BucketedRedistributionRuntime,
    _BucketPlan,
    _build_bucket_plans,
    _device_mesh_ranks,
    _validate_bucket_plans_across_ranks,
    assign_balanced_owners,
    BucketSpec,
)


__all__ = ["BucketSpec", "assign_balanced_owners", "Owned"]



@dataclass(frozen=True, slots=True)
class Owned:
    """Require a complete matrix; sharded storage uses a ``BucketSpec`` owner."""


@dataclass(frozen=True, slots=True)
class _PreparedParameterComputeView:
    global_compute_shape: torch.Size
    local_compute_tensor: Tensor


class DistributedMuon(Optimizer):
    """Internal runtime constructed through ``build_distributed_muon``."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        bucket_spec: Sequence[BucketSpec],
        _prepared_compute_views: Mapping[
            str, _PreparedParameterComputeView
        ],
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
        params = [
            dict(param_or_group)
            if isinstance(param_or_group, dict)
            else param_or_group
            for param_or_group in params
        ]
        self._first_step_validated = False
        self._prepared_compute_views = dict(_prepared_compute_views)
        super().__init__(params, defaults)
        assert all(
            isinstance(param, DTensor) and param.device.type == "cuda"
            for group in self.param_groups
            for param in group["params"]
        ), "DistributedMuon requires CUDA DTensor parameters"
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
        ](self._tensor_device)
        self._frozen_param_names = tuple(
            tuple(group.get("param_names", ())) for group in self.param_groups
        )

    @torch.no_grad()
    def step(
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._preflight_step()
        self._redistribution_runtime.run(
            self._plans,
            local_tensor_spec=self._local_tensor_spec,
            compute_shape=self._compute_shape,
            prepare=self._prepare_local,
            compute=self._compute_update,
            finalize=self._apply_update,
        )
        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if hasattr(self, "_plans"):
            raise RuntimeError(
                "DistributedMuon parameter groups are frozen"
            )
        super().add_param_group(param_group)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Compute layout is intentionally not duplicated in optimizer state. TorchTitan
        # must reconstruct it from the same model and optimizer config before resume.
        saved_groups = state_dict.get("param_groups", ())
        if len(saved_groups) != len(self._frozen_param_names) or any(
            "param_names" in saved and tuple(saved["param_names"]) != names
            for saved, names in zip(
                saved_groups, self._frozen_param_names, strict=True
            )
        ):
            raise ValueError("checkpoint changed DistributedMuon's parameter groups")
        super().load_state_dict(state_dict)
        self._validate_plan_across_ranks()
        self._first_step_validated = False

    def _validate_groups(self) -> None:
        for group_index, group in enumerate(self.param_groups):
            if group.get("fused") or group.get("foreach"):
                raise NotImplementedError(
                    "DistributedMuon does not support fused or foreach"
                )
            ns_steps = group["ns_steps"]
            coefficients = group["ns_coefficients"]
            if (
                any(
                    group[name] < 0
                    for name in ("lr", "weight_decay", "momentum", "eps")
                )
                or not isinstance(ns_steps, int)
                or not 0 <= ns_steps < 100
                or len(coefficients) != 3
                or not all(isinstance(value, (int, float)) for value in coefficients)
                or group["adjust_lr_fn"]
                not in (None, "original", "match_rms_adamw", "spectral_unclamped")
            ):
                raise ValueError(f"invalid DistributedMuon group {group_index}")

    def _build_parameter_compute_layouts(
        self,
    ) -> tuple[_ParameterComputeLayout, ...]:
        parameters = []
        seen_names = set()
        seen_params = set()
        for group_index, group in enumerate(self.param_groups):
            params = group["params"]
            names = group.get("param_names")
            if names is None or len(names) != len(params):
                raise ValueError(
                    "DistributedMuon requires param_names aligned with params"
                )
            for fqn, param in zip(names, params, strict=True):
                if fqn in seen_names or id(param) in seen_params:
                    raise ValueError(f"duplicate Muon parameter {fqn!r}")
                seen_names.add(fqn)
                seen_params.add(id(param))
                parameters.append((group_index, fqn, param))

        prepared_fqns = self._prepared_compute_views.keys()
        if prepared_fqns != seen_names:
            raise ValueError(
                "prepared compute views must exactly cover parameter FQNs; "
                f"missing={sorted(seen_names - prepared_fqns)}, "
                f"extra={sorted(prepared_fqns - seen_names)}"
            )
        compute_layouts = []
        for group_index, fqn, param in parameters:
            compute_placement = self._group_compute_placements[group_index]
            prepared = self._prepared_compute_views[fqn]
            if not isinstance(prepared, _PreparedParameterComputeView):
                raise TypeError(
                    f"invalid prepared compute view for parameter {fqn!r}"
                )
            global_compute_shape = torch.Size(prepared.global_compute_shape)
            local_compute_tensor = prepared.local_compute_tensor
            compute_locally = _validate_muon_parameter(
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
                    global_compute_shape=global_compute_shape,
                    local_compute_tensor=local_compute_tensor,
                    compute_placement=compute_placement,
                    compute_locally=compute_locally,
                )
            )
        return tuple(compute_layouts)

    def _initialize_plan(self) -> None:
        compute_layouts = self._build_parameter_compute_layouts()
        result = _build_bucket_plans(
            compute_layouts,
            self._specs,
            fqn=lambda item: item.fqn,
            compute_locally=lambda item: item.compute_locally,
            storage_dtensor=lambda item: item.param,
        )
        self._plans = result.plans
        self._parameter_compute_layouts = result.ordered_items
        self._tensor_device = self._plans[0].device

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
            compute_layout.compute_locally,
            _compute_placement_key(compute_layout.compute_placement),
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
                and compute_layout.param.to_local().untyped_storage().data_ptr()
                != compute_layout.local_compute_tensor.untyped_storage().data_ptr()
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
            lr=group["lr"],
            ns_coefficients=group["ns_coefficients"],
            ns_steps=group["ns_steps"],
            eps=group["eps"],
            adjust_lr_fn=group["adjust_lr_fn"],
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

    @staticmethod
    def _compute_shape(
        compute_layout: _ParameterComputeLayout,
    ) -> torch.Size:
        return compute_layout.global_compute_shape

@dataclass(frozen=True, slots=True)
class _ParameterComputeLayout:
    fqn: str
    param: DTensor
    group_index: int
    global_compute_shape: torch.Size
    local_compute_tensor: Tensor
    compute_placement: Owned | Shard
    compute_locally: bool


def _has_replicated_storage(param: DTensor) -> bool:
    return all(type(placement) is Replicate for placement in param.placements)


def _has_dim0_sharded_storage(param: DTensor) -> bool:
    has_shard = False
    for placement in param.placements:
        # FSDP2 emits _StridedShard when a later TP/EP axis already shards
        # this dimension. Keep the allowlist exact so new placements fail closed.
        if type(placement) in (Shard, _StridedShard):
            if getattr(placement, "dim") % param.ndim != 0:
                return False
            has_shard = True
        elif type(placement) is not Replicate:
            return False
    return has_shard


def _validate_muon_parameter(
    fqn: str,
    param: DTensor,
    global_compute_shape: torch.Size,
    local_compute_tensor: Tensor,
    compute_placement: object,
) -> bool:
    local = param.to_local()
    if (
        torch.is_complex(param)
        or param.ndim < 2
        or not local.is_contiguous()
        or tuple(param.stride())
        != tuple(torch.empty(param.shape, device="meta").stride())
    ):
        raise ValueError(
            f"Muon parameter {fqn!r} has unsupported shape or storage"
        )

    if (
        len(global_compute_shape) < 2
        or local_compute_tensor.ndim < 2
        or math.prod(global_compute_shape) != param.numel()
        or local_compute_tensor.numel() != local.numel()
        or local_compute_tensor.dtype != local.dtype
        or local_compute_tensor.device != local.device
        or not local_compute_tensor.is_contiguous()
        or local_compute_tensor.data_ptr() != local.data_ptr()
    ):
        raise ValueError(
            f"invalid prepared compute view for parameter {fqn!r}"
        )

    if compute_placement is None:
        raise ValueError(
            f"Muon parameter {fqn!r} requires explicit compute_placement"
        )

    replicated_storage = _has_replicated_storage(param)
    if isinstance(compute_placement, Shard):
        if len(global_compute_shape) < 3:
            raise ValueError(
                "compute Shard requires a batch of complete Muon matrices"
            )
        compute_dim = _normalize_dim(
            compute_placement.dim, len(global_compute_shape)
        )
        if compute_dim != 0:
            raise ValueError("DistributedMuon currently supports compute Shard(0)")
        if local_compute_tensor.ndim != len(global_compute_shape):
            raise ValueError(
                f"compute Shard(0) for {fqn!r} must keep complete matrices local"
            )
        if replicated_storage:
            if local_compute_tensor.shape != global_compute_shape:
                raise ValueError(
                    f"replicated storage for {fqn!r} must contain the complete "
                    "compute tensor"
                )
        elif (
            local_compute_tensor.shape[1:] != global_compute_shape[1:]
            or not _has_dim0_sharded_storage(param)
        ):
            raise ValueError(
                f"compute Shard(0) for {fqn!r} must already match storage sharding"
            )
        return True
    elif not isinstance(compute_placement, Owned):
        raise TypeError(f"unsupported compute placement {compute_placement!r}")
    elif len(global_compute_shape) != 2 or param.ndim != 2:
        raise ValueError(
            f"owned Muon parameter {fqn!r} requires matrix storage"
        )
    elif replicated_storage:
        if local_compute_tensor.shape != global_compute_shape:
            raise ValueError(
                f"replicated storage for {fqn!r} must contain the complete "
                "compute tensor"
            )
        return True
    elif (
        param.device_mesh.ndim != 1
        or len(param.placements) != 1
        or type(param.placements[0]) is not Shard
    ):
        raise ValueError(
            f"owned Muon parameter {fqn!r} requires replicated or 1D Shard "
            "matrix storage"
        )
    return False


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim if dim >= 0 else dim + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dimension {dim} is invalid for a rank-{ndim} tensor")
    return normalized


def _compute_placement_key(
    placement: Owned | Shard,
) -> tuple[Any, ...]:
    if isinstance(placement, Owned):
        return ("owned",)
    return ("shard", placement.dim)


# Keep the functional math aligned with torch.optim.Muon while owning the
# implementation here so the distributed runtime has no Muon dependency.
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


def _adjust_learning_rate(
    lr: float,
    adjust_lr_fn: str | None,
    compute_matrix_shape: torch.Size,
) -> float:
    rows, columns = compute_matrix_shape[-2:]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1, rows / columns))
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
    lr: float,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    out: Tensor,
) -> tuple[Tensor, float]:
    direction = _zeropower_via_newtonschulz(
        prepared,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )
    adjusted_lr = _adjust_learning_rate(lr, adjust_lr_fn, prepared.shape)
    # Pre-scaling the direction can change FP32 rounding versus Muon's add_.
    out.copy_(direction)
    return out, adjusted_lr
