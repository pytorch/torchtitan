# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed Muon optimizer implementation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any, cast, overload

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from ..flex_optimizer_reshard import (
    _bind_bucket_configs,
    _BucketedRedistributionRuntime,
    _build_bucket_plans,
    _device_mesh_ranks,
    _validate_bucket_plans_across_ranks,
    BucketConfig,
)
from .storage_to_compute import (
    _compute_distribution_key,
    _ParameterComputeLayout,
    _PreparedParameterComputeView,
    _resolve_muon_redistribution_plans,
    _resolve_storage_to_compute_transition,
)


__all__ = ["DistributedMuon"]


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
        self._set_checkpoint_layout_fingerprints()
        self.register_state_dict_post_hook(_add_layout_fingerprints_to_state_dict)
        self.register_load_state_dict_pre_hook(_validate_layout_fingerprints_on_load)
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

    def _set_checkpoint_layout_fingerprints(self) -> None:
        self._layout_fingerprints_by_fqn = {}
        for layout in self._parameter_compute_layouts:
            descriptor = (
                layout.fqn,
                tuple(layout.param.shape),
                layout.compute_view_key,
                tuple(layout.global_compute_shape),
            )
            self._layout_fingerprints_by_fqn[layout.fqn] = _layout_fingerprint(
                descriptor
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
                compute_layout.storage_is_compute_ready
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


def _add_layout_fingerprints_to_state_dict(
    optimizer: Optimizer,
    state_dict: dict[str, Any],
) -> None:
    muon = cast(DistributedMuon, optimizer)
    for saved_group, current_group in zip(
        state_dict["param_groups"], muon.param_groups, strict=True
    ):
        for param_id, fqn in zip(
            saved_group["params"], current_group["param_names"], strict=True
        ):
            state_dict["state"].setdefault(param_id, {})[
                _LAYOUT_FINGERPRINT_KEY
            ] = muon._layout_fingerprints_by_fqn[fqn]


def _validate_layout_fingerprints_on_load(
    optimizer: Optimizer,
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    muon = cast(DistributedMuon, optimizer)
    saved_groups = state_dict.get("param_groups", ())
    saved_state = state_dict.get("state", {})
    normalized_groups = []
    for saved_group, current_group in zip(saved_groups, muon.param_groups, strict=True):
        for param_id, fqn in zip(
            saved_group["params"], current_group["param_names"], strict=True
        ):
            fingerprint = saved_state.get(param_id, {}).get(_LAYOUT_FINGERPRINT_KEY)
            if fingerprint != muon._layout_fingerprints_by_fqn[fqn]:
                raise ValueError("checkpoint changed DistributedMuon's compute layout")
        normalized_group = dict(saved_group)
        # Optimizer.load_state_dict otherwise replaces canonical current FQNs
        # with checkpoint param_names after positional state matching.
        normalized_group.pop("param_names", None)
        normalized_groups.append(normalized_group)
    return {**state_dict, "param_groups": normalized_groups}


def _after_load_state_dict(optimizer: Optimizer) -> None:
    muon = cast(DistributedMuon, optimizer)
    muon._initialize_plan(muon._parameter_compute_layouts)
    muon._validate_plan_across_ranks()
    muon._first_step_validated = False


_LAYOUT_FINGERPRINT_KEY = "_distributed_muon_layout_fingerprint"
_LAYOUT_FINGERPRINT_VERSION = 1


def _layout_fingerprint(descriptor: tuple[Any, ...]) -> tuple[int, bytes]:
    # Optimizer.load_state_dict rebuilds iterable state values via
    # type(value)(generator), which round-trips bytes but not strings.
    return (
        _LAYOUT_FINGERPRINT_VERSION,
        hashlib.sha256(repr(descriptor).encode()).digest(),
    )


def _local_storage_signature(tensor: Tensor) -> tuple[Any, ...]:
    return (
        tensor.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )
