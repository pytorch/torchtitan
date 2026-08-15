# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexShard configuration and execution for ``torch.optim.Muon``."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast, NoReturn, Protocol, TypeAlias
from weakref import ref, ReferenceType, WeakKeyDictionary

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.optim import Muon, Optimizer
from torch.utils.hooks import RemovableHandle

from ._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
    _BufferSlot,
    _LocalBucketExecutor,
)

from ._optimizer_reshard_schedule import (
    _bind_bucket_configs,
    _BucketExecutionPlan,
    _BucketPlanningContext,
    _BucketPlanningResult,
    _BucketSpec,
    _build_bucket_plans,
    _build_dim0_shard_redistribution_plan,
    _build_owned_redistribution_plan,
    _build_whole_tensor_redistribution_plan,
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
    BlockShard,
    BucketConfig,
    ComputeLayout,
    flex_optimizer_reshard,
    Owned,
)


__all__ = [
    "build_flex_shard_muon",
]


_ParameterGroupMembershipSignature: TypeAlias = tuple[tuple[tuple[int, str], ...], ...]
_RESHARD_INTEGRATIONS: WeakKeyDictionary[
    Muon, ReferenceType[_MuonReshardIntegration]
] = WeakKeyDictionary()


class _MuonPrepare(Protocol):
    def __call__(
        self,
        gradient: Tensor,
        momentum_buffer: Tensor,
        *,
        momentum: float,
        nesterov: bool,
        out: Tensor,
    ) -> Tensor:
        ...


class _MuonOrthogonalize(Protocol):
    def __call__(
        self,
        prepared: Tensor,
        *,
        ns_coefficients: tuple[float, float, float],
        ns_steps: int,
        eps: float,
        out: Tensor,
    ) -> Tensor:
        ...


class _MuonApply(Protocol):
    def __call__(
        self,
        parameter: Tensor,
        direction: Tensor,
        *,
        lr: float | Tensor,
        weight_decay: float,
        adjust_lr_fn: str | None,
        logical_matrix_shape: torch.Size,
    ) -> Tensor:
        ...


class _RegisterStepExecutor(Protocol):
    def __call__(
        self,
        executor: _MuonReshardIntegration,
    ) -> RemovableHandle:
        ...


@dataclass(frozen=True, slots=True)
class _TorchMuonApis:
    prepare: _MuonPrepare
    orthogonalize: _MuonOrthogonalize
    apply: _MuonApply
    register_step_executor: _RegisterStepExecutor


def _require_torch_muon_apis(
    optimizer: Muon,
) -> _TorchMuonApis:
    register = getattr(optimizer, "register_step_executor", None)
    if not callable(register):
        raise RuntimeError(
            "FlexShard Muon requires a PyTorch build with "
            "Muon.register_step_executor support"
        )
    operations = {
        name: getattr(torch.optim, name, None)
        for name in ("muon_prepare", "muon_orthogonalize", "muon_apply")
    }
    missing_operations = tuple(
        name for name, operation in operations.items() if not callable(operation)
    )
    if missing_operations:
        raise RuntimeError(
            "FlexShard Muon requires a PyTorch build with public torch.optim "
            f"operations {missing_operations}"
        )
    return _TorchMuonApis(
        prepare=cast(_MuonPrepare, operations["muon_prepare"]),
        orthogonalize=cast(
            _MuonOrthogonalize,
            operations["muon_orthogonalize"],
        ),
        apply=cast(_MuonApply, operations["muon_apply"]),
        register_step_executor=cast(_RegisterStepExecutor, register),
    )


def _get_muon_reshard_integration(optimizer: Muon) -> _MuonReshardIntegration:
    integration_ref = _RESHARD_INTEGRATIONS.get(optimizer)
    integration = None if integration_ref is None else integration_ref()
    if integration is None:
        raise RuntimeError("Muon is not configured with flex_optimizer_reshard")
    return integration


def build_flex_shard_muon(
    params: Iterable[dict[str, Any]],
    *,
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    bucket_configs: Sequence[BucketConfig],
    **kwargs: Any,
) -> Muon:
    """Construct a ``torch.optim.Muon`` with FlexShard redistribution.

    ``BlockShard(dim=0, block_size=R)`` interprets a parameter
    stored as ``[M * R, C]`` as ``M`` independent matrices ``[M, R, C]`` for
    optimizer-local Muon compute. A native 3D ``[M, R, C]`` parameter can use
    ``Shard(0)`` to distribute complete matrices. A 2D parameter without
    ``BlockShard`` must use whole-matrix compute such as ``Owned``.
    """
    return flex_optimizer_reshard(
        Muon(_normalize_param_groups(params), **kwargs),
        compute_sharding_by_fqn=compute_sharding_by_fqn,
        bucket_configs=bucket_configs,
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
        sharding
        for axis_name, sharding in compute_layout.shardings_by_mesh_axis.items()
        if type(sharding) is BlockShard
        and (applicable_axis_names is None or axis_name in applicable_axis_names)
    )
    if not block_shards:
        return None

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


def _configure_muon_reshard(
    optimizer: Muon,
    *,
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    bucket_configs: Sequence[BucketConfig],
) -> None:
    """Configure one newly constructed ``torch.optim.Muon`` optimizer.

    Every group must provide aligned ``params`` and ``param_names``. Every
    local parameter FQN must have one entry in ``compute_sharding_by_fqn``;
    extra compute-sharding entries for parameters on other pipeline stages are
    ignored. Parameter groups, compute layouts, and bucket configuration are
    frozen because optimizer state and collectives depend on them.
    """
    integration_ref = _RESHARD_INTEGRATIONS.get(optimizer)
    if integration_ref is not None and integration_ref() is not None:
        raise ValueError("flex_optimizer_reshard cannot be applied more than once")
    _validate_compute_sharding_configuration(compute_sharding_by_fqn)
    torch_muon_apis = _require_torch_muon_apis(optimizer)
    integration = _MuonReshardIntegration(optimizer, torch_muon_apis)
    integration._validate_groups()

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
    tensor_device = integration._validate_parameter_storage()
    compute_layouts = integration._build_parameter_compute_layouts(
        prepared_compute_layouts
    )
    specs = _bind_bucket_configs(
        tuple(bucket_configs),
        compute_layouts,
        get_fqn=lambda layout: layout.fqn,
        get_storage_dtensor=lambda layout: layout.param,
        get_redistribution_storage_mesh_axes=lambda layout: (
            layout.redistribution_storage_mesh_axes
        ),
        get_bucket_compatibility_key=_bucket_compatibility_key,
    )
    binding = integration._compile_optimizer_reshard_binding(
        specs,
        compute_layouts,
        tensor_device,
    )
    integration.binding = binding
    pre_hook_handle = None
    post_hook_handle = None
    try:
        pre_hook_handle = optimizer.register_load_state_dict_pre_hook(
            partial(_before_load_state_dict, integration),
            prepend=True,
        )
        post_hook_handle = optimizer.register_load_state_dict_post_hook(
            partial(_after_load_state_dict, integration),
            prepend=True,
        )
        integration.step_executor_handle = torch_muon_apis.register_step_executor(
            integration
        )
    except Exception:
        if pre_hook_handle is not None:
            pre_hook_handle.remove()
        if post_hook_handle is not None:
            post_hook_handle.remove()
        raise
    _RESHARD_INTEGRATIONS[optimizer] = ref(integration)


@dataclass
class _MuonReshardIntegration:
    """Execute one ``torch.optim.Muon`` through a compiled FlexShard binding.

    Parameter groups, FQNs, storage layouts, compute layouts, and bucket plans
    are frozen after resharding is applied. Every configured parameter must
    have a layout-compatible DTensor gradient before each rank enters
    ``step()``.

    Matrix-batch compute views use batched BF16 kernels. They implement the
    same mathematical update as ``torch.optim.Muon`` running one matrix at a
    time, but bitwise equality across the two kernel schedules is not part of
    the contract.
    """

    optimizer: Muon
    torch_muon_apis: _TorchMuonApis
    binding: _MuonReshardBinding | None = None
    first_step_validated: bool = False
    step_executor_handle: RemovableHandle | None = None

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    @property
    def state(self) -> dict[Tensor, dict[str, Any]]:
        return self.optimizer.state

    @torch.no_grad()
    def __call__(
        self,
        optimizer: Optimizer,
        closure: Callable[[], float | Tensor] | None = None,
    ) -> float | Tensor | None:
        if optimizer is not self.optimizer:
            raise RuntimeError("Muon FlexShard executor received another optimizer")
        binding = self._require_optimizer_reshard_binding()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._preflight_step()
        binding.runtime.run(
            binding.plans,
            local_tensor_spec=self._local_tensor_spec,
            prepare=self._prepare_local,
            compute=self._compute_update,
            finalize=self._apply_update,
            local_bucket_executor=binding,
        )
        return loss

    def _require_optimizer_reshard_binding(
        self,
    ) -> _MuonReshardBinding:
        binding = self.binding
        if binding is None:
            raise RuntimeError(
                "Muon FlexShard configuration is unavailable; rebuild the optimizer"
            )
        return binding

    @property
    def _specs(self) -> tuple[_BucketSpec, ...]:
        return self._require_optimizer_reshard_binding().bucket_specs

    @property
    def _bucket_plans(
        self,
    ) -> tuple[_BucketExecutionPlan[_ParameterComputeLayout], ...]:
        return self._require_optimizer_reshard_binding().plans

    def _validate_groups(self) -> None:
        if len(self.param_groups) != 1:
            raise ValueError("FlexShard Muon requires exactly one parameter group")
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
                raise ValueError(f"unsupported Muon group {group_index}")

    def _validate_parameter_storage(self) -> torch.device:
        local_devices = set()
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, DTensor):
                    raise TypeError("FlexShard Muon requires DTensor parameters")
                local_device = param.to_local().device
                local_devices.add(local_device)
        if len(local_devices) != 1 or next(iter(local_devices)).type != "cuda":
            raise ValueError("FlexShard Muon requires one CUDA device per process")
        return local_devices.pop()

    def _build_parameter_compute_layouts(
        self,
        prepared_compute_layouts: Mapping[str, _PreparedParameterComputeLayout],
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
            prepared = prepared_compute_layouts[fqn]
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
                    redistribution_storage_mesh_axes=(
                        resolved_transition.redistribution_storage_mesh_axes
                    ),
                )
            )
        return tuple(compute_layouts)

    def _build_optimizer_reshard_plan(
        self,
        specs: Sequence[_BucketSpec],
        compute_layouts: Sequence[_ParameterComputeLayout],
    ) -> _BucketPlanningResult[_ParameterComputeLayout]:
        ns_steps_by_group = tuple(group["ns_steps"] for group in self.param_groups)
        return _build_bucket_plans(
            compute_layouts,
            specs,
            get_fqn=lambda item: item.fqn,
            get_storage_dtensor=lambda item: item.param,
            requires_redistribution=lambda item: (not item.storage_is_compute_ready),
            resolve_redistribution_plans=partial(
                _resolve_muon_redistribution_plans,
                ns_steps_by_group=ns_steps_by_group,
            ),
        )

    def _compile_optimizer_reshard_binding(
        self,
        specs: tuple[_BucketSpec, ...],
        compute_layouts: Sequence[_ParameterComputeLayout],
        tensor_device: torch.device,
        parameter_group_membership_signature: (
            _ParameterGroupMembershipSignature | None
        ) = None,
    ) -> _MuonReshardBinding:
        result = self._build_optimizer_reshard_plan(specs, compute_layouts)
        self._validate_plan_across_ranks(result.plans)
        runtime = _BucketedRedistributionRuntime[_ParameterComputeLayout](tensor_device)
        binding = _MuonReshardBinding(
            integration=self,
            bucket_specs=specs,
            plans=result.plans,
            plan_items=result.plan_items,
            runtime=runtime,
            parameter_group_membership_signature=(
                self._parameter_group_membership_signature()
                if parameter_group_membership_signature is None
                else parameter_group_membership_signature
            ),
        )
        runtime.reserve_buffers(
            binding.plans,
            local_tensor_spec=self._local_tensor_spec,
            local_bucket_executor=binding,
        )
        return binding

    def _validate_plan_across_ranks(
        self,
        plans: Sequence[_BucketExecutionPlan[_ParameterComputeLayout]],
    ) -> None:
        _validate_bucket_plans_across_ranks(
            plans,
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
            compute_layout.redistribution_storage_mesh_axes,
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
        binding = self._require_optimizer_reshard_binding()
        try:
            current_membership = self._parameter_group_membership_signature()
        except ValueError as error:
            raise RuntimeError(
                "Muon parameter groups changed; rebuild the optimizer"
            ) from error
        if current_membership != binding.parameter_group_membership_signature:
            raise RuntimeError("Muon parameter groups changed; rebuild the optimizer")

        initialize_state = not self.first_step_validated
        missing_gradients = []
        changed_parameter_storage_fqn = None
        changed_gradient_storage_fqn = None
        gradients = [] if initialize_state else None
        for compute_layout in binding.plan_items:
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
                "FlexShard Muon requires every configured gradient before "
                f"step(); missing gradients: {missing_gradients}"
            )
        if changed_parameter_storage_fqn is not None:
            raise RuntimeError(
                f"parameter local storage changed for "
                f"{changed_parameter_storage_fqn!r}; "
                "rebuild the optimizer"
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
            self.first_step_validated = True

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

    def _parameter_group_membership_signature(
        self,
    ) -> _ParameterGroupMembershipSignature:
        signature = []
        for group in self.param_groups:
            params = tuple(group["params"])
            raw_param_names = group.get("param_names")
            param_names = () if raw_param_names is None else tuple(raw_param_names)
            if raw_param_names is None or len(params) != len(param_names):
                raise ValueError("params and param_names must be aligned")
            signature.append(
                tuple(
                    (id(param), fqn)
                    for param, fqn in zip(params, param_names, strict=True)
                )
            )
        return tuple(signature)

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
        self.torch_muon_apis.prepare(
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
        if compute_layout.compute_view is not None:
            compute = compute_layout.compute_view.view_as_matrix_batch(compute)
        self._compute_direction(compute_layout, compute)

    def _compute_direction(
        self, compute_layout: _ParameterComputeLayout, compute: Tensor
    ) -> None:
        group = self._group(compute_layout)
        self.torch_muon_apis.orthogonalize(
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
        self.torch_muon_apis.apply(
            local_param,
            direction,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            adjust_lr_fn=group["adjust_lr_fn"],
            logical_matrix_shape=compute_layout.global_compute_shape,
        )
        torch.autograd.graph.increment_version(compute_layout.param)

    def _plan_local_bucket(
        self,
        binding: _MuonReshardBinding,
        local_work: tuple[_ParameterComputeLayout, ...],
    ) -> dict[tuple[torch.device, torch.dtype], int]:
        execution_plan = self._build_local_execution_plan(local_work)
        binding._local_execution_plans[_local_bucket_key(local_work)] = execution_plan

        requirements: dict[tuple[torch.device, torch.dtype], int] = {}
        for local_execution in execution_plan:
            if isinstance(local_execution, _LocalMatrixBatch):
                shape = local_execution.shape
                dtype = local_execution.dtype
                device = local_execution.device
            else:
                shape, dtype, device = self._local_tensor_spec(local_execution)
            key = (device, dtype)
            requirements[key] = max(requirements.get(key, 0), math.prod(shape))
        return requirements

    def _execute_local_bucket(
        self,
        binding: _MuonReshardBinding,
        local_work: tuple[_ParameterComputeLayout, ...],
        slot: _BufferSlot,
    ) -> None:
        for local_execution in binding._local_execution_plans[
            _local_bucket_key(local_work)
        ]:
            if not isinstance(local_execution, _LocalMatrixBatch):
                self._execute_local_item(local_execution, slot)
                continue

            ns_signature = self._ns_signature(local_execution.slices[0].layout)
            if any(
                self._ns_signature(batch_slice.layout) != ns_signature
                for batch_slice in local_execution.slices[1:]
            ):
                # Optimizer group dictionaries remain mutable. Preserve
                # per-group semantics if NS settings diverge after planning.
                for batch_slice in local_execution.slices:
                    self._execute_local_item(batch_slice.layout, slot)
                continue

            prepared = slot.compute_buffer(
                local_execution.shape,
                dtype=local_execution.dtype,
                device=local_execution.device,
            )
            for batch_slice in local_execution.slices:
                matrix_batch = prepared.narrow(0, batch_slice.offset, batch_slice.size)
                self._prepare_local(
                    batch_slice.layout,
                    _physical_compute_tensor(batch_slice.layout, matrix_batch),
                )

            self._compute_direction(local_execution.slices[0].layout, prepared)
            for batch_slice in local_execution.slices:
                matrix_batch = prepared.narrow(0, batch_slice.offset, batch_slice.size)
                self._apply_update(
                    batch_slice.layout,
                    _physical_compute_tensor(batch_slice.layout, matrix_batch),
                )

    def _execute_local_item(
        self,
        layout: _ParameterComputeLayout,
        slot: _BufferSlot,
    ) -> None:
        shape, dtype, device = self._local_tensor_spec(layout)
        prepared = slot.compute_buffer(shape, dtype=dtype, device=device)
        self._prepare_local(layout, prepared)
        self._compute_update(layout, prepared)
        self._apply_update(layout, prepared)

    def _ns_signature(
        self,
        layout: _ParameterComputeLayout,
    ) -> tuple[Any, ...]:
        group = self._group(layout)
        return (
            tuple(group["ns_coefficients"]),
            group["ns_steps"],
            group["eps"],
        )

    def _build_local_execution_plan(
        self,
        layouts: tuple[_ParameterComputeLayout, ...],
    ) -> tuple[_ParameterComputeLayout | _LocalMatrixBatch, ...]:
        grouped: dict[tuple[Any, ...], list[_ParameterComputeLayout]] = {}
        for layout in layouts:
            shape, dtype, device = _local_matrix_batch_spec(layout)
            if len(shape) != 3:
                key = (layout.fqn,)
            else:
                # Communication buckets may span layers. Restrict batching to
                # sibling tensors so their combined scratch stays layer-local.
                parent_fqn, separator, _ = layout.fqn.rpartition(".")
                key = (
                    parent_fqn if separator else layout.fqn,
                    tuple(shape[1:]),
                    dtype,
                    device,
                    self._ns_signature(layout),
                )
            grouped.setdefault(key, []).append(layout)

        execution_plan: list[_ParameterComputeLayout | _LocalMatrixBatch] = []
        for compatible_layouts in grouped.values():
            batch_layouts = []
            batch_bytes = 0
            for layout in compatible_layouts:
                shape, _dtype, _device = _local_matrix_batch_spec(layout)
                tensor_bytes = math.prod(shape) * layout.param.element_size()
                if (
                    batch_layouts
                    and batch_bytes + tensor_bytes > _MAX_LOCAL_MATRIX_BATCH_BYTES
                ):
                    execution_plan.append(_make_local_matrix_execution(batch_layouts))
                    batch_layouts = []
                    batch_bytes = 0
                batch_layouts.append(layout)
                batch_bytes += tensor_bytes
            execution_plan.append(_make_local_matrix_execution(batch_layouts))
        return tuple(execution_plan)

    @staticmethod
    def _local_tensor_spec(
        compute_layout: _ParameterComputeLayout,
    ) -> tuple[torch.Size, torch.dtype, torch.device]:
        assert compute_layout.storage_is_compute_ready
        tensor = compute_layout.param.to_local().detach()
        return tensor.shape, tensor.dtype, tensor.device


# Bound persistent scratch for a combined local batch. A single layout may
# exceed this cap and continues through the existing unbatched path.
_MAX_LOCAL_MATRIX_BATCH_BYTES = 256 * 1024 * 1024


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
    sharding: Replicate | Shard | BlockShard,
    *,
    mesh_axis_size: int,
) -> tuple[tuple[int, int], ...]:
    """Return ``(start, end)`` row intervals along one mesh axis."""
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
    target_sharding_by_storage_mesh_axis: Mapping[int, Replicate | Shard | BlockShard],
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
    compute_sharding: _ComputeSharding
    storage_to_compute_transition: _StorageToComputeTransition
    resolved_compute_layout_signature: tuple[Any, ...]
    redistribution_storage_mesh_axes: tuple[int, ...]

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


_ComputeSharding = Owned | Replicate | Shard
_ComputeTensorSharding = Replicate | Shard | BlockShard


@dataclass(frozen=True, slots=True)
class _BucketCompatibilityKey:
    replicated_fanout_fqn: str | None


def _bucket_compatibility_key(
    compute_layout: _ParameterComputeLayout,
) -> _BucketCompatibilityKey:
    if (
        not compute_layout.storage_is_compute_ready
        and type(compute_layout.compute_sharding) is Replicate
    ):
        # Replicated fanout reads more exchange spans than writeback sends.
        # Isolate it so in-place writeback cannot overwrite another item's input.
        return _BucketCompatibilityKey(replicated_fanout_fqn=compute_layout.fqn)
    return _BucketCompatibilityKey(replicated_fanout_fqn=None)


@dataclass(frozen=True, slots=True)
class _UnsupportedStoragePlacement:
    type_name: str
    representation: str


@dataclass(frozen=True, slots=True)
class _ResolvedStorageToComputeTransition:
    compute_sharding: _ComputeSharding
    storage_to_compute_transition: _StorageToComputeTransition
    resolved_compute_layout_signature: tuple[Any, ...]
    redistribution_storage_mesh_axes: tuple[int, ...] = ()


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

    group_local_storage = _dtensor_storage_regions(
        compute_layout.param,
        group.participants,
        required_storage_mesh_axes=(compute_layout.redistribution_storage_mesh_axes),
    )
    storage_regions = group_local_storage.regions
    group_local_storage_shape = group_local_storage.logical_shape
    global_storage_shape = tuple(compute_layout.param.shape)
    global_compute_shape = tuple(compute_layout.global_compute_shape)
    if compute_layout.compute_view is None:
        group_local_compute_shape = group_local_storage_shape
    elif group_local_storage_shape == global_storage_shape:
        group_local_compute_shape = global_compute_shape
    else:
        raise NotImplementedError(
            f"Muon parameter {compute_layout.fqn!r} cannot yet use BlockShard "
            "matrix-batch compute while preserving a non-replicated storage "
            "mesh axis"
        )

    compute_sharding = compute_layout.compute_sharding
    if type(compute_sharding) is Owned:
        assert owner_rank is not None
        assert owner_rank in group.participants
        return _build_owned_redistribution_plan(
            storage_regions,
            participants=group.participants,
            owner_rank=owner_rank,
            logical_shape=group_local_compute_shape,
        )

    assert owner_rank is None
    if type(compute_sharding) is Replicate:
        return _build_whole_tensor_redistribution_plan(
            storage_regions,
            participants=group.participants,
            compute_participants=group.participants,
            logical_shape=group_local_compute_shape,
        )

    assert type(compute_sharding) is Shard
    if compute_layout.compute_view is None:
        return _build_dim0_shard_redistribution_plan(
            storage_regions,
            participants=group.participants,
            participant_by_shard_index=group.participant_by_shard_index,
            logical_shape=group_local_compute_shape,
        )

    return _build_matrix_batch_redistribution_plan(
        storage_regions,
        participants=group.participants,
        participant_by_shard_index=group.participant_by_shard_index,
        storage_shape=group_local_storage_shape,
        compute_shape=group_local_compute_shape,
    )


def _build_matrix_batch_redistribution_plan(
    storage_regions: Sequence[tuple[tuple[int, ...], _TensorRegion]],
    *,
    participants: tuple[int, ...],
    participant_by_shard_index: tuple[int, ...],
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

    _require_valid_plan(
        len(participant_by_shard_index) == len(participants)
        and set(participant_by_shard_index) == set(participants),
        "matrix-batch participant-by-shard-index mapping must contain every "
        "participant",
    )
    shard_index_by_participant = {
        participant: index
        for index, participant in enumerate(participant_by_shard_index)
    }
    compute_matrix_assignments = []
    compute_partitions = []
    for participant in participants:
        shard_index = shard_index_by_participant[participant]
        local_num_matrices, matrix_offset = Shard.local_shard_size_and_offset(
            num_matrices,
            len(participants),
            shard_index,
        )
        logical_region = _TensorRegion(
            offsets=(matrix_offset * matrix_rows, 0),
            shape=(local_num_matrices * matrix_rows, matrix_columns),
        )
        compute_matrix_assignments.append(
            ((participant,), matrix_offset, local_num_matrices)
        )
        compute_partitions.append(
            _ParticipantPartition(
                participant=participant,
                tensor_shape=logical_region.shape,
                logical_regions=(logical_region,),
            )
        )

    storage_to_compute_routes = []
    for source_holders, storage_region, storage_tensor_base in storage_endpoints:
        storage_row_offset = storage_region.offsets[0]
        storage_row_end = storage_row_offset + storage_region.shape[0]
        for (
            destination_participants,
            matrix_offset,
            local_num_matrices,
        ) in compute_matrix_assignments:
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
                            destination_participants,
                        ),
                    )
                )

    return _RedistributionPlan(
        participants=participants,
        logical_shape=storage_shape,
        storage_partitions=storage_partitions,
        compute_partitions=tuple(compute_partitions),
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

    normalized_target_sharding_by_storage_mesh_axis: dict[
        int, _ComputeTensorSharding
    ] = {}
    declared_shard_dims = []
    for (
        storage_mesh_axis,
        sharding,
    ) in applicable_compute_shardings_by_storage_mesh_axis.items():
        if type(sharding) is Owned:
            continue
        placement = cast(_ComputeTensorSharding, sharding)
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
        if active_owned_storage_mesh_axes:
            if changed_storage_mesh_axes:
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} cannot combine Owned "
                    f"and placement redistribution on mesh axes {axis_names}"
                )
            for storage_mesh_axis, placement in enumerate(param.placements):
                if (
                    storage_mesh_axis not in active_owned_storage_mesh_axes
                    and param.device_mesh.size(storage_mesh_axis) > 1
                ):
                    preserved_storage_sharding = _normalize_storage_placement(
                        placement,
                        ndim=param.ndim,
                        mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
                    )
                    if type(preserved_storage_sharding) is not Replicate:
                        raise NotImplementedError(
                            f"Muon parameter {fqn!r} cannot preserve non-replicated "
                            f"mesh axis {mesh_axis_names[storage_mesh_axis]!r} "
                            f"outside joint Owned axes {axis_names}"
                        )
        elif compute_view is not None:
            raise NotImplementedError(
                f"Muon parameter {fqn!r} cannot use BlockShard while "
                f"redistributing multiple mesh axes {axis_names}"
            )
        elif any(
            type(normalized_target_sharding_by_storage_mesh_axis[axis]) is not Replicate
            for axis in transport_mesh_axes
        ):
            raise NotImplementedError(
                f"Muon parameter {fqn!r} multi-axis redistribution currently "
                f"requires Replicate targets on mesh axes {axis_names}"
            )

    transport_axis_set = set(transport_mesh_axes)
    if transport_mesh_axes:
        transport_axis_names = [mesh_axis_names[axis] for axis in transport_mesh_axes]
        single_transport_axis = (
            transport_mesh_axes[0] if len(transport_mesh_axes) == 1 else None
        )
        single_transport_placement = (
            param.placements[single_transport_axis]
            if single_transport_axis is not None
            else None
        )
        for storage_mesh_axis, placement in enumerate(param.placements):
            if storage_mesh_axis in transport_axis_set:
                if type(placement) not in (Replicate, Shard):
                    raise NotImplementedError(
                        f"Muon parameter {fqn!r} cannot redistribute "
                        f"{type(placement).__name__} storage on mesh axis "
                        f"{mesh_axis_names[storage_mesh_axis]!r}"
                    )
            elif type(placement) not in (Replicate, Shard):
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} cannot preserve "
                    f"{type(placement).__name__} storage on mesh axis "
                    f"{mesh_axis_names[storage_mesh_axis]!r} while redistributing "
                    f"along mesh axes {transport_axis_names}"
                )
            elif (
                single_transport_axis is not None
                and single_transport_axis < storage_mesh_axis
                and param.device_mesh.size(storage_mesh_axis) > 1
                and type(single_transport_placement) is Shard
                and type(placement) is Shard
                and single_transport_placement.dim % param.ndim
                == placement.dim % param.ndim
            ):
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} cannot yet preserve a later mesh "
                    f"axis that repeats storage sharding on tensor dimension "
                    f"{placement.dim % param.ndim} across mesh axes "
                    f"{mesh_axis_names[single_transport_axis]!r} and "
                    f"{mesh_axis_names[storage_mesh_axis]!r}"
                )
            elif compute_view is not None:
                preserved_storage_sharding = _normalize_storage_placement(
                    placement,
                    ndim=param.ndim,
                    mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
                )
                if type(preserved_storage_sharding) is not Replicate:
                    raise NotImplementedError(
                        f"Muon parameter {fqn!r} cannot yet use BlockShard "
                        "matrix-batch compute while preserving non-replicated "
                        f"storage mesh axis {mesh_axis_names[storage_mesh_axis]!r}"
                    )

    resolved_target_signature = []
    resolved_target_by_storage_mesh_axis: dict[
        int, Owned | Replicate | Shard | BlockShard
    ] = {}
    resolved_shard_dims = []
    owned_axis_set = set(applicable_owned_storage_mesh_axes)
    for storage_mesh_axis, axis_name in enumerate(mesh_axis_names):
        if storage_mesh_axis in owned_axis_set:
            target_sharding: (
                Owned | Replicate | Shard | BlockShard | _UnsupportedStoragePlacement
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
        if isinstance(target_sharding, _UnsupportedStoragePlacement):
            raise NotImplementedError(
                f"Muon parameter {fqn!r} has unsupported compute placement "
                f"{target_sharding.type_name!r} "
                f"({target_sharding.representation}) on mesh axis {axis_name!r}"
            )
        resolved_target_signature.append((axis_name, target_sharding))
        resolved_target_by_storage_mesh_axis[storage_mesh_axis] = target_sharding
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
    if not active_owned_storage_mesh_axes and compute_shard_dims:
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

    if active_owned_storage_mesh_axes:
        compute_sharding: _ComputeSharding = Owned()
    elif transport_mesh_axes:
        transport_shardings = tuple(
            resolved_target_by_storage_mesh_axis[axis] for axis in transport_mesh_axes
        )
        if len(transport_shardings) == 1:
            transport_sharding = transport_shardings[0]
            if type(transport_sharding) is Replicate:
                compute_sharding = Replicate()
            elif type(transport_sharding) is Shard:
                compute_sharding = Shard(transport_sharding.dim)
            else:
                assert type(transport_sharding) is BlockShard
                compute_sharding = Shard(0)
        else:
            assert all(type(sharding) is Replicate for sharding in transport_shardings)
            compute_sharding = Replicate()
    elif compute_shard_dims:
        compute_sharding = Shard(0)
    elif applicable_owned_storage_mesh_axes:
        compute_sharding = Owned()
    else:
        compute_sharding = Replicate()

    if not transport_mesh_axes:
        return _ResolvedStorageToComputeTransition(
            compute_sharding=compute_sharding,
            storage_to_compute_transition=_NoRedistributionTransition(),
            resolved_compute_layout_signature=resolved_compute_layout_signature,
        )

    return _ResolvedStorageToComputeTransition(
        compute_sharding=compute_sharding,
        storage_to_compute_transition=_RedistributionTransition(),
        resolved_compute_layout_signature=resolved_compute_layout_signature,
        redistribution_storage_mesh_axes=transport_mesh_axes,
    )


def _raise_unsupported_layout(fqn: str) -> NoReturn:
    raise ValueError(f"unsupported storage-to-compute layout for {fqn!r}")


def _normalize_compute_placement(
    placement: _ComputeTensorSharding,
    *,
    ndim: int,
    mesh_axis_size: int,
) -> _ComputeTensorSharding:
    if type(placement) is Replicate:
        return Replicate()
    if type(placement) is Shard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if mesh_axis_size == 1:
            return Replicate()
        return Shard(normalized_dim)
    assert type(placement) is BlockShard
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


@dataclass(frozen=True, slots=True)
class _LocalMatrixSlice:
    layout: _ParameterComputeLayout
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class _LocalMatrixBatch:
    slices: tuple[_LocalMatrixSlice, ...]
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


@dataclass(slots=True)
class _MuonReshardBinding(_LocalBucketExecutor[_ParameterComputeLayout]):
    integration: _MuonReshardIntegration
    bucket_specs: tuple[_BucketSpec, ...]
    plans: tuple[_BucketExecutionPlan[_ParameterComputeLayout], ...]
    plan_items: tuple[_ParameterComputeLayout, ...]
    runtime: _BucketedRedistributionRuntime[_ParameterComputeLayout]
    parameter_group_membership_signature: _ParameterGroupMembershipSignature
    _local_execution_plans: dict[
        tuple[str, ...], tuple[_ParameterComputeLayout | _LocalMatrixBatch, ...]
    ] = field(default_factory=dict)

    def _plan_local_bucket(
        self,
        local_work: tuple[_ParameterComputeLayout, ...],
    ) -> dict[tuple[torch.device, torch.dtype], int]:
        return self.integration._plan_local_bucket(self, local_work)

    def _execute_local_bucket(
        self,
        local_work: tuple[_ParameterComputeLayout, ...],
        slot: _BufferSlot,
    ) -> None:
        self.integration._execute_local_bucket(self, local_work, slot)


def _make_local_matrix_execution(
    layouts: Sequence[_ParameterComputeLayout],
) -> _ParameterComputeLayout | _LocalMatrixBatch:
    if len(layouts) == 1:
        return layouts[0]

    first_shape, dtype, device = _local_matrix_batch_spec(layouts[0])
    assert len(first_shape) == 3
    offset = 0
    slices = []
    for layout in layouts:
        shape, layout_dtype, layout_device = _local_matrix_batch_spec(layout)
        assert shape[1:] == first_shape[1:]
        assert layout_dtype == dtype and layout_device == device
        size = shape[0]
        slices.append(_LocalMatrixSlice(layout, offset, size))
        offset += size
    return _LocalMatrixBatch(
        slices=tuple(slices),
        shape=torch.Size((offset, *first_shape[1:])),
        dtype=dtype,
        device=device,
    )


def _local_matrix_batch_spec(
    layout: _ParameterComputeLayout,
) -> tuple[torch.Size, torch.dtype, torch.device]:
    assert layout.storage_is_compute_ready
    tensor = layout.param.to_local().detach()
    shape = torch.Size(tensor.shape)
    if layout.compute_view is not None:
        shape = layout.compute_view.matrix_batch_shape(shape)
    return shape, tensor.dtype, tensor.device


def _physical_compute_tensor(
    layout: _ParameterComputeLayout,
    matrix_batch: Tensor,
) -> Tensor:
    if layout.compute_view is None:
        return matrix_batch
    return matrix_batch.flatten(0, 1)


def _local_bucket_key(
    layouts: tuple[_ParameterComputeLayout, ...],
) -> tuple[str, ...]:
    return tuple(layout.fqn for layout in layouts)


def _before_load_state_dict(
    integration: _MuonReshardIntegration,
    optimizer: Optimizer,
    state_dict: dict[str, Any],
) -> None:
    if optimizer is not integration.optimizer:
        raise RuntimeError("Muon FlexShard hook received another optimizer")
    binding = integration._require_optimizer_reshard_binding()
    try:
        current_membership = integration._parameter_group_membership_signature()
    except ValueError as error:
        raise ValueError(
            "current Muon parameter groups do not match configured FlexShard FQNs"
        ) from error
    if current_membership != binding.parameter_group_membership_signature:
        raise ValueError(
            "current Muon parameter groups do not match configured FlexShard FQNs"
        )

    loaded_groups = state_dict["param_groups"]
    if len(loaded_groups) != len(integration.param_groups):
        return
    for group_index, (configured_group, loaded_group) in enumerate(
        zip(
            binding.parameter_group_membership_signature,
            loaded_groups,
            strict=True,
        )
    ):
        loaded_param_names = loaded_group.get("param_names")
        if loaded_param_names is not None and tuple(loaded_param_names) != tuple(
            fqn for _param_id, fqn in configured_group
        ):
            raise ValueError(
                "loaded optimizer param_names for group "
                f"{group_index} do not match configured FlexShard FQNs"
            )


def _after_load_state_dict(
    integration: _MuonReshardIntegration,
    optimizer: Optimizer,
) -> None:
    if optimizer is not integration.optimizer:
        raise RuntimeError("Muon FlexShard hook received another optimizer")
    # Optimizer.load_state_dict restores group values such as ns_steps after
    # construction, and those values affect compute planning and buffer sizes.
    binding = integration._require_optimizer_reshard_binding()
    try:
        integration._validate_groups()
        tensor_device = integration._validate_parameter_storage()
        rebuilt_binding = integration._compile_optimizer_reshard_binding(
            binding.bucket_specs,
            binding.plan_items,
            tensor_device,
            parameter_group_membership_signature=(
                binding.parameter_group_membership_signature
            ),
        )
    except Exception:
        # Loaded group values may no longer agree with the old plans. Do not
        # leave those plans executable after a failed rebuild.
        integration.binding = None
        raise
    integration.binding = rebuilt_binding
    # init_optim_state may have validated placeholder state before the load.
    integration.first_step_validated = False


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
