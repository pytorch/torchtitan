# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone bucketed Distributed Muon optimizer."""

from __future__ import annotations

import fnmatch
import hashlib
import heapq
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.tensor.placement_types as placement_types
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement
from torch.optim import Optimizer


__all__ = ["BucketSpec", "assign_balanced_owners", "DistributedMuon"]

_DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


@dataclass(frozen=True, slots=True)
class BucketSpec:
    """One ordered optimizer-work bucket selected by canonical FQN.

    Patterns use case-sensitive ``fnmatch`` syntax. Every optimizer FQN must
    match exactly one bucket, and sequence order controls execution order.
    ``owner_rank_by_fqn`` must exactly cover the bucket's whole-matrix-owned
    parameters using process-group-local ranks. Local and matrix-batch-sharded
    parameters have no owner entry. ``name`` is diagnostic metadata only.
    """

    patterns: tuple[str, ...]
    owner_rank_by_fqn: Mapping[str, int]
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", tuple(self.patterns))
        object.__setattr__(self, "owner_rank_by_fqn", dict(self.owner_rank_by_fqn))


def assign_balanced_owners(
    bucket_fqns: Sequence[Sequence[str]],
    memory_estimate_by_fqn: Mapping[str, int],
    *,
    num_ranks: int,
    initial_memory_by_rank: Sequence[int] | None = None,
) -> tuple[dict[str, int], ...]:
    """Greedily balance selected parameters across group-local ranks."""
    initial_memory_by_rank = initial_memory_by_rank or (0,) * num_ranks
    rank_loads = list(zip(initial_memory_by_rank, range(num_ranks), strict=True))
    heapq.heapify(rank_loads)
    owners_by_bucket = []
    for bucket in bucket_fqns:
        bucket_owners = {}
        candidates = (fqn for fqn in bucket if fqn in memory_estimate_by_fqn)
        for fqn in sorted(
            candidates, key=lambda name: (-memory_estimate_by_fqn[name], name)
        ):
            load, rank = heapq.heappop(rank_loads)
            bucket_owners[fqn] = rank
            heapq.heappush(
                rank_loads, (load + memory_estimate_by_fqn[fqn], rank)
            )
        owners_by_bucket.append(bucket_owners)
    return tuple(owners_by_bucket)


class DistributedMuon(Optimizer):
    """CUDA Muon with bucketed storage-to-matrix-compute routing."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        bucket_spec: Sequence[BucketSpec],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = _DEFAULT_NS_COEFFICIENTS,
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
        self._communication_context: _CommunicationContext | None = None
        self._first_step_validated = False
        super().__init__(params, defaults)
        assert all(
            isinstance(param, DTensor) and param.device.type == "cuda"
            for group in self.param_groups
            for param in group["params"]
        ), "DistributedMuon requires CUDA DTensor parameters"

        self._control_group = self._infer_control_group()
        self._control_group_ranks = tuple(
            dist.get_process_group_ranks(self._control_group)
            if self._control_group is not None
            else range(dist.get_world_size())
        )

        self._specs = tuple(bucket_spec)

        setup_error: Exception | None = None
        try:
            self._validate_groups()
            self._initialize_plan()
        except Exception as error:
            setup_error = error
        self._synchronize_setup_error(setup_error)
        self._validate_plan_across_ranks()
        self._frozen_group_metadata = self._group_metadata()

    @torch.no_grad()
    def step(
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._preflight_step()
        self._pipelined_step()
        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if hasattr(self, "_plans"):
            raise RuntimeError(
                "DistributedMuon parameter groups are frozen"
            )
        super().add_param_group(param_group)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_groups = state_dict.get("param_groups", ())
        if len(saved_groups) != len(self._frozen_group_metadata) or any(
            ("param_names" in saved and tuple(saved["param_names"]) != names)
            or saved.get("matrix_shape") != matrix_shape
            or _effective_matrix_block_dim(saved) != matrix_block_dim
            for saved, (names, matrix_shape, matrix_block_dim) in zip(
                saved_groups, self._frozen_group_metadata, strict=True
            )
        ):
            raise ValueError("checkpoint changed DistributedMuon's static plan")
        super().load_state_dict(state_dict)
        self._validate_plan_across_ranks()
        self._first_step_validated = False

    def _infer_control_group(self) -> dist.ProcessGroup | None:
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, DTensor) and param.device_mesh.ndim == 1:
                    return param.device_mesh.get_group()
        return None

    def _validate_groups(self) -> None:
        for group_index, group in enumerate(self.param_groups):
            if group.get("fused") or group.get("foreach"):
                raise NotImplementedError(
                    "DistributedMuon does not support fused or foreach"
                )
            ns_steps = group["ns_steps"]
            coefficients = group["ns_coefficients"]
            matrix_block_dim = group.get("matrix_block_dim")
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
                or matrix_block_dim not in (None, 0, 1)
                or (
                    matrix_block_dim is not None
                    and group.get("matrix_shape") is None
                )
            ):
                raise ValueError(f"invalid DistributedMuon group {group_index}")

    def _unassigned_bindings(self) -> tuple[_ParamPlan, ...]:
        bindings = []
        seen_names = set()
        seen_params = set()
        for group_index, group in enumerate(self.param_groups):
            params = group["params"]
            names = group.get("param_names")
            if names is None or len(names) != len(params):
                raise ValueError(
                    "DistributedMuon requires param_names aligned with params"
                )
            matrix_shape = group.get("matrix_shape")
            matrix_block_dim = _effective_matrix_block_dim(group)
            for fqn, param in zip(names, params, strict=True):
                if fqn in seen_names or id(param) in seen_params:
                    raise ValueError(f"duplicate Muon parameter {fqn!r}")
                seen_names.add(fqn)
                seen_params.add(id(param))
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
                _validate_matrix_shape(param, matrix_shape)
                local_blocks = _local_block_layout(
                    param, matrix_shape, matrix_block_dim
                )
                sharded_blocks = _matrix_batch_shard_layout(
                    param, matrix_shape, matrix_block_dim
                )
                if not local_blocks and not sharded_blocks:
                    if (
                        param.ndim != 2
                        or param.device_mesh.ndim != 1
                        or len(param.placements) != 1
                        or type(param.placements[0]) is not Shard
                        or param.placements[0].dim % param.ndim != 0
                    ):
                        raise ValueError(
                            f"Muon parameter {fqn!r} is neither complete local "
                            "matrix blocks nor 1D Shard(0)"
                        )
                if not local_blocks:
                    world_size = param.device_mesh.size()
                    if sharded_blocks:
                        assert matrix_shape is not None
                        matrix_count = param.shape[1] // matrix_shape[1]
                        if matrix_count < world_size:
                            raise ValueError(
                                f"Muon parameter {fqn!r} has fewer matrix blocks "
                                "than compute ranks"
                            )
                bindings.append(
                    _ParamPlan(
                        fqn=fqn,
                        param=param,
                        group_index=group_index,
                        matrix_shape=matrix_shape,
                        local_blocks=local_blocks,
                        sharded_blocks=sharded_blocks,
                        global_shape=torch.Size(param.shape),
                        global_stride=tuple(param.stride()),
                        local_shape=torch.Size(local.shape),
                        local_stride=tuple(local.stride()),
                        mesh_ranks=_storage_mesh_ranks(param.device_mesh),
                        placements=tuple(param.placements),
                    )
                )
        return tuple(bindings)

    def _resolve_buckets(
        self, bindings: tuple[_ParamPlan, ...]
    ) -> list[list[_ParamPlan]]:
        resolved = [[] for _ in self._specs]
        for binding in bindings:
            matches = [
                index
                for index, spec in enumerate(self._specs)
                if any(
                    fnmatch.fnmatchcase(binding.fqn, pattern)
                    for pattern in spec.patterns
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Muon parameter {binding.fqn!r} must match one bucket"
                )
            resolved[matches[0]].append(binding)
        return resolved

    def _initialize_plan(self) -> None:
        bindings = self._unassigned_bindings()
        resolved = self._resolve_buckets(bindings)
        plans = []
        planned_bindings = []
        expected_distributed_ranks: tuple[int, ...] | None = None
        for spec, bucket in zip(self._specs, resolved, strict=True):
            if not bucket:
                continue
            local_bindings = tuple(
                sorted(
                    (binding for binding in bucket if binding.local_blocks),
                    key=lambda item: item.fqn,
                )
            )
            distributed = tuple(
                sorted(
                    (binding for binding in bucket if not binding.local_blocks),
                    key=lambda item: item.fqn,
                )
            )
            expected_owners = {
                binding.fqn
                for binding in distributed
                if not binding.sharded_blocks
            }
            provided_owners = set(spec.owner_rank_by_fqn)
            if provided_owners != expected_owners:
                raise ValueError(
                    f"bucket {spec.name!r} owner assignment must exactly cover "
                    "whole-matrix-owned parameters; "
                    f"missing={sorted(expected_owners - provided_owners)}, "
                    f"extra={sorted(provided_owners - expected_owners)}"
                )
            distributed_bindings = tuple(
                replace(
                    binding,
                    owner_rank=spec.owner_rank_by_fqn.get(binding.fqn, -1),
                )
                for binding in distributed
            )
            planned_bindings.extend(local_bindings)
            planned_bindings.extend(distributed_bindings)

            if not distributed_bindings:
                local_tensor = local_bindings[0].param.to_local()
                plans.append(
                    _BucketPlan(
                        local_bindings=local_bindings,
                        distributed_bindings=(),
                        process_group=None,
                        group_rank=-1,
                        world_size=0,
                        input_split_sizes=[],
                        output_split_sizes=[],
                        send_segments_by_binding=(),
                        receive_segments_by_binding=(),
                        dtype=local_tensor.dtype,
                        device=local_tensor.device,
                        local_buffer_numel=0,
                        routed_buffer_numel=0,
                    )
                )
                continue

            ranks = distributed_bindings[0].mesh_ranks
            if any(
                binding.mesh_ranks != ranks for binding in distributed_bindings
            ) or (
                expected_distributed_ranks is not None
                and ranks != expected_distributed_ranks
            ):
                raise ValueError(
                    "redistributed Muon parameters must use one process group"
                )
            expected_distributed_ranks = ranks
            mesh = distributed_bindings[0].param.device_mesh
            process_group = mesh.get_group()
            group_rank = mesh.get_local_rank()
            world_size = mesh.size()
            owner_ranks = [
                binding.owner_rank
                for binding in distributed_bindings
                if not binding.sharded_blocks
            ]
            if any(rank not in range(world_size) for rank in owner_ranks):
                raise ValueError(
                    f"bucket {spec.name!r} has owner outside its process group"
                )
            local_tensors = [binding.param.to_local() for binding in distributed_bindings]
            dtype = local_tensors[0].dtype
            device = local_tensors[0].device
            if any(
                tensor.dtype != dtype or tensor.device != device
                for tensor in local_tensors
            ):
                raise ValueError(f"bucket {spec.name!r} mixes dtype or device")
            (
                input_splits,
                output_splits,
                send_segments,
                receive_segments,
            ) = _routing_metadata(
                distributed_bindings, group_rank, world_size
            )
            plans.append(
                _BucketPlan(
                    local_bindings=local_bindings,
                    distributed_bindings=distributed_bindings,
                    process_group=process_group,
                    group_rank=group_rank,
                    world_size=world_size,
                    input_split_sizes=input_splits,
                    output_split_sizes=output_splits,
                    send_segments_by_binding=_segments_by_binding(
                        send_segments, len(distributed_bindings)
                    ),
                    receive_segments_by_binding=_segments_by_binding(
                        receive_segments, len(distributed_bindings)
                    ),
                    dtype=dtype,
                    device=device,
                    local_buffer_numel=sum(input_splits),
                    routed_buffer_numel=sum(output_splits),
                )
            )

        self._plans = tuple(plans)
        self._bindings = tuple(planned_bindings)
        self._tensor_device = self._plans[0].device
        if (
            expected_distributed_ranks is not None
            and expected_distributed_ranks != self._control_group_ranks
        ):
            raise ValueError(
                "redistributed Muon parameters must use the optimizer control group"
            )

    def _synchronize_setup_error(self, error: Exception | None) -> None:
        first_param = cast(DTensor, self.param_groups[0]["params"][0])
        device = first_param.to_local().device
        status = torch.tensor(int(error is not None), dtype=torch.int32, device=device)
        dist.all_reduce(status, group=self._control_group)
        if status.item():
            if error is not None:
                raise error
            raise RuntimeError("DistributedMuon setup failed on another rank")

    def _validate_plan_across_ranks(self) -> None:
        description = [
            (
                str(plan.dtype),
                plan.device.type,
                plan.world_size,
                tuple(
                    _routing_metadata(
                        plan.distributed_bindings, rank, plan.world_size
                    )
                    for rank in range(plan.world_size)
                ),
                [
                    (
                        binding.fqn,
                        binding.group_index,
                        tuple(binding.global_shape),
                        binding.global_stride,
                        tuple(binding.local_shape),
                        binding.local_stride,
                        str(binding.param.dtype),
                        binding.param.to_local().device.type,
                        binding.matrix_shape,
                        binding.local_blocks,
                        binding.sharded_blocks,
                        binding.owner_rank,
                        binding.mesh_ranks,
                        tuple(map(str, binding.placements)),
                        self._group_signature(binding),
                    )
                    for binding in plan.local_bindings + plan.distributed_bindings
                ],
            )
            for plan in self._plans
        ]
        digest = hashlib.sha256(repr(description).encode()).digest()
        plan_hash = int.from_bytes(digest[:7], "little")
        local_hash = torch.tensor(plan_hash, dtype=torch.int64, device=self._tensor_device)
        gathered = [
            torch.empty_like(local_hash)
            for _ in range(dist.get_world_size(self._control_group))
        ]
        dist.all_gather(gathered, local_hash, group=self._control_group)
        if any(value.item() != plan_hash for value in gathered):
            raise RuntimeError("DistributedMuon plans differ across ranks")

    def _group(self, binding: _ParamPlan) -> dict[str, Any]:
        return self.param_groups[binding.group_index]

    def _group_signature(self, binding: _ParamPlan) -> tuple[Any, ...]:
        group = self._group(binding)
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

    def _group_metadata(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                tuple(group.get("param_names", ())),
                group.get("matrix_shape"),
                _effective_matrix_block_dim(group),
            )
            for group in self.param_groups
        )

    def _preflight_step(self) -> None:
        initialize_state = not self._first_step_validated
        if initialize_state:
            missing = sum(binding.param.grad is None for binding in self._bindings)
            status = torch.tensor(
                missing, dtype=torch.int32, device=self._tensor_device
            )
            dist.all_reduce(status, group=self._control_group)
            if status.item():
                raise RuntimeError(
                    "DistributedMuon requires every configured gradient"
                )

        gradients = []
        for binding in self._bindings:
            grad = self._gradient(binding)
            gradients.append((binding, grad))
            if initialize_state:
                self._validate_momentum(binding)

        # State creation happens only after every gradient and existing state
        # tensor has passed validation, so a deterministic input error cannot
        # partially update an earlier bucket.
        if initialize_state:
            for binding, grad in gradients:
                self._momentum(binding, grad)
            self._first_step_validated = True

    @staticmethod
    def _has_storage_layout(tensor: DTensor, binding: _ParamPlan) -> bool:
        local = tensor.to_local()
        param_local = binding.param.to_local()
        return (
            torch.Size(tensor.shape) == binding.global_shape
            and tuple(tensor.stride()) == binding.global_stride
            and _storage_mesh_ranks(tensor.device_mesh) == binding.mesh_ranks
            and tuple(tensor.placements) == binding.placements
            and local.shape == binding.local_shape
            and tuple(local.stride()) == binding.local_stride
            and local.dtype == param_local.dtype
            and local.device == param_local.device
            and local.is_contiguous()
        )

    def _gradient(self, binding: _ParamPlan) -> DTensor:
        grad = binding.param.grad
        if not isinstance(grad, DTensor) or not self._has_storage_layout(
            grad, binding
        ):
            raise RuntimeError(f"gradient layout changed for {binding.fqn!r}")
        return grad

    def _validate_momentum(self, binding: _ParamPlan) -> None:
        momentum = self.state.get(binding.param, {}).get("momentum_buffer")
        if momentum is None:
            return
        if not isinstance(momentum, DTensor) or not self._has_storage_layout(
            momentum, binding
        ):
            raise RuntimeError(f"momentum layout changed for {binding.fqn!r}")

    def _momentum(self, binding: _ParamPlan, grad: DTensor) -> DTensor:
        state = self.state[binding.param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(
                grad, memory_format=torch.preserve_format
            )
        return state["momentum_buffer"]

    def _update_local_momentum(
        self, binding: _ParamPlan
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        grad = cast(DTensor, binding.param.grad)
        momentum = cast(DTensor, self.state[binding.param]["momentum_buffer"])
        local_grad = grad.to_local()
        local_momentum = momentum.to_local()
        group = self._group(binding)
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

    def _prepare_local(self, binding: _ParamPlan, out: Tensor) -> None:
        grad, momentum, group = self._update_local_momentum(binding)
        self._write_prepared(group, grad, momentum, out)

    def _compute_update(self, binding: _ParamPlan, prepared: Tensor) -> Tensor:
        group = self._group(binding)
        logical_prepared = _matrix_view(prepared, binding.matrix_shape)
        update = _compute_muon_update(
            logical_prepared,
            lr=group["lr"],
            ns_coefficients=group["ns_coefficients"],
            ns_steps=group["ns_steps"],
            eps=group["eps"],
            adjust_lr_fn=group["adjust_lr_fn"],
            out=logical_prepared,
        )
        return update.view(prepared.shape)

    def _apply_update(self, binding: _ParamPlan, update: Tensor) -> None:
        group = self._group(binding)
        local_param = binding.param.to_local()
        local_param.mul_(1 - group["lr"] * group["weight_decay"])
        local_param.add_(update)
        torch.autograd.graph.increment_version(binding.param)

    def _compute_local_bindings(
        self, plan: _BucketPlan, slot: _BufferSlot
    ) -> None:
        for binding in plan.local_bindings:
            local_param = binding.param.to_local()
            prepared = slot.compute_buffer(
                local_param.shape,
                dtype=local_param.dtype,
                device=local_param.device,
            )
            self._prepare_local(binding, prepared)
            self._apply_update(binding, self._compute_update(binding, prepared))

    def _prepare_distributed(
        self, plan: _BucketPlan, local_buffer: Tensor
    ) -> None:
        for index, binding in enumerate(plan.distributed_bindings):
            grad, momentum, group = self._update_local_momentum(binding)
            for segment in plan.send_segments_by_binding[index]:
                out = local_buffer[
                    segment.buffer_offset : segment.buffer_offset + segment.numel
                ]
                if binding.sharded_blocks:
                    assert binding.matrix_shape is not None
                    matrix_columns = binding.matrix_shape[1]
                    matrix_count = binding.global_shape[1] // matrix_columns
                    grad_blocks = grad.view(
                        segment.storage_row_count,
                        matrix_count,
                        matrix_columns,
                    )
                    momentum_blocks = momentum.view_as(grad_blocks)
                    block_slice = slice(
                        segment.matrix_block_offset,
                        segment.matrix_block_offset + segment.matrix_block_count,
                    )
                    grad_piece = grad_blocks[:, block_slice, :].movedim(1, 0)
                    momentum_piece = momentum_blocks[:, block_slice, :].movedim(
                        1, 0
                    )
                    out = out.view(
                        segment.matrix_block_count,
                        segment.storage_row_count,
                        matrix_columns,
                    )
                    self._write_prepared(
                        group, grad_piece, momentum_piece, out
                    )
                else:
                    self._write_prepared(
                        group, grad, momentum, out.view(grad.shape)
                    )

    @staticmethod
    def _forward(work: _BucketWork) -> None:
        plan = work.plan
        assert plan.process_group is not None
        dist.all_to_all_single(
            work.routed_buffer,
            work.local_buffer,
            output_split_sizes=plan.output_split_sizes,
            input_split_sizes=plan.input_split_sizes,
            group=plan.process_group,
        )

    def _compute_redistributed(
        self, work: _BucketWork, slot: _BufferSlot
    ) -> None:
        plan = work.plan
        for index, binding in enumerate(plan.distributed_bindings):
            compute_shape = _redistributed_compute_shape(
                binding, plan.group_rank, plan.world_size
            )
            if compute_shape is None:
                continue
            compute = slot.compute_buffer(
                compute_shape, dtype=plan.dtype, device=plan.device
            )
            segments = plan.receive_segments_by_binding[index]
            for segment in segments:
                received = work.routed_buffer[
                    segment.buffer_offset : segment.buffer_offset + segment.numel
                ]
                if binding.sharded_blocks:
                    assert binding.matrix_shape is not None
                    matrix_columns = binding.matrix_shape[1]
                    compute[
                        :,
                        segment.storage_row_offset : (
                            segment.storage_row_offset
                            + segment.storage_row_count
                        ),
                        :,
                    ].copy_(
                        received.view(
                            segment.matrix_block_count,
                            segment.storage_row_count,
                            matrix_columns,
                        )
                    )
                else:
                    compute[
                        segment.storage_row_offset : (
                            segment.storage_row_offset
                            + segment.storage_row_count
                        )
                    ].copy_(received.view(segment.storage_row_count, -1))

            self._compute_update(binding, compute)

            for segment in segments:
                routed = work.routed_buffer[
                    segment.buffer_offset : segment.buffer_offset + segment.numel
                ]
                if binding.sharded_blocks:
                    assert binding.matrix_shape is not None
                    routed.view(
                        segment.matrix_block_count,
                        segment.storage_row_count,
                        binding.matrix_shape[1],
                    ).copy_(
                        compute[
                            :,
                            segment.storage_row_offset : (
                                segment.storage_row_offset
                                + segment.storage_row_count
                            ),
                            :,
                        ]
                    )
                else:
                    routed.view(
                        segment.storage_row_count, *binding.global_shape[1:]
                    ).copy_(
                        compute[
                            segment.storage_row_offset : (
                                segment.storage_row_offset
                                + segment.storage_row_count
                            )
                        ]
                    )

    @staticmethod
    def _reverse(work: _BucketWork) -> None:
        plan = work.plan
        assert plan.process_group is not None
        dist.all_to_all_single(
            work.local_buffer,
            work.routed_buffer,
            output_split_sizes=plan.input_split_sizes,
            input_split_sizes=plan.output_split_sizes,
            group=plan.process_group,
        )

    def _finalize_distributed(self, work: _BucketWork) -> None:
        plan = work.plan
        for index, binding in enumerate(plan.distributed_bindings):
            local_param = binding.param.to_local()
            segments = plan.send_segments_by_binding[index]
            if not binding.sharded_blocks:
                assert len(segments) == 1
                segment = segments[0]
                update = work.local_buffer[
                    segment.buffer_offset : segment.buffer_offset + segment.numel
                ].view(local_param.shape)
                self._apply_update(binding, update)
                continue

            assert binding.matrix_shape is not None
            group = self._group(binding)
            matrix_columns = binding.matrix_shape[1]
            matrix_count = binding.global_shape[1] // matrix_columns
            local_blocks = local_param.view(
                local_param.shape[0], matrix_count, matrix_columns
            )
            local_blocks.mul_(1 - group["lr"] * group["weight_decay"])
            for segment in segments:
                update = work.local_buffer[
                    segment.buffer_offset : segment.buffer_offset + segment.numel
                ].view(
                    segment.matrix_block_count,
                    segment.storage_row_count,
                    matrix_columns,
                )
                block_slice = slice(
                    segment.matrix_block_offset,
                    segment.matrix_block_offset + segment.matrix_block_count,
                )
                local_blocks[:, block_slice, :].add_(update.movedim(0, 1))
            torch.autograd.graph.increment_version(binding.param)

    def _begin_pipelined(
        self,
        plan: _BucketPlan,
        slot: _BufferSlot,
        caller_stream: torch.Stream,
        context: _CommunicationContext,
    ) -> _BucketWork:
        handle = context.device_handle
        transfer = context.transfer_stream
        with handle.stream(transfer):
            local_buffer, routed_buffer = slot.communication_buffers(plan)
            work = _BucketWork(plan, local_buffer, routed_buffer)
            self._prepare_distributed(plan, local_buffer)
            self._forward(work)
            work.forward_ready = handle.Event()
            work.forward_ready.record(transfer)

        with handle.stream(caller_stream):
            self._compute_local_bindings(plan, slot)
            caller_stream.wait_event(work.forward_ready)
            self._compute_redistributed(work, slot)
            work.compute_done = handle.Event()
            work.compute_done.record(caller_stream)
        return work

    def _complete_pipelined(
        self, work: _BucketWork, context: _CommunicationContext
    ) -> None:
        assert work.compute_done is not None
        handle = context.device_handle
        transfer = context.transfer_stream
        with handle.stream(transfer):
            transfer.wait_event(work.compute_done)
            self._reverse(work)
            self._finalize_distributed(work)
            work.done = handle.Event()
            work.done.record(transfer)

    @staticmethod
    def _release_pipelined(work: _BucketWork, caller_stream: torch.Stream) -> None:
        assert work.done is not None
        caller_stream.wait_event(work.done)

    def _pipelined_step(self) -> None:
        if self._communication_context is None:
            self._communication_context = _CommunicationContext.create(
                self._tensor_device
            )
        context = self._communication_context
        handle = context.device_handle
        caller = handle.current_stream(self._tensor_device)
        context.transfer_stream.wait_stream(caller)

        pending: list[_BucketWork] = []
        distributed_index = 0
        try:
            for plan in self._plans:
                slot = context.slots[distributed_index % 2]
                if not plan.distributed_bindings:
                    with handle.stream(caller):
                        self._compute_local_bindings(plan, slot)
                    continue
                work = self._begin_pipelined(plan, slot, caller, context)
                distributed_index += 1
                pending.append(work)
                if len(pending) == 2:
                    oldest = pending.pop(0)
                    self._complete_pipelined(oldest, context)
                    self._release_pipelined(oldest, caller)
            for work in pending:
                self._complete_pipelined(work, context)
                self._release_pipelined(work, caller)
        except Exception:
            # Preserve allocator lifetime ordering for work already enqueued on
            # either stream. This is an error-path drain, not synchronization.
            context.transfer_stream.wait_stream(caller)
            caller.wait_stream(context.transfer_stream)
            raise

@dataclass(frozen=True, slots=True)
class _ParamPlan:
    fqn: str
    param: DTensor
    group_index: int
    matrix_shape: tuple[int, int] | None
    local_blocks: bool
    sharded_blocks: bool
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    local_shape: torch.Size
    local_stride: tuple[int, ...]
    mesh_ranks: tuple[int, ...]
    placements: tuple[Placement, ...]
    owner_rank: int = -1


@dataclass(frozen=True, slots=True)
class _RouteSegment:
    binding_index: int
    buffer_offset: int
    numel: int
    storage_row_offset: int
    storage_row_count: int
    matrix_block_offset: int = 0
    matrix_block_count: int = 0


@dataclass(slots=True)
class _BucketPlan:
    local_bindings: tuple[_ParamPlan, ...]
    distributed_bindings: tuple[_ParamPlan, ...]
    process_group: dist.ProcessGroup | None
    group_rank: int
    world_size: int
    input_split_sizes: list[int]
    output_split_sizes: list[int]
    send_segments_by_binding: tuple[tuple[_RouteSegment, ...], ...]
    receive_segments_by_binding: tuple[tuple[_RouteSegment, ...], ...]
    dtype: torch.dtype
    device: torch.device
    local_buffer_numel: int
    routed_buffer_numel: int


@dataclass(slots=True)
class _BucketWork:
    plan: _BucketPlan
    local_buffer: Tensor
    routed_buffer: Tensor
    forward_ready: torch.Event | None = None
    compute_done: torch.Event | None = None
    done: torch.Event | None = None


@dataclass(slots=True)
class _BufferSlot:
    local_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
        default_factory=dict
    )
    routed_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
        default_factory=dict
    )
    compute_storage: dict[tuple[torch.device, torch.dtype], Tensor] = field(
        default_factory=dict
    )

    @staticmethod
    def _ensure_capacity(
        storage: dict[tuple[torch.device, torch.dtype], Tensor],
        *,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        key = (device, dtype)
        buffer = storage.get(key)
        if buffer is None or buffer.numel() < numel:
            buffer = torch.empty(numel, dtype=dtype, device=device)
            storage[key] = buffer
        return buffer[:numel]

    def communication_buffers(self, plan: _BucketPlan) -> tuple[Tensor, Tensor]:
        return (
            self._ensure_capacity(
                self.local_storage,
                numel=plan.local_buffer_numel,
                dtype=plan.dtype,
                device=plan.device,
            ),
            self._ensure_capacity(
                self.routed_storage,
                numel=plan.routed_buffer_numel,
                dtype=plan.dtype,
                device=plan.device,
            ),
        )

    def compute_buffer(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        return self._ensure_capacity(
            self.compute_storage,
            numel=math.prod(shape),
            dtype=dtype,
            device=device,
        ).view(shape)


@dataclass(slots=True)
class _CommunicationContext:
    device_handle: ModuleType
    transfer_stream: torch.Stream
    slots: tuple[_BufferSlot, _BufferSlot]

    @classmethod
    def create(cls, device: torch.device) -> _CommunicationContext:
        device_handle = torch.get_device_module(device)
        transfer_stream = device_handle.Stream(device=device, priority=0)
        return cls(
            device_handle=device_handle,
            transfer_stream=transfer_stream,
            slots=(_BufferSlot(), _BufferSlot()),
        )


def _is_shard_like(placement: Placement) -> bool:
    predicate = getattr(placement_types, "_is_shard_like", None)
    if predicate is not None:
        return predicate(placement)
    strided_shard_type = getattr(placement_types, "_StridedShard", None)
    return isinstance(placement, Shard) or (
        strided_shard_type is not None and isinstance(placement, strided_shard_type)
    )


def _storage_mesh_ranks(mesh: DeviceMesh) -> tuple[int, ...]:
    if mesh.ndim == 1:
        return tuple(dist.get_process_group_ranks(mesh.get_group()))
    return tuple(mesh.mesh.flatten().tolist())


def _effective_matrix_block_dim(group: Mapping[str, Any]) -> int | None:
    if group.get("matrix_shape") is None:
        return None
    matrix_block_dim = group.get("matrix_block_dim")
    return 0 if matrix_block_dim is None else matrix_block_dim


def _validate_matrix_shape(
    tensor: Tensor, matrix_shape: tuple[int, int] | None
) -> None:
    if matrix_shape is None:
        return
    if (
        not isinstance(matrix_shape, tuple)
        or len(matrix_shape) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in matrix_shape)
        or tensor.numel() % math.prod(matrix_shape)
    ):
        raise ValueError(
            f"invalid matrix_shape {matrix_shape!r} for {tuple(tensor.shape)}"
        )


def _matrix_view(tensor: Tensor, matrix_shape: tuple[int, int] | None) -> Tensor:
    if matrix_shape is None:
        return tensor
    return tensor.view(-1, *matrix_shape)


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
    matrix_shape: torch.Size,
) -> float:
    rows, columns = matrix_shape[-2:]
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
) -> Tensor:
    direction = _zeropower_via_newtonschulz(
        prepared,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )
    adjusted_lr = _adjust_learning_rate(lr, adjust_lr_fn, prepared.shape)
    out.zero_()
    out.add_(direction, alpha=-adjusted_lr)
    return out


def _local_block_layout(
    param: DTensor,
    matrix_shape: tuple[int, int] | None,
    matrix_block_dim: int | None,
) -> bool:
    if matrix_block_dim == 1:
        return False
    local = param.to_local()
    shard_placements = []
    for placement in param.placements:
        if _is_shard_like(placement):
            shard_dim = getattr(placement, "dim") % param.ndim
            if shard_dim != 0:
                return False
            shard_placements.append(placement)
        elif not isinstance(placement, Replicate):
            return False
    if not shard_placements:
        return False

    if matrix_shape is not None:
        rows, columns = matrix_shape
        if param.shape[-1] != columns or local.shape[-1] != columns:
            return False
        if local.numel() % (rows * columns):
            return False
        if param.device_mesh.ndim == 1 and len(param.placements) == 1:
            rank = param.device_mesh.get_local_rank()
            local_rows, row_offset = Shard.local_shard_size_and_offset(
                param.shape[0], param.device_mesh.size(), rank
            )
            if row_offset % rows or local_rows % rows:
                return False
        return True

    return (
        param.ndim > 2
        and tuple(local.shape[-2:]) == tuple(param.shape[-2:])
    )


def _matrix_batch_shard_layout(
    param: DTensor,
    matrix_shape: tuple[int, int] | None,
    matrix_block_dim: int | None,
) -> bool:
    if matrix_block_dim != 1:
        return False
    matrix_rows, matrix_columns = cast(tuple[int, int], matrix_shape)
    if (
        param.ndim != 2
        or param.device_mesh.ndim != 1
        or len(param.placements) != 1
        or type(param.placements[0]) is not Shard
        or param.placements[0].dim % param.ndim != 0
        or param.shape[0] != matrix_rows
        or param.shape[1] % matrix_columns
    ):
        raise ValueError(
            "matrix_block_dim=1 requires a rank-2 1D Shard(0) matrix batch"
        )
    return True


def _redistributed_compute_shape(
    binding: _ParamPlan, group_rank: int, world_size: int
) -> torch.Size | None:
    if not binding.sharded_blocks:
        return binding.global_shape if binding.owner_rank == group_rank else None
    assert binding.matrix_shape is not None
    matrix_count = binding.global_shape[1] // binding.matrix_shape[1]
    local_count, _ = Shard.local_shard_size_and_offset(
        matrix_count, world_size, group_rank
    )
    return torch.Size((local_count, *binding.matrix_shape))


def _segments_by_binding(
    segments: tuple[_RouteSegment, ...], binding_count: int
) -> tuple[tuple[_RouteSegment, ...], ...]:
    return tuple(
        tuple(
            segment
            for segment in segments
            if segment.binding_index == binding_index
        )
        for binding_index in range(binding_count)
    )


def _route_segment(
    binding: _ParamPlan,
    binding_index: int,
    source_rank: int,
    destination_rank: int,
    world_size: int,
    buffer_offset: int,
) -> _RouteSegment | None:
    storage_row_count, storage_row_offset = Shard.local_shard_size_and_offset(
        binding.global_shape[0], world_size, source_rank
    )
    matrix_block_count = 0
    matrix_block_offset = 0
    if binding.sharded_blocks:
        assert binding.matrix_shape is not None
        matrix_count = binding.global_shape[1] // binding.matrix_shape[1]
        matrix_block_count, matrix_block_offset = (
            Shard.local_shard_size_and_offset(
                matrix_count, world_size, destination_rank
            )
        )
        numel = (
            storage_row_count * matrix_block_count * binding.matrix_shape[1]
        )
    elif binding.owner_rank == destination_rank:
        numel = storage_row_count * math.prod(binding.global_shape[1:])
    else:
        return None
    return _RouteSegment(
        binding_index=binding_index,
        buffer_offset=buffer_offset,
        numel=numel,
        storage_row_offset=storage_row_offset,
        storage_row_count=storage_row_count,
        matrix_block_offset=matrix_block_offset,
        matrix_block_count=matrix_block_count,
    )


def _routing_metadata(
    bindings: tuple[_ParamPlan, ...], group_rank: int, world_size: int
) -> tuple[
    list[int],
    list[int],
    tuple[_RouteSegment, ...],
    tuple[_RouteSegment, ...],
]:
    input_split_sizes = []
    send_segments = []
    send_cursor = 0
    for destination_rank in range(world_size):
        split_start = send_cursor
        for binding_index, binding in enumerate(bindings):
            segment = _route_segment(
                binding,
                binding_index,
                group_rank,
                destination_rank,
                world_size,
                send_cursor,
            )
            if segment is None:
                continue
            send_segments.append(segment)
            send_cursor += segment.numel
        input_split_sizes.append(send_cursor - split_start)

    output_split_sizes = []
    receive_segments = []
    receive_cursor = 0
    for source_rank in range(world_size):
        split_start = receive_cursor
        for binding_index, binding in enumerate(bindings):
            segment = _route_segment(
                binding,
                binding_index,
                source_rank,
                group_rank,
                world_size,
                receive_cursor,
            )
            if segment is None:
                continue
            receive_segments.append(segment)
            receive_cursor += segment.numel
        output_split_sizes.append(receive_cursor - split_start)
    expected_send_numel = sum(
        Shard.local_shard_size_and_offset(
            binding.global_shape[0], world_size, group_rank
        )[0]
        * math.prod(binding.global_shape[1:])
        for binding in bindings
    )
    expected_receive_numel = sum(
        math.prod(compute_shape)
        for binding in bindings
        if (
            compute_shape := _redistributed_compute_shape(
                binding, group_rank, world_size
            )
        )
        is not None
    )
    assert send_cursor == expected_send_numel
    assert receive_cursor == expected_receive_numel
    return (
        input_split_sizes,
        output_split_sizes,
        tuple(send_segments),
        tuple(receive_segments),
    )
