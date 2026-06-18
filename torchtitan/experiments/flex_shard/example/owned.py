# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import heapq
import math

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from typing_extensions import override

from ..flex_shard.placement_contract import (
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..flex_shard.utils import _record_comm_if_eager, _record_function_if_eager
from ._copy import pack_tensors_into_flat_buffer

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo


class Owned(Placement):
    """Placement where one rank owns the full parameter and other ranks hold empty tensors."""

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        offsets: list[int]
        numels: list[int]
        pg: Any
        debug_fqn: str | None

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        offsets: list[int]
        numels: list[int]
        rank: int
        world_size: int
        pg: Any
        debug_fqn: str | None

    def __init__(self, owner_rank: int):
        self.owner_rank = owner_rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Owned):
            return False
        return self.owner_rank == other.owner_rank

    def __hash__(self) -> int:
        return hash((type(self), self.owner_rank))

    def __repr__(self) -> str:
        return f"Owned({self.owner_rank})"

    def _validate_owner_rank(self, world_size: int) -> None:
        if self.owner_rank < 0 or self.owner_rank >= world_size:
            raise ValueError(
                f"Owned placement requires owner_rank in [0, {world_size}), "
                f"but got {self.owner_rank}."
            )

    def _try_get_contiguous_flat_bucket_view(
        self,
        tensors: list[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return a flat alias when bucket tensors are already contiguous."""
        if not tensors:
            return None

        device = tensors[0].device
        non_empty_tensors = [tensor for tensor in tensors if tensor.numel() > 0]
        if not non_empty_tensors:
            tensor = tensors[0]
            if tensor.dtype != dtype or tensor.device != device:
                return None
            return tensor.reshape(-1)

        first_tensor = non_empty_tensors[0]
        storage_data_ptr = first_tensor.untyped_storage().data_ptr()
        expected_storage_offset = first_tensor.storage_offset()
        total_numel = 0
        for tensor in tensors:
            numel = tensor.numel()
            if numel == 0:
                continue
            if tensor.dtype != dtype or tensor.device != device:
                return None
            if not tensor.is_contiguous():
                return None
            if tensor.untyped_storage().data_ptr() != storage_data_ptr:
                return None
            if tensor.storage_offset() != expected_storage_offset:
                return None
            expected_storage_offset += numel
            total_numel += numel

        return torch.as_strided(
            first_tensor,
            (total_numel,),
            (1,),
            storage_offset=first_tensor.storage_offset(),
        )

    def _require_uniform_dtype(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        attr: str,
        op_name: str,
    ) -> torch.dtype:
        if not infos or not tensors:
            raise AssertionError(f"Expected at least one tensor for {op_name}.")
        dtype = getattr(infos[0], attr, tensors[0].dtype)
        for tensor, info in zip(tensors[1:], infos[1:], strict=True):
            other = getattr(info, attr, tensor.dtype)
            if other != dtype:
                raise ValueError(
                    f"Owned {op_name} requires one communication dtype per bucket, "
                    f"but {infos[0].fqn!r} uses {dtype} and {info.fqn!r} uses "
                    f"{other}."
                )
        return dtype

    def _flat_layout(
        self,
        infos: list[ParamInfo],
    ) -> tuple[list[int], list[int], int]:
        numels = [math.prod(info.global_shape) for info in infos]
        offsets: list[int] = []
        total_numel = 0
        for numel in numels:
            offsets.append(total_numel)
            total_numel += numel
        return offsets, numels, total_numel

    def _views_from_flat(
        self,
        flat: torch.Tensor,
        infos: list[ParamInfo],
        offsets: list[int],
        numels: list[int],
    ) -> list[torch.Tensor]:
        if len(infos) != len(offsets) or len(infos) != len(numels):
            raise AssertionError("Expected flat bucket metadata to match infos.")
        return [
            flat.narrow(0, offset, numel).view(info.global_shape)
            for info, offset, numel in zip(infos, offsets, numels, strict=True)
        ]

    @override
    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        self._validate_owner_rank(world_size)
        if rank == self.owner_rank:
            return global_shape
        return torch.Size([0] * len(global_shape))

    @override
    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        self._validate_owner_rank(world_size)
        if rank == self.owner_rank:
            return math.prod(global_shape)
        return 0

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        self._validate_owner_rank(world_size)
        if rank == self.owner_rank:
            return param
        return param.new_empty(0)

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Prepare one flat full-parameter buffer for owner-rank broadcast."""
        rank = mesh.get_local_rank()
        self._validate_owner_rank(mesh.size())
        if len(tensors) != len(infos):
            raise AssertionError(
                f"Expected {len(tensors)} tensors to match {len(infos)} infos."
            )
        dtype = self._require_uniform_dtype(
            tensors,
            infos,
            "unsharded_dtype",
            "unshard",
        )
        offsets, numels, total_numel = self._flat_layout(infos)

        with _record_function_if_eager("FlexShard::broadcast_copy_in", debug_fqn):
            if rank == self.owner_rank:
                flat = None
                if not torch.compiler.is_compiling():
                    flat = self._try_get_contiguous_flat_bucket_view(tensors, dtype)
                if flat is None:
                    flat = pack_tensors_into_flat_buffer(tensors, dtype)
            else:
                flat = torch.empty(total_numel, dtype=dtype, device=tensors[0].device)
            if flat.numel() != total_numel:
                raise AssertionError(
                    f"Owned unshard flat buffer has {flat.numel()} elements, "
                    f"expected {total_numel}."
                )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[flat],
            placement_state=Owned._UnshardState(
                infos=infos,
                offsets=offsets,
                numels=numels,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """Broadcast the prepared flat full-parameter buffer from the owner rank."""
        if not isinstance(prepared.placement_state, Owned._UnshardState):
            raise AssertionError(
                "Expected Owned._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        if len(prepared.buffers) != 1:
            raise AssertionError(
                f"Expected one flat unshard buffer, got {len(prepared.buffers)}."
            )
        with _record_comm_if_eager(
            "FlexShard::broadcast",
            prepared.placement_state.debug_fqn,
        ):
            dist.broadcast(
                prepared.buffers[0],
                src=self.owner_rank,
                group=prepared.placement_state.pg,
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Return full-parameter views produced by the owner-rank broadcast."""
        if not isinstance(prepared.placement_state, Owned._UnshardState):
            raise AssertionError(
                "Expected Owned._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        if len(prepared.buffers) != 1:
            raise AssertionError(
                f"Expected one flat unshard buffer, got {len(prepared.buffers)}."
            )
        flat = prepared.buffers[0]
        full_params = self._views_from_flat(
            flat,
            prepared.placement_state.infos,
            prepared.placement_state.offsets,
            prepared.placement_state.numels,
        )
        return PlacementUnshardResult(full_params, prepared.buffers)

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        """Pack full gradients into one flat buffer for reduce-to-owner."""
        self._validate_owner_rank(mesh.size())
        if len(tensors) != len(infos):
            raise AssertionError(
                f"Expected {len(tensors)} tensors to match {len(infos)} infos."
            )
        dtype = self._require_uniform_dtype(
            tensors,
            infos,
            "grad_reduce_dtype",
            "reduce-grad",
        )
        offsets, numels, total_numel = self._flat_layout(infos)
        with _record_function_if_eager("FlexShard::reduce_copy_in", debug_fqn):
            flat = pack_tensors_into_flat_buffer(tensors, dtype)
            if flat.numel() != total_numel:
                raise AssertionError(
                    f"Owned reduce-grad flat buffer has {flat.numel()} elements, "
                    f"expected {total_numel}."
                )
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[flat],
            placement_state=Owned._ReduceGradState(
                infos=infos,
                offsets=offsets,
                numels=numels,
                rank=mesh.get_local_rank(),
                world_size=mesh.size(),
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        """Reduce one flat full-gradient buffer to the owner rank."""
        if not isinstance(prepared.placement_state, Owned._ReduceGradState):
            raise AssertionError(
                "Expected Owned._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        if len(prepared.buffers) != 1:
            raise AssertionError(
                f"Expected one flat reduce-grad buffer, got {len(prepared.buffers)}."
            )
        flat = prepared.buffers[0]
        with _record_comm_if_eager(
            "FlexShard::reduce",
            prepared.placement_state.debug_fqn,
        ):
            dist.reduce(
                flat,
                dst=self.owner_rank,
                op=dist.ReduceOp.SUM,
                group=prepared.placement_state.pg,
            )
        if prepared.placement_state.rank == self.owner_rank:
            flat.div_(prepared.placement_state.world_size)
            sharded_grads = self._views_from_flat(
                flat,
                prepared.placement_state.infos,
                prepared.placement_state.offsets,
                prepared.placement_state.numels,
            )
        else:
            sharded_grads = [
                flat.new_empty(
                    self.compute_local_shape(
                        info.global_shape,
                        prepared.placement_state.rank,
                        prepared.placement_state.world_size,
                    )
                )
                for info in prepared.placement_state.infos
            ]
        return PlacementReduceGradResult(sharded_grads)


def _assign_params_to_ranks(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
) -> dict[str, int]:
    """Greedily assign each full parameter to the currently least-loaded rank."""
    loads = [(0, rank) for rank in range(world_size)]
    heapq.heapify(loads)
    assignments: dict[str, int] = {}
    for fqn, param in sorted(
        named_params,
        key=lambda item: item[1].numel(),
        reverse=True,
    ):
        load, rank = heapq.heappop(loads)
        assignments[fqn] = rank
        heapq.heappush(loads, (load + param.numel(), rank))
    return assignments


def param_boundary_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Assign each full parameter to one rank using Owned placements."""
    assignments = _assign_params_to_ranks(named_params, mesh.size())
    return {fqn: (Owned(assignments[fqn]),) for fqn, _ in named_params}


__all__ = [
    "Owned",
    "param_boundary_placements",
]
