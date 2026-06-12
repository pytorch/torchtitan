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
from ._copy import copy_tensors_to_dtype

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo


class Owned(Placement):
    """Placement where one rank owns the full parameter and other ranks hold empty tensors."""

    @dataclass(frozen=True)
    class _UnshardState:
        pg: Any
        debug_fqn: str | None

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
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
        """Prepare full-parameter buffers for owner-rank broadcast."""
        rank = mesh.get_local_rank()
        self._validate_owner_rank(mesh.size())

        with _record_function_if_eager("FlexShard::broadcast_copy_in", debug_fqn):
            if rank == self.owner_rank:
                full_params = copy_tensors_to_dtype(
                    tensors,
                    [info.unsharded_dtype for info in infos],
                )
            else:
                full_params = [
                    torch.empty(
                        info.global_shape,
                        dtype=info.unsharded_dtype,
                        device=tensor.device,
                    )
                    for tensor, info in zip(tensors, infos, strict=True)
                ]
        return PlacementPreparedUnshard(
            placement=self,
            buffers=full_params,
            placement_state=Owned._UnshardState(
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """Broadcast prepared full-parameter buffers from the owner rank."""
        if not isinstance(prepared.placement_state, Owned._UnshardState):
            raise AssertionError(
                "Expected Owned._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        for full_param in prepared.buffers:
            with _record_comm_if_eager(
                "FlexShard::broadcast",
                prepared.placement_state.debug_fqn,
            ):
                dist.broadcast(
                    full_param,
                    src=self.owner_rank,
                    group=prepared.placement_state.pg,
                )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Return the full parameters produced by the owner-rank broadcast."""
        return PlacementUnshardResult(list(prepared.buffers))

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        """Pack full gradients for reduce-to-owner."""
        self._validate_owner_rank(mesh.size())
        with _record_function_if_eager("FlexShard::reduce_copy_in", debug_fqn):
            send_tensors = copy_tensors_to_dtype(
                tensors,
                [info.grad_reduce_dtype for info in infos],
            )
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=send_tensors,
            placement_state=Owned._ReduceGradState(
                infos=infos,
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
        """Reduce full gradients to the owner rank and return local owned grads."""
        if not isinstance(prepared.placement_state, Owned._ReduceGradState):
            raise AssertionError(
                "Expected Owned._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        sharded_grads: list[torch.Tensor] = []
        for send_tensor, info in zip(
            prepared.buffers,
            prepared.placement_state.infos,
            strict=True,
        ):
            with _record_comm_if_eager(
                "FlexShard::reduce",
                prepared.placement_state.debug_fqn,
            ):
                dist.reduce(
                    send_tensor,
                    dst=self.owner_rank,
                    op=dist.ReduceOp.SUM,
                    group=prepared.placement_state.pg,
                )
            if prepared.placement_state.rank == self.owner_rank:
                send_tensor.div_(prepared.placement_state.world_size)
                sharded_grads.append(send_tensor)
            else:
                sharded_grads.append(
                    send_tensor.new_empty(
                        self.compute_local_shape(
                            info.global_shape,
                            prepared.placement_state.rank,
                            prepared.placement_state.world_size,
                        )
                    )
                )
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
