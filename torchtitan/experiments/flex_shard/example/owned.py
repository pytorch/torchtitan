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

    from ..flex_shard.bucket_storage import ParamInfo, PlacementFn


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
    """Assign each full parameter to one rank using Owned placements.

    Load-balances each parameter independently, so two params in the same bucket
    may get different owners. Only safe when each bucket holds a single parameter,
    because a FlexShard bucket must use one uniform placement. To keep a multi-param
    bucket (e.g. a whole transformer layer) on one rank, use
    :func:`make_owned_placement_fn` instead.
    """
    assignments = _assign_params_to_ranks(named_params, mesh.size())
    return {fqn: (Owned(assignments[fqn]),) for fqn, _ in named_params}


def make_owned_placement_fn(owner_rank: int) -> PlacementFn:
    """Return a placement function that assigns one owner to a whole bucket.

    Unlike :func:`param_boundary_placements`, which load-balances each parameter
    independently, this assigns *every* parameter in the bucket to ``owner_rank``.
    That keeps the bucket's placement uniform (FlexShard requires all params in a
    bucket to share one placement), so it is the helper to use when a bucket groups
    several parameters that must live together on one rank — e.g. a transformer
    layer whose 2D matrices the owner runs Muon on without any collective.
    """

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        return {fqn: (Owned(owner_rank),) for fqn, _ in named_params}

    return placement_fn


def assign_layer_owners_lpt(layer_numels: list[int], world_size: int) -> list[int]:
    """Balance whole layers across ranks with greedy Longest-Processing-Time.

    Returns one owner rank per layer such that total owned numel is as even as
    possible, subject to keeping each layer on a single rank (so the layer can be
    one uniform-placement Owned bucket). For homogeneous layers this matches
    round-robin; for heterogeneous layers (e.g. dense vs MoE, fatter first/last
    blocks) it balances far better. Residual imbalance is bounded by the largest
    single layer, which cannot be split while it stays in one bucket.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")
    loads = [(0, rank) for rank in range(world_size)]
    heapq.heapify(loads)
    owners = [0] * len(layer_numels)
    for layer_idx in sorted(
        range(len(layer_numels)),
        key=lambda i: layer_numels[i],
        reverse=True,
    ):
        load, rank = heapq.heappop(loads)
        owners[layer_idx] = rank
        heapq.heappush(loads, (load + layer_numels[layer_idx], rank))
    return owners


def assign_matrix_owners_per_layer_balanced(
    layer_matrix_numels: list[list[int]], world_size: int
) -> list[list[int]]:
    """Assign each layer's matrices to owner ranks, balanced *within every layer*.

    Use this for per-matrix ``Owned`` placement when you want every rank to hold a
    balanced share of *each* layer's matrices (so a layer's forward issues several
    smaller broadcasts from several sources, instead of one big single-source
    broadcast of the whole layer).

    Args:
        layer_matrix_numels: ``layer_matrix_numels[l]`` is the numels (or bytes) of
            layer ``l``'s ``Owned`` matrices.
        world_size: Number of ranks.

    Returns:
        ``owners`` where ``owners[l][k]`` is the owner rank of layer ``l``'s ``k``-th
        matrix (same order as the input).

    Within each layer the matrices are balanced across ranks with greedy
    Longest-Processing-Time; the residual per-rank spread within a layer is bounded
    by the largest single matrix (which cannot be split without breaking comm-free
    Muon). The per-layer assignment is then rotated by the layer index -- a
    bijection on ranks, so it preserves the within-layer balance while spreading the
    heaviest slot across ranks layer by layer, which also keeps the running totals
    balanced across layers (exactly so for homogeneous stacks whose layer count is a
    multiple of ``world_size``).
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")
    owners: list[list[int]] = []
    for layer_idx, numels in enumerate(layer_matrix_numels):
        loads = [(0, vrank) for vrank in range(world_size)]
        heapq.heapify(loads)
        layer_owners = [0] * len(numels)
        for i in sorted(range(len(numels)), key=lambda j: numels[j], reverse=True):
            load, vrank = heapq.heappop(loads)
            # Rotate virtual -> real rank so the heaviest slot moves each layer.
            layer_owners[i] = (vrank + layer_idx) % world_size
            heapq.heappush(loads, (load + numels[i], vrank))
        owners.append(layer_owners)
    return owners


__all__ = [
    "assign_layer_owners_lpt",
    "assign_matrix_owners_per_layer_balanced",
    "make_owned_placement_fn",
    "Owned",
    "param_boundary_placements",
]
