# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from typing_extensions import override

from ..flex_shard.placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..flex_shard.utils import _record_comm_if_eager, _record_function_if_eager

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import (
        BucketLayout,
        BucketParamLayout,
        ParamInfo,
        PlacementFn,
    )


class RaggedShard(Placement):
    """Ragged contiguous sharding over flattened prefix tensor dimensions.

    ``dims`` selects the prefix dimensions to flatten and partition. The first
    implementation keeps the contract intentionally strict: ``dims`` must be
    ``(0,)``, ``(0, 1)``, etc. ``local_units`` gives each rank's relative
    ownership along the flattened prefix. For example, a shape ``[8, hidden]``
    with ``local_units=(1, 2, 1, 0)`` gives ranks prefix lengths
    ``[2, 4, 2, 0]``.
    """

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        world_size: int
        pg: Any
        debug_fqn: str | None
        per_rank_param_offsets: list[list[int]]

    @dataclass(frozen=True)
    class _ReduceGradLayout:
        per_rank_param_offsets: list[list[int]]
        per_rank_sizes: list[int]
        padded_segment_numel: int

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        layout: RaggedShard._ReduceGradLayout
        rank: int
        world_size: int
        pg: Any
        debug_fqn: str | None

    def __init__(
        self,
        dims: tuple[int, ...] = (0,),
        local_units: tuple[int, ...] = (1,),
    ) -> None:
        if not dims:
            raise ValueError("RaggedShard dims must be non-empty.")
        expected_prefix_dims = tuple(range(len(dims)))
        if dims != expected_prefix_dims:
            raise ValueError(
                "RaggedShard currently requires prefix dims "
                f"{expected_prefix_dims}, but got {dims}."
            )
        if any(unit < 0 for unit in local_units):
            raise ValueError("RaggedShard local_units must be non-negative.")
        if sum(local_units) == 0:
            raise ValueError(
                "RaggedShard local_units must contain at least one positive unit."
            )
        self.dims = dims
        self.local_units = local_units

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RaggedShard):
            return False
        return self.dims == other.dims and self.local_units == other.local_units

    def __hash__(self) -> int:
        return hash((type(self), self.dims, self.local_units))

    def __repr__(self) -> str:
        return f"RaggedShard(dims={self.dims}, local_units={self.local_units})"

    def _validate_world_size(self, world_size: int) -> None:
        if len(self.local_units) != world_size:
            raise ValueError(
                "RaggedShard local_units length must match world size: "
                f"got {len(self.local_units)} local units for world size {world_size}."
            )

    def _prefix_numel(self, global_shape: torch.Size) -> int:
        if len(global_shape) < len(self.dims):
            raise ValueError(
                f"RaggedShard dims {self.dims} are invalid for parameter shape "
                f"{tuple(global_shape)}."
            )
        return math.prod(global_shape[: len(self.dims)])

    def _unit_prefix_numel(self, global_shape: torch.Size) -> int:
        prefix_numel = self._prefix_numel(global_shape)
        total_units = sum(self.local_units)
        if prefix_numel % total_units != 0:
            raise ValueError(
                "RaggedShard requires prod(global_shape[:len(dims)]) to be "
                "divisible by sum(local_units): "
                f"got prefix numel {prefix_numel} and local_units "
                f"{self.local_units}."
            )
        return prefix_numel // total_units

    def _suffix_shape(self, global_shape: torch.Size) -> torch.Size:
        return torch.Size(global_shape[len(self.dims) :])

    def _suffix_numel(self, global_shape: torch.Size) -> int:
        return math.prod(self._suffix_shape(global_shape))

    def _rank_prefix_range(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> tuple[int, int]:
        self._validate_world_size(world_size)
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"RaggedShard rank must be in [0, {world_size}), got {rank}."
            )
        unit_prefix_numel = self._unit_prefix_numel(global_shape)
        start = sum(self.local_units[:rank]) * unit_prefix_numel
        end = start + self.local_units[rank] * unit_prefix_numel
        return start, end

    def _rank_flat_range(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> tuple[int, int]:
        start_prefix, end_prefix = self._rank_prefix_range(
            global_shape,
            rank,
            world_size,
        )
        suffix_numel = self._suffix_numel(global_shape)
        return start_prefix * suffix_numel, end_prefix * suffix_numel

    def _bucket_layout(
        self,
        infos: list[ParamInfo],
        world_size: int,
    ) -> tuple[list[list[int]], list[int]]:
        per_rank_param_offsets: list[list[int]] = []
        per_rank_sizes: list[int] = []
        for rank in range(world_size):
            offset = 0
            offsets_r: list[int] = []
            for info in infos:
                offsets_r.append(offset)
                offset += self.compute_local_numel(
                    info.global_shape,
                    rank,
                    world_size,
                )
            per_rank_param_offsets.append(offsets_r)
            per_rank_sizes.append(offset)
        return per_rank_param_offsets, per_rank_sizes

    @override
    def compute_local_shape(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> torch.Size:
        start_prefix, end_prefix = self._rank_prefix_range(
            global_shape,
            rank,
            world_size,
        )
        return torch.Size(
            [end_prefix - start_prefix, *self._suffix_shape(global_shape)]
        )

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        start, end = self._rank_flat_range(param.shape, rank, world_size)
        return (
            param.contiguous()
            .view(-1)[start:end]
            .view(self.compute_local_shape(param.shape, rank, world_size))
        )

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Prepare buffers for ragged all-gather unshard."""
        world_size = mesh.size()
        dtype = tensors[0].dtype
        device = tensors[0].device

        with _record_function_if_eager("FlexShard::all_gather_copy_in", debug_fqn):
            send_buf = torch.cat([tensor.reshape(-1) for tensor in tensors])
            per_rank_param_offsets, per_rank_sizes = self._bucket_layout(
                infos,
                world_size,
            )
            gathered = [
                torch.empty(per_rank_sizes[rank], dtype=dtype, device=device)
                for rank in range(world_size)
            ]

        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send_buf, *gathered],
            placement_state=RaggedShard._UnshardState(
                infos=infos,
                world_size=world_size,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                per_rank_param_offsets=per_rank_param_offsets,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """Launch the prepared ragged all-gather."""
        if not isinstance(prepared.placement_state, RaggedShard._UnshardState):
            raise AssertionError(
                "Expected RaggedShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        send_buf = prepared.buffers[0]
        gathered = prepared.buffers[1:]
        with _record_comm_if_eager(
            "FlexShard::all_gather",
            prepared.placement_state.debug_fqn,
        ):
            dist.all_gather(gathered, send_buf, group=prepared.placement_state.pg)

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Finish ragged all-gather and assemble full parameters."""
        if not isinstance(prepared.placement_state, RaggedShard._UnshardState):
            raise AssertionError(
                "Expected RaggedShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        gathered = prepared.buffers[1:]
        with _record_function_if_eager(
            "FlexShard::all_gather_copy_out",
            prepared.placement_state.debug_fqn,
        ):
            full_params: list[torch.Tensor] = []
            for info_idx, info in enumerate(prepared.placement_state.infos):
                per_rank_shards: list[torch.Tensor] = []
                for rank in range(prepared.placement_state.world_size):
                    numel = self.compute_local_numel(
                        info.global_shape,
                        rank,
                        prepared.placement_state.world_size,
                    )
                    offset = prepared.placement_state.per_rank_param_offsets[rank][
                        info_idx
                    ]
                    per_rank_shards.append(gathered[rank][offset : offset + numel])
                full_params.append(torch.cat(per_rank_shards).view(info.global_shape))

        return PlacementUnshardResult(full_params, prepared.buffers)

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        """Pack full gradients into padded per-rank reduce-scatter segments."""
        world_size = mesh.size()
        with _record_function_if_eager("FlexShard::reduce_scatter_copy_in", debug_fqn):
            per_rank_param_offsets, per_rank_sizes = self._bucket_layout(
                infos,
                world_size,
            )
            padded_segment_numel = max(per_rank_sizes)
            dtype = tensors[0].dtype
            device = tensors[0].device
            send_buf = torch.zeros(
                world_size * padded_segment_numel,
                dtype=dtype,
                device=device,
            )
            send_buf_by_rank = send_buf.view(world_size, padded_segment_numel)
            for rank in range(world_size):
                for info_idx, (tensor, info) in enumerate(
                    zip(tensors, infos, strict=True)
                ):
                    shard = self.extract_local_shard(
                        tensor,
                        rank,
                        world_size,
                    ).reshape(-1)
                    offset = per_rank_param_offsets[rank][info_idx]
                    send_buf_by_rank[rank, offset : offset + shard.numel()].copy_(shard)
            layout = RaggedShard._ReduceGradLayout(
                per_rank_param_offsets=per_rank_param_offsets,
                per_rank_sizes=per_rank_sizes,
                padded_segment_numel=padded_segment_numel,
            )

        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send_buf],
            placement_state=RaggedShard._ReduceGradState(
                infos=infos,
                layout=layout,
                rank=mesh.get_local_rank(),
                world_size=world_size,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        """Reduce ragged full gradients and return this rank's local shards."""
        if not isinstance(prepared.placement_state, RaggedShard._ReduceGradState):
            raise AssertionError(
                "Expected RaggedShard._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        send_buf = prepared.buffers[0]
        recv_buf = torch.empty(
            state.layout.padded_segment_numel,
            dtype=send_buf.dtype,
            device=send_buf.device,
        )
        with _record_comm_if_eager(
            "FlexShard::reduce_scatter",
            state.debug_fqn,
        ):
            # TODO: Plumb the reduction/scaling policy from SPMD gradient semantics.
            # AVG is a convenient default, but delayed grad scaling may need SUM
            # plus an explicit scale at a different point in the training step.
            dist.reduce_scatter_tensor(
                output=recv_buf,
                input=send_buf,
                op=dist.ReduceOp.AVG,
                group=state.pg,
            )

        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_out",
            state.debug_fqn,
        ):
            sharded_grads: list[torch.Tensor] = []
            valid_recv_buf = recv_buf[: state.layout.per_rank_sizes[state.rank]]
            for info_idx, info in enumerate(state.infos):
                numel = self.compute_local_numel(
                    info.global_shape,
                    state.rank,
                    state.world_size,
                )
                shape = self.compute_local_shape(
                    info.global_shape,
                    state.rank,
                    state.world_size,
                )
                offset = state.layout.per_rank_param_offsets[state.rank][info_idx]
                sharded_grads.append(
                    valid_recv_buf[offset : offset + numel].view(shape).contiguous()
                )
        return PlacementReduceGradResult(sharded_grads, [recv_buf])


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"Expected positive alignment, got {alignment}.")
    return ((value + alignment - 1) // alignment) * alignment


class GroupedRaggedShard(RaggedShard):
    """Bucket-global RaggedShard with DBuffer-style param-major layout.

    Unlike ``RaggedShard``, this placement plans the whole bucket as one
    param-major logical buffer, then shards that buffer into rank-local ranges.
    That makes the all-gather output directly viewable as full parameters.
    """

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        pg: Any
        debug_fqn: str | None

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        rank: int
        pg: Any
        debug_fqn: str | None
        padded_segment_numel: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupedRaggedShard):
            return False
        return self.dims == other.dims and self.local_units == other.local_units

    def __hash__(self) -> int:
        return hash((type(self), self.dims, self.local_units))

    def __repr__(self) -> str:
        return (
            "GroupedRaggedShard(" f"dims={self.dims}, local_units={self.local_units})"
        )

    def _bucket_layout(self, info: ParamInfo) -> BucketLayout:
        layout = info.bucket_layout
        if layout is None:
            raise AssertionError(
                "Expected GroupedRaggedShard ParamInfo to carry a bucket layout."
            )
        return layout

    def _param_layout(self, info: ParamInfo) -> BucketParamLayout:
        layout = self._bucket_layout(info)
        return layout.param_layouts[info.fqn]

    @override
    def compute_local_shape(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> torch.Size:
        raise NotImplementedError(
            "GroupedRaggedShard computes local shapes from the whole bucket. "
            "Use bucket_storage_layout() instead."
        )

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "GroupedRaggedShard extracts local shards from bucket-global ranges. "
            "Use bucket storage views instead."
        )

    def _param_alignment_numel(
        self,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> int:
        alignment = 1
        for _, param in named_params:
            self._prefix_numel(param.shape)
            suffix_numel = self._suffix_numel(param.shape)
            alignment = math.lcm(alignment, suffix_numel)
        return alignment

    def _bucket_param_offsets(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        alignment_numel: int,
    ) -> tuple[dict[str, int], int]:
        offsets: dict[str, int] = {}
        current_offset = 0
        for fqn, param in named_params:
            current_offset = _align_up(current_offset, alignment_numel)
            offsets[fqn] = current_offset
            current_offset += param.numel()
        return offsets, current_offset

    def _bucket_rank_layout(
        self,
        unpadded_global_numel: int,
        alignment_numel: int,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        total_units = sum(self.local_units)
        padded_global_numel = _align_up(
            unpadded_global_numel,
            alignment_numel * total_units,
        )
        elements_per_unit = padded_global_numel // total_units
        offsets: list[int] = []
        numels: list[int] = []
        offset = 0
        for units in self.local_units:
            offsets.append(offset)
            numel = units * elements_per_unit
            numels.append(numel)
            offset += numel
        return padded_global_numel, tuple(offsets), tuple(numels)

    def _local_shape_from_numel(
        self,
        global_shape: torch.Size,
        local_numel: int,
    ) -> torch.Size:
        suffix_shape = self._suffix_shape(global_shape)
        suffix_numel = math.prod(suffix_shape)
        if local_numel % suffix_numel != 0:
            raise ValueError(
                "GroupedRaggedShard bucket split produced a shard that is not "
                f"aligned to flattened prefix rows for shape {tuple(global_shape)}."
            )
        return torch.Size([local_numel // suffix_numel, *suffix_shape])

    @override
    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout:
        from ..flex_shard.bucket_storage import BucketLayout, BucketParamLayout

        if not named_params:
            return BucketStorageLayout(param_layouts={}, total_bytes=0)
        if len(self.local_units) != mesh.size():
            raise ValueError(
                "GroupedRaggedShard local_units length must match world size: "
                f"got {len(self.local_units)} local units for world size "
                f"{mesh.size()}."
            )

        rank = mesh.get_local_rank()
        dtype = named_params[0][1].dtype
        alignment_numel = self._param_alignment_numel(named_params)
        param_offsets, unpadded_global_numel = self._bucket_param_offsets(
            named_params,
            alignment_numel,
        )
        (
            padded_global_numel,
            rank_offsets,
            rank_numels,
        ) = self._bucket_rank_layout(unpadded_global_numel, alignment_numel)
        rank_start = rank_offsets[rank]
        rank_end = rank_start + rank_numels[rank]

        local_metadata: dict[str, tuple[torch.Size, int, int]] = {}
        param_layouts: dict[str, BucketParamLayout] = {}
        has_local_param_data = False
        for fqn, param in named_params:
            if param.dtype != dtype:
                raise ValueError(
                    "GroupedRaggedShard requires one dtype per bucket: "
                    f"{named_params[0][0]!r} uses {dtype} but {fqn!r} uses "
                    f"{param.dtype}."
                )
            placements = param_placements[fqn]
            if placements != (self,):
                raise ValueError(
                    "GroupedRaggedShard requires the same placement instance "
                    f"for every parameter in a bucket; {fqn!r} uses {placements!r}."
                )
            param_start = param_offsets[fqn]
            param_end = param_start + param.numel()
            local_start = max(rank_start, param_start)
            local_end = min(rank_end, param_end)
            local_numel = max(0, local_end - local_start)
            if local_numel > 0:
                has_local_param_data = True
            local_shape = self._local_shape_from_numel(param.shape, local_numel)
            byte_offset = (local_start - rank_start) * dtype.itemsize
            local_metadata[fqn] = (local_shape, local_numel, byte_offset)
            param_layouts[fqn] = BucketParamLayout(
                param_offset=param_start,
                local_global_offset=local_start,
            )

        bucket_layout = BucketLayout(
            global_numel=padded_global_numel,
            local_numel=rank_numels[rank],
            rank_offsets=rank_offsets,
            rank_numels=rank_numels,
            param_layouts=param_layouts,
        )
        storage_layouts: dict[str, BucketParamStorageLayout] = {}
        for fqn, _ in named_params:
            local_shape, local_numel, byte_offset = local_metadata[fqn]
            storage_layouts[fqn] = BucketParamStorageLayout(
                local_shape=local_shape,
                local_numel=local_numel,
                byte_offset=byte_offset,
                storage_nbytes=local_numel * dtype.itemsize,
                bucket_layout=bucket_layout,
            )

        if rank_numels[rank] > 0 and not has_local_param_data:
            raise ValueError(
                "GroupedRaggedShard planned a rank-local bucket range that "
                "contains only padding. Split the bucket or choose local_units "
                "that assign parameter data to every non-empty rank."
            )

        return BucketStorageLayout(
            param_layouts=storage_layouts,
            total_bytes=rank_numels[rank] * dtype.itemsize,
        )

    @override
    def copy_param_to_storage(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        param_data = param.detach()
        if param_data.device.type == "meta" or info.local_numel == 0:
            return
        param_layout = self._param_layout(info)
        param_offset = param_layout.local_global_offset - param_layout.param_offset
        shard = param_data.contiguous().view(-1)[
            param_offset : param_offset + info.local_numel
        ]
        nbytes = shard.numel() * shard.element_size()
        byte_storage[info.byte_offset : info.byte_offset + nbytes].copy_(
            shard.view(torch.uint8)
        )

    def _make_local_bucket_view(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
    ) -> torch.Tensor:
        bucket_layout = self._bucket_layout(infos[0])
        local_bucket_numel = bucket_layout.local_numel
        if local_bucket_numel == 0:
            return tensors[0].new_empty(0)

        bucket_storage: torch.UntypedStorage | None = None
        bucket_start_ptr: int | None = None
        bucket_tensor: torch.Tensor | None = None
        for tensor, info in zip(tensors, infos, strict=True):
            if tensor.numel() == 0:
                continue
            storage = tensor.untyped_storage()
            storage_start_ptr = tensor.data_ptr() - info.byte_offset
            if bucket_storage is None:
                bucket_storage = storage
                bucket_start_ptr = storage_start_ptr
                bucket_tensor = tensor
            elif (
                storage.data_ptr() != bucket_storage.data_ptr()
                or storage_start_ptr != bucket_start_ptr
            ):
                raise ValueError(
                    "GroupedRaggedShard expected local tensors to share one "
                    "bucket storage allocation."
                )
            storage_offset_bytes = storage_start_ptr - storage.data_ptr()
            if storage_offset_bytes % tensor.element_size() != 0:
                raise AssertionError(
                    "GroupedRaggedShard local bucket storage is not dtype-aligned."
                )

        if bucket_storage is None or bucket_start_ptr is None or bucket_tensor is None:
            raise ValueError(
                "GroupedRaggedShard cannot create a view-in send buffer because "
                "this rank has no parameter tensor covering its non-empty bucket "
                "range."
            )
        storage_offset_bytes = bucket_start_ptr - bucket_storage.data_ptr()
        storage_offset = storage_offset_bytes // bucket_tensor.element_size()
        return bucket_tensor.new_empty(0).set_(
            bucket_storage,
            storage_offset,
            (local_bucket_numel,),
            (1,),
        )

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        dtype = tensors[0].dtype
        device = tensors[0].device
        send_buf = self._make_local_bucket_view(tensors, infos)
        bucket_layout = self._bucket_layout(infos[0])
        gathered_bucket = torch.empty(
            bucket_layout.global_numel,
            dtype=dtype,
            device=device,
        )
        gathered_views = [
            gathered_bucket[offset : offset + numel]
            for offset, numel in zip(
                bucket_layout.rank_offsets,
                bucket_layout.rank_numels,
                strict=True,
            )
        ]
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send_buf, gathered_bucket, *gathered_views],
            placement_state=GroupedRaggedShard._UnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        if not isinstance(prepared.placement_state, GroupedRaggedShard._UnshardState):
            raise AssertionError(
                "Expected GroupedRaggedShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        send_buf = prepared.buffers[0]
        gathered_views = prepared.buffers[2:]
        with _record_comm_if_eager(
            "FlexShard::all_gather",
            prepared.placement_state.debug_fqn,
        ):
            dist.all_gather(gathered_views, send_buf, group=prepared.placement_state.pg)

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        if not isinstance(prepared.placement_state, GroupedRaggedShard._UnshardState):
            raise AssertionError(
                "Expected GroupedRaggedShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        gathered_bucket = prepared.buffers[1]
        full_params: list[torch.Tensor] = []
        for info in prepared.placement_state.infos:
            param_offset = self._param_layout(info).param_offset
            full_params.append(
                gathered_bucket[param_offset : param_offset + info.global_numel].view(
                    info.global_shape
                )
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
        world_size = mesh.size()
        dtype = tensors[0].dtype
        device = tensors[0].device
        bucket_layout = self._bucket_layout(infos[0])
        padded_segment_numel = max(bucket_layout.rank_numels)
        with _record_function_if_eager("FlexShard::reduce_scatter_copy_in", debug_fqn):
            global_grad_bucket = torch.zeros(
                bucket_layout.global_numel,
                dtype=dtype,
                device=device,
            )
            for tensor, info in zip(tensors, infos, strict=True):
                param_layout = self._param_layout(info)
                global_grad_bucket[
                    param_layout.param_offset : param_layout.param_offset
                    + info.global_numel
                ].copy_(tensor.reshape(-1))

            send_buf = torch.zeros(
                world_size * padded_segment_numel,
                dtype=dtype,
                device=device,
            )
            send_buf_by_rank = send_buf.view(world_size, padded_segment_numel)
            for rank, (offset, numel) in enumerate(
                zip(
                    bucket_layout.rank_offsets,
                    bucket_layout.rank_numels,
                    strict=True,
                )
            ):
                send_buf_by_rank[rank, :numel].copy_(
                    global_grad_bucket[offset : offset + numel]
                )

        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send_buf],
            placement_state=GroupedRaggedShard._ReduceGradState(
                infos=infos,
                rank=mesh.get_local_rank(),
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                padded_segment_numel=padded_segment_numel,
            ),
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        if not isinstance(
            prepared.placement_state, GroupedRaggedShard._ReduceGradState
        ):
            raise AssertionError(
                "Expected GroupedRaggedShard._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        send_buf = prepared.buffers[0]
        recv_buf = torch.empty(
            state.padded_segment_numel,
            dtype=send_buf.dtype,
            device=send_buf.device,
        )
        with _record_comm_if_eager(
            "FlexShard::reduce_scatter",
            state.debug_fqn,
        ):
            dist.reduce_scatter_tensor(
                output=recv_buf,
                input=send_buf,
                op=dist.ReduceOp.AVG,
                group=state.pg,
            )

        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_out",
            state.debug_fqn,
        ):
            sharded_grads = [
                recv_buf[
                    info.byte_offset
                    // info.dtype.itemsize : info.byte_offset
                    // info.dtype.itemsize
                    + info.local_numel
                ].view(info.local_shape)
                for info in state.infos
            ]
        return PlacementReduceGradResult(sharded_grads, [recv_buf])


def make_ragged_placement_fn(
    *,
    dims: tuple[int, ...],
    local_units: tuple[int, ...],
) -> PlacementFn:
    """Return a placement function assigning one RaggedShard to every param."""

    def ragged_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        if len(local_units) != mesh.size():
            raise ValueError(
                "RaggedShard local_units length must match mesh size: "
                f"got {len(local_units)} local units for mesh size {mesh.size()}."
            )
        placement = RaggedShard(dims=dims, local_units=local_units)
        return {fqn: (placement,) for fqn, _ in named_params}

    return ragged_placements


def make_grouped_ragged_placement_fn(
    *,
    dims: tuple[int, ...],
    local_units: tuple[int, ...],
) -> PlacementFn:
    """Return a placement function assigning one grouped RaggedShard per bucket."""

    def grouped_ragged_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        if len(local_units) != mesh.size():
            raise ValueError(
                "GroupedRaggedShard local_units length must match mesh size: "
                f"got {len(local_units)} local units for mesh size {mesh.size()}."
            )
        placement = GroupedRaggedShard(dims=dims, local_units=local_units)
        return {fqn: (placement,) for fqn, _ in named_params}

    return grouped_ragged_placements


def per_param_ragged_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Even-unit RaggedShard over dim 0 per parameter."""
    return make_ragged_placement_fn(
        dims=(0,),
        local_units=(1,) * mesh.size(),
    )(named_params, mesh)


__all__ = [
    "GroupedRaggedShard",
    "make_grouped_ragged_placement_fn",
    "make_ragged_placement_fn",
    "per_param_ragged_placements",
    "RaggedShard",
]
