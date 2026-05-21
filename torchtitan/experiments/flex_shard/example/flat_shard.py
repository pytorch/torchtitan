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
from torch._prims_common import make_contiguous_strides_for
from typing_extensions import override

from ..flex_shard.placement_contract import (
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..flex_shard.utils import _record_comm_if_eager, _record_function_if_eager

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo


class FlatShard(Placement):
    """FSDP1-style flat bucket sharding with rank-flat bucket storage."""

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        total_numel: int
        shard_numel: int
        world_size: int
        pg: Any
        debug_fqn: str | None

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        total_numel: int
        shard_numel: int
        rank: int
        world_size: int
        pg: Any
        debug_fqn: str | None

    def __init__(
        self,
        total_numel: int,
        param_start: int,
        param_numel: int,
        *,
        bucket_key: object | None = None,
    ) -> None:
        if total_numel < 0 or param_start < 0 or param_numel < 0:
            raise ValueError(
                "FlatShard requires non-negative total_numel, param_start, "
                f"and param_numel, but got total_numel={total_numel}, "
                f"param_start={param_start}, param_numel={param_numel}."
            )
        if param_start + param_numel > total_numel:
            raise ValueError(
                "FlatShard param interval must fit within the flat bucket, "
                f"but [{param_start}, {param_start + param_numel}) exceeds "
                f"total_numel={total_numel}."
            )
        self.total_numel = total_numel
        self.param_start = param_start
        self.param_numel = param_numel
        self.bucket_key = bucket_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlatShard):
            return False
        return (
            self.total_numel == other.total_numel
            and self.param_start == other.param_start
            and self.param_numel == other.param_numel
            and self.bucket_key == other.bucket_key
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.total_numel,
                self.param_start,
                self.param_numel,
                self.bucket_key,
            )
        )

    def __repr__(self) -> str:
        return (
            "FlatShard("
            f"total_numel={self.total_numel}, "
            f"param_start={self.param_start}, "
            f"param_numel={self.param_numel}, "
            f"bucket_key={self.bucket_key!r})"
        )

    @override
    def bucket_compatibility_key(self) -> object:
        return (type(self), self.bucket_key, self.total_numel)

    @override
    def uses_bucket_param_infos(self) -> bool:
        return True

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        if numerator < 0:
            raise ValueError(f"Expected non-negative numerator, got {numerator}.")
        if denominator <= 0:
            raise ValueError(f"Expected positive denominator, got {denominator}.")
        return (numerator + denominator - 1) // denominator

    def _rank_flat_bounds(self, rank: int, world_size: int) -> tuple[int, int]:
        if world_size <= 0:
            raise ValueError(f"Expected positive world_size, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"Expected rank in [0, {world_size}), but got rank={rank}."
            )
        shard_numel = self._ceil_div(self.total_numel, world_size)
        start = rank * shard_numel
        return start, start + shard_numel

    def _local_param_bounds(self, rank: int, world_size: int) -> tuple[int, int]:
        rank_start, rank_end = self._rank_flat_bounds(rank, world_size)
        local_start = min(max(rank_start - self.param_start, 0), self.param_numel)
        local_end = min(max(rank_end - self.param_start, 0), self.param_numel)
        return local_start, local_end

    @override
    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        global_numel = math.prod(global_shape)
        if global_numel != self.param_numel:
            raise ValueError(
                "FlatShard param_numel must match the parameter shape, but "
                f"param_numel={self.param_numel} and shape {tuple(global_shape)} "
                f"has {global_numel} elements."
            )
        local_start, local_end = self._local_param_bounds(rank, world_size)
        return local_end - local_start

    @override
    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        return torch.Size([self.compute_local_numel(global_shape, rank, world_size)])

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        if param.numel() != self.param_numel:
            raise ValueError(
                "FlatShard param_numel must match the source tensor, but "
                f"param_numel={self.param_numel} and tensor shape "
                f"{tuple(param.shape)} has {param.numel()} elements."
            )
        local_start, local_end = self._local_param_bounds(rank, world_size)
        return param.reshape(-1)[local_start:local_end]

    @staticmethod
    def _flat_shards_from_named_params(
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
    ) -> list[FlatShard]:
        placements: list[FlatShard] = []
        reference_key: object | None = None
        reference_fqn: str | None = None
        for fqn, param in named_params:
            placements_for_param = param_placements[fqn]
            if len(placements_for_param) != 1:
                raise ValueError(
                    "FlatShard bucket storage currently supports exactly one "
                    f"placement per parameter, but {fqn!r} has "
                    f"{len(placements_for_param)} placements."
                )
            placement = placements_for_param[0]
            if not isinstance(placement, FlatShard):
                raise ValueError(
                    "FlatShard bucket storage requires FlatShard placements, "
                    f"but {fqn!r} uses {placement!r}."
                )
            key = placement.bucket_compatibility_key()
            if reference_key is None:
                reference_key = key
                reference_fqn = fqn
            elif key != reference_key:
                assert reference_fqn is not None
                raise ValueError(
                    "FlatShard bucket storage requires bucket-compatible "
                    f"placements, but {reference_fqn!r} and {fqn!r} are "
                    "incompatible."
                )
            if param.numel() != placement.param_numel:
                raise ValueError(
                    f"FlatShard placement for {fqn!r} has "
                    f"param_numel={placement.param_numel}, but the parameter "
                    f"has {param.numel()} elements."
                )
            placements.append(placement)

        if placements:
            total_numel = placements[0].total_numel
            covered_numel = 0
            for start, end, fqn in sorted(
                (
                    placement.param_start,
                    placement.param_start + placement.param_numel,
                    fqn,
                )
                for (fqn, _), placement in zip(
                    named_params,
                    placements,
                    strict=True,
                )
            ):
                if start != covered_numel:
                    raise ValueError(
                        "FlatShard placements must exactly cover the flat "
                        "bucket without gaps or overlaps, but expected the "
                        f"next interval to start at {covered_numel} and {fqn!r} "
                        f"starts at {start}."
                    )
                covered_numel = end
            if covered_numel != total_numel:
                raise ValueError(
                    "FlatShard placements must exactly cover the flat bucket, "
                    f"but covered {covered_numel} elements and total_numel is "
                    f"{total_numel}."
                )
        return placements

    @override
    def create_bucket_param_infos(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> tuple[dict[str, ParamInfo], int] | None:
        from ..flex_shard.bucket_storage import (
            BucketLayout,
            BucketParamLayout,
            ParamInfo,
        )

        if not named_params:
            return {}, 0

        placements = self._flat_shards_from_named_params(
            named_params,
            param_placements,
        )
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        dtype = named_params[0][1].dtype
        total_numel = placements[0].total_numel
        shard_numel = self._ceil_div(total_numel, world_size)
        padded_total_numel = shard_numel * world_size
        rank_offsets = tuple(rank * shard_numel for rank in range(world_size))
        rank_numels = tuple(shard_numel for _ in range(world_size))
        rank_start = rank_offsets[rank]

        param_layouts = {
            fqn: BucketParamLayout(
                param_offset=placement.param_start,
                local_global_offset=placement.param_start
                + placement._local_param_bounds(rank, world_size)[0],
            )
            for (fqn, _), placement in zip(
                named_params,
                placements,
                strict=True,
            )
        }
        bucket_layout = BucketLayout(
            global_numel=padded_total_numel,
            local_numel=shard_numel,
            rank_offsets=rank_offsets,
            rank_numels=rank_numels,
            param_layouts=param_layouts,
        )

        param_infos: dict[str, ParamInfo] = {}
        for (fqn, param), placement in zip(
            named_params,
            placements,
            strict=True,
        ):
            if param.dtype != dtype:
                raise ValueError(
                    "FlatShard requires one dtype per bucket: "
                    f"{named_params[0][0]!r} uses {dtype} but {fqn!r} uses "
                    f"{param.dtype}."
                )
            local_start, local_end = placement._local_param_bounds(rank, world_size)
            local_numel = local_end - local_start
            byte_offset = (
                (placement.param_start + local_start - rank_start) * dtype.itemsize
                if local_numel > 0
                else 0
            )
            param_infos[fqn] = ParamInfo(
                fqn=fqn,
                global_shape=param.shape,
                global_stride=tuple(make_contiguous_strides_for(param.shape)),
                dtype=dtype,
                requires_grad=param.requires_grad,
                placements=param_placements[fqn],
                local_shape=torch.Size([local_numel]),
                local_numel=local_numel,
                byte_offset=byte_offset,
                storage_nbytes=local_numel * dtype.itemsize,
                global_numel=param.numel(),
                bucket_layout=bucket_layout,
            )

        return param_infos, shard_numel * dtype.itemsize

    @override
    def copy_param_to_storage(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        bucket_layout = info.bucket_layout
        if (
            bucket_layout is not None
            and next(iter(bucket_layout.param_layouts)) == info.fqn
        ):
            byte_storage.zero_()
        super().copy_param_to_storage(
            byte_storage,
            info,
            param,
            rank,
            world_size,
        )

    @staticmethod
    def _flat_shards_from_infos(infos: list[ParamInfo]) -> list[FlatShard]:
        if not infos:
            raise ValueError("FlatShard bucket collective requires at least one param.")

        placements: list[FlatShard] = []
        reference_key: object | None = None
        reference_info: ParamInfo | None = None
        for info in infos:
            placement = info.placement
            if not isinstance(placement, FlatShard):
                raise ValueError(
                    "FlatShard bucket collective requires FlatShard placements, "
                    f"but {info.fqn!r} uses {placement!r}."
                )
            key = placement.bucket_compatibility_key()
            if reference_key is None:
                reference_key = key
                reference_info = info
            elif key != reference_key:
                assert reference_info is not None
                raise ValueError(
                    "FlatShard bucket collective requires bucket-compatible "
                    f"placements, but {reference_info.fqn!r} uses "
                    f"{reference_info.placement!r} and {info.fqn!r} uses "
                    f"{placement!r}."
                )
            if info.global_numel != placement.param_numel:
                raise ValueError(
                    f"FlatShard placement for {info.fqn!r} has "
                    f"param_numel={placement.param_numel}, but ParamInfo "
                    f"has global_numel={info.global_numel}."
                )
            placements.append(placement)
        return placements

    @staticmethod
    def _copy_local_tensors_into_rank_shard(
        send_buf: torch.Tensor,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        placements: list[FlatShard],
        rank: int,
        world_size: int,
    ) -> None:
        for tensor, info, placement in zip(tensors, infos, placements, strict=True):
            local_start, local_end = placement._local_param_bounds(rank, world_size)
            local_numel = local_end - local_start
            if tensor.numel() != local_numel:
                raise ValueError(
                    f"FlatShard expected local tensor for {info.fqn!r} to have "
                    f"{local_numel} elements, but got {tensor.numel()}."
                )
            if local_numel == 0:
                continue
            rank_start, _ = placement._rank_flat_bounds(rank, world_size)
            dst_start = placement.param_start + local_start - rank_start
            send_buf[dst_start : dst_start + local_numel].copy_(tensor.reshape(-1))

    @staticmethod
    def _validate_local_tensor_numels(
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        placements: list[FlatShard],
        rank: int,
        world_size: int,
    ) -> None:
        for tensor, info, placement in zip(tensors, infos, placements, strict=True):
            local_start, local_end = placement._local_param_bounds(rank, world_size)
            local_numel = local_end - local_start
            if tensor.numel() != local_numel:
                raise ValueError(
                    f"FlatShard expected local tensor for {info.fqn!r} to have "
                    f"{local_numel} elements, but got {tensor.numel()}."
                )

    @staticmethod
    def _rank_valid_numel(total_numel: int, shard_numel: int, rank: int) -> int:
        rank_start = rank * shard_numel
        if rank_start >= total_numel:
            return 0
        return min(shard_numel, total_numel - rank_start)

    @staticmethod
    def _try_make_bucket_storage_view(
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
    ) -> torch.Tensor | None:
        bucket_layout = infos[0].bucket_layout
        if bucket_layout is None:
            return None

        local_bucket_numel = bucket_layout.local_numel
        if local_bucket_numel == 0:
            return tensors[0].new_empty(0)

        bucket_storage: torch.UntypedStorage | None = None
        bucket_storage_offset_bytes: int | None = None
        bucket_tensor: torch.Tensor | None = None
        for tensor, info in zip(tensors, infos, strict=True):
            if info.bucket_layout != bucket_layout:
                raise ValueError(
                    "FlatShard expected every ParamInfo in a bucket to share "
                    "the same bucket layout."
                )
            if tensor.dtype != tensors[0].dtype or tensor.device != tensors[0].device:
                return None
            if not tensor.is_contiguous():
                return None
            if info.byte_offset % tensor.element_size() != 0:
                raise AssertionError(
                    "FlatShard local bucket storage is not dtype-aligned."
                )

            storage = tensor.untyped_storage()
            storage_offset_bytes = tensor.storage_offset() * tensor.element_size()
            if storage_offset_bytes < info.byte_offset:
                return None
            candidate_bucket_offset_bytes = storage_offset_bytes - info.byte_offset
            if bucket_storage is None:
                bucket_storage = storage
                bucket_storage_offset_bytes = candidate_bucket_offset_bytes
                bucket_tensor = tensor
            elif (
                storage.data_ptr() != bucket_storage.data_ptr()
                or candidate_bucket_offset_bytes != bucket_storage_offset_bytes
            ):
                return None

        if (
            bucket_storage is None
            or bucket_storage_offset_bytes is None
            or bucket_tensor is None
        ):
            return None
        required_nbytes = (
            bucket_storage_offset_bytes
            + local_bucket_numel * bucket_tensor.element_size()
        )
        if bucket_storage.nbytes() < required_nbytes:
            return None
        if bucket_storage_offset_bytes % bucket_tensor.element_size() != 0:
            raise AssertionError("FlatShard bucket storage is not dtype-aligned.")

        return bucket_tensor.new_empty(0).set_(
            bucket_storage,
            bucket_storage_offset_bytes // bucket_tensor.element_size(),
            (local_bucket_numel,),
            (1,),
        )

    def _make_all_gather_input(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        placements: list[FlatShard],
        rank: int,
        world_size: int,
        total_numel: int,
        shard_numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        bucket_storage_view = self._try_make_bucket_storage_view(tensors, infos)
        if bucket_storage_view is not None:
            return bucket_storage_view

        valid_numel = self._rank_valid_numel(total_numel, shard_numel, rank)
        send_buf = torch.empty(shard_numel, dtype=dtype, device=device)
        self._copy_local_tensors_into_rank_shard(
            send_buf,
            tensors,
            infos,
            placements,
            rank,
            world_size,
        )
        if valid_numel < shard_numel:
            send_buf[valid_numel:].zero_()
        return send_buf

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Prepare buffers for flat-bucket all-gather."""
        placements = self._flat_shards_from_infos(infos)
        world_size = mesh.size()
        rank = mesh.get_local_rank()
        total_numel = placements[0].total_numel
        shard_numel = self._ceil_div(total_numel, world_size)
        dtype = tensors[0].dtype
        device = tensors[0].device

        with _record_function_if_eager(
            "FlexShard::flat_all_gather_copy_in",
            debug_fqn,
        ):
            self._validate_local_tensor_numels(
                tensors,
                infos,
                placements,
                rank,
                world_size,
            )
            send_buf = self._make_all_gather_input(
                tensors,
                infos,
                placements,
                rank,
                world_size,
                total_numel,
                shard_numel,
                dtype,
                device,
            )
            all_gather_output = torch.empty(
                shard_numel * world_size,
                dtype=dtype,
                device=device,
            )

        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send_buf, all_gather_output],
            placement_state=FlatShard._UnshardState(
                infos=infos,
                total_numel=total_numel,
                shard_numel=shard_numel,
                world_size=world_size,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """Launch the prepared flat-bucket all-gather."""
        if not isinstance(prepared.placement_state, FlatShard._UnshardState):
            raise AssertionError(
                "Expected FlatShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        send_buf = prepared.buffers[0]
        all_gather_output = prepared.buffers[1]
        with _record_comm_if_eager(
            "FlexShard::flat_all_gather",
            prepared.placement_state.debug_fqn,
        ):
            if prepared.placement_state.shard_numel == 0:
                return
            if prepared.placement_state.world_size == 1:
                all_gather_output.copy_(send_buf)
            else:
                dist.all_gather_into_tensor(
                    all_gather_output,
                    send_buf,
                    group=prepared.placement_state.pg,
                )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Finish flat-bucket all-gather and return full parameters."""
        if not isinstance(prepared.placement_state, FlatShard._UnshardState):
            raise AssertionError(
                "Expected FlatShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        with _record_function_if_eager(
            "FlexShard::flat_all_gather_copy_out",
            prepared.placement_state.debug_fqn,
        ):
            flat_full = prepared.buffers[1][: prepared.placement_state.total_numel]
            placements = self._flat_shards_from_infos(prepared.placement_state.infos)

            full_params: list[torch.Tensor] = []
            for info, placement in zip(
                prepared.placement_state.infos,
                placements,
                strict=True,
            ):
                param_end = placement.param_start + placement.param_numel
                full_params.append(
                    flat_full[placement.param_start : param_end].view(info.global_shape)
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
        """Pack full gradients into a padded flat bucket for reduce-scatter."""
        placements = self._flat_shards_from_infos(infos)
        world_size = mesh.size()
        total_numel = placements[0].total_numel
        shard_numel = self._ceil_div(total_numel, world_size)
        dtype = tensors[0].dtype
        device = tensors[0].device

        with _record_function_if_eager(
            "FlexShard::flat_reduce_scatter_copy_in",
            debug_fqn,
        ):
            padded_numel = shard_numel * world_size
            send_buf = torch.empty(
                padded_numel,
                dtype=dtype,
                device=device,
            )
            for tensor, info, placement in zip(
                tensors,
                infos,
                placements,
                strict=True,
            ):
                if tensor.numel() != placement.param_numel:
                    raise ValueError(
                        f"FlatShard expected full gradient for {info.fqn!r} "
                        f"to have {placement.param_numel} elements, but got "
                        f"{tensor.numel()}."
                    )
                param_end = placement.param_start + placement.param_numel
                send_buf[placement.param_start : param_end].copy_(tensor.reshape(-1))
            if total_numel < padded_numel:
                send_buf[total_numel:].zero_()

        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send_buf],
            placement_state=FlatShard._ReduceGradState(
                infos=infos,
                total_numel=total_numel,
                shard_numel=shard_numel,
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
        """Reduce a prepared flat bucket and return local gradient shards."""
        if not isinstance(prepared.placement_state, FlatShard._ReduceGradState):
            raise AssertionError(
                "Expected FlatShard._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        send_buf = prepared.buffers[0]
        recv_buf = torch.empty(
            prepared.placement_state.shard_numel,
            dtype=send_buf.dtype,
            device=send_buf.device,
        )
        with _record_comm_if_eager(
            "FlexShard::flat_reduce_scatter",
            prepared.placement_state.debug_fqn,
        ):
            dist.reduce_scatter_tensor(
                output=recv_buf,
                input=send_buf,
                op=dist.ReduceOp.AVG,
                group=prepared.placement_state.pg,
            )
        with _record_function_if_eager(
            "FlexShard::flat_reduce_scatter_copy_out",
            prepared.placement_state.debug_fqn,
        ):
            placements = self._flat_shards_from_infos(prepared.placement_state.infos)
            sharded_grads: list[torch.Tensor] = []
            for placement in placements:
                local_start, local_end = placement._local_param_bounds(
                    prepared.placement_state.rank,
                    prepared.placement_state.world_size,
                )
                local_numel = local_end - local_start
                if local_numel == 0:
                    sharded_grads.append(recv_buf.new_empty(0))
                    continue
                rank_start, _ = placement._rank_flat_bounds(
                    prepared.placement_state.rank,
                    prepared.placement_state.world_size,
                )
                src_start = placement.param_start + local_start - rank_start
                sharded_grads.append(
                    recv_buf[src_start : src_start + local_numel].contiguous()
                )

        return PlacementReduceGradResult(sharded_grads, [recv_buf])


def flat_shard_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Shard one flat bucket across ranks and expose per-parameter flat slices."""
    total_numel = sum(param.numel() for _, param in named_params)
    bucket_key = tuple(fqn for fqn, _ in named_params)
    placements: dict[str, tuple[Placement, ...]] = {}
    offset = 0
    for fqn, param in named_params:
        placements[fqn] = (
            FlatShard(
                total_numel,
                offset,
                param.numel(),
                bucket_key=bucket_key,
            ),
        )
        offset += param.numel()
    return placements


__all__ = [
    "FlatShard",
    "flat_shard_placements",
]
