# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Moves the trainer's model state dict into the generators' prefetch buffers with
one-sided reads over a ``torch.distributed._transport`` backend.

See ``MODEL_STATE_DICT_TRANSPORT.md``.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import weakref
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import torch

# TODO: torch.distributed._transport, torch.cuda._pin_memory_utils,
# torch.distributed.tensor._utils and placement_types._StridedShard are private
# upstream APIs; switch to public ones once they exist (no upstream issue yet).
import torch.cuda._pin_memory_utils as pin_memory_utils
from torch.distributed._transport import new_transport, wait_all
from torch.distributed.tensor import DTensor, Partial, Placement, Shard
from torch.distributed.tensor._utils import (
    _compute_local_shape_and_global_offset,
    compute_local_shape_and_global_offset,
)
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.config import Configurable

logger = logging.getLogger(__name__)

# Byte alignment of each staged piece, so it can be viewed as any dtype.
_ALIGNMENT = 16
# A delta row is an int64 header [count, version], then capacity values, then
# capacity int32 flat indices into the staged piece. count is the number of
# changed elements; the row holds them only if count <= capacity.
# Elements compared at once: bounds the accelerator temporaries of a piece,
# besides a copy of its row, to about 40 bytes per element, ~160 MB, or per
# element of one dim-0 slice of a larger non-contiguous piece.
_CHUNK_NUMEL = 1 << 22
# Elements are compared bitwise, as integers of their size.
_BITS_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class TensorShard(NamedTuple):
    """A generator rank's box of one tensor and the dtype it receives it in."""

    offset: tuple[int, ...]
    """Global offset of the box."""
    shape: tuple[int, ...]
    """Shape of the box, i.e. of the rank's local tensor."""
    dtype: torch.dtype


class StagedPiece(NamedTuple):
    """The part of a generator rank's shard one trainer rank holds, and where that
    trainer rank stages it."""

    byte_offset: int
    """Offset in the trainer rank's staging buffer."""
    offset: tuple[int, ...]
    """Global offset of the piece."""
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainerSource:
    """What a generator rank needs to read one trainer rank's staging buffer."""

    pieces: list[dict[str, StagedPiece]]
    """``pieces[rank][key]``: the staged part of generator rank ``rank``'s shard of
    ``key``, for each key this trainer rank holds part of."""
    remote_buffer: Any
    """The staging buffer's ``torch.distributed._transport.RemoteBuffer``."""
    url: bytes
    delta_rows: dict[int, tuple[int, int]]
    """Staged piece byte offset -> (byte offset, capacity) of its delta row in
    the staging buffer; empty without ``delta_max_fraction``."""


class _Read(NamedTuple):
    source: int
    """Index of the trainer rank in the generator's sources."""
    byte_offset: int
    """Offset in the trainer rank's staging buffer."""
    nbytes: int
    key: str
    local_byte_offset: int
    """Offset in the generator's local tensor of ``key``."""


def _global_offset_and_local_shape(
    tensor: torch.Tensor, placements: tuple[Placement, ...] | None = None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Global offset and shape of this rank's local part of ``tensor``, if placed
    with ``placements`` (default: its own)."""
    if not isinstance(tensor, DTensor):
        return (0,) * tensor.ndim, tuple(tensor.shape)
    placements = tensor.placements if placements is None else placements
    for p in placements:
        if isinstance(p, Partial):
            raise ValueError(f"Partial placements are unsupported: {placements}")
        # An uneven strided shard is not one contiguous box.
        if isinstance(p, _StridedShard) and tensor.shape[p.dim] % p.split_factor:
            raise ValueError(
                f"Uneven {p} of a tensor of shape {tuple(tensor.shape)} is unsupported"
            )
    shape, offset = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, placements
    )
    return tuple(offset), tuple(shape)


def _staged_placements(
    tensor: DTensor, generator_shards: list[TensorShard]
) -> tuple[Placement, ...] | None:
    """The placements the trainer stages ``tensor`` in, or None for its own.

    A piece is read as one contiguous run per index of the dims before the last
    one it does not span. A trainer box that cuts a generator box on a dim after
    the first sharded dim multiplies the runs, so its shards of later dims are
    then moved to the first sharded dim. E.g. a row-parallel weight (FSDP
    ``Shard(0)``, TP ``Shard(1)``) read by generator ranks split on dim 0 is
    staged as ``(Shard(0), Shard(0))``, read at once. If every trainer box
    contains the generator boxes it overlaps on those dims (e.g. generator TP
    on dim 1 matching the trainer's), the restage would not reduce the runs.

    Decided from every trainer rank's box so all trainer ranks agree: a
    redistribute is a collective.
    """
    dims = [p.dim for p in tensor.placements if isinstance(p, Shard)]
    if not dims:
        return None
    first = min(dims)
    staged = tuple(
        Shard(first) if isinstance(p, (Shard, _StridedShard)) and p.dim > first else p
        for p in tensor.placements
    )
    if staged == tensor.placements:
        return None
    # Only the axes sharding later dims change a trainer box on those dims.
    axes = [
        axis
        for axis, p in enumerate(tensor.placements)
        if isinstance(p, (Shard, _StridedShard)) and p.dim > first
    ]
    mesh_shape = tensor.device_mesh.shape
    coordinate = [0] * len(mesh_shape)
    later = range(first + 1, tensor.ndim)
    for index in itertools.product(*(range(mesh_shape[axis]) for axis in axes)):
        for axis, i in zip(axes, index):
            coordinate[axis] = i
        shape, offset = _compute_local_shape_and_global_offset(
            tensor.shape, mesh_shape, coordinate, tensor.placements
        )
        for shard in generator_shards:
            spans = [
                (
                    offset[d],
                    offset[d] + shape[d],
                    shard.offset[d],
                    shard.offset[d] + shard.shape[d],
                )
                for d in later
            ]
            overlaps = all(lo < g_hi and g_lo < hi for lo, hi, g_lo, g_hi in spans)
            contains = all(lo <= g_lo and g_hi <= hi for lo, hi, g_lo, g_hi in spans)
            if overlaps and not contains:
                return staged
    return None


def _pinned_buffer(nbytes: int) -> torch.Tensor:
    """A pinned CPU byte buffer, pageable without an accelerator."""
    if not torch.cuda.is_available():
        return torch.empty(
            nbytes, dtype=torch.uint8, pin_memory=torch.accelerator.is_available()
        )
    # TODO: switch to a device-agnostic API once one exists. Unlike
    # pin_memory=True, this does not round the buffer up to a power of two.
    buffer = torch.empty(nbytes, dtype=torch.uint8)
    pin_memory_utils.pin_memory(buffer.data_ptr(), buffer.nbytes)
    weakref.finalize(buffer, pin_memory_utils.unpin_memory, buffer.data_ptr())
    return buffer


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class _DeltaRow(NamedTuple):
    """Views of one staged piece's delta row."""

    header: torch.Tensor
    values: torch.Tensor
    indices: torch.Tensor


def _delta_row_nbytes(capacity: int, dtype: torch.dtype) -> int:
    nbytes = 16 + capacity * (dtype.itemsize + 4)
    return -(-nbytes // _ALIGNMENT) * _ALIGNMENT


def _delta_row(row: torch.Tensor, capacity: int, dtype: torch.dtype) -> _DeltaRow:
    """View the bytes ``row`` as a delta row."""
    # capacity is a multiple of 8, so the indices are 4-byte aligned.
    indices = 16 + capacity * dtype.itemsize
    return _DeltaRow(
        row[:16].view(torch.int64),
        row[16:indices].view(dtype),
        row[indices : indices + capacity * 4].view(torch.int32),
    )


def _stage_delta(
    piece: torch.Tensor, staged: torch.Tensor, row: _DeltaRow, version: int
) -> None:
    """Write the elements of ``piece`` whose bits differ from ``staged``, which
    holds the previous version, into ``row``; then copy ``piece`` into ``staged``.

    Asynchronous, without host syncs: ``nonzero_static`` bounds the changed
    elements to the row's capacity on the device.
    """
    bits = _BITS_DTYPES[piece.dtype.itemsize]
    capacity = row.values.numel()
    # Slot capacity collects the changes past capacity and is never read.
    values = piece.new_empty(capacity + 1)
    indices = torch.empty(capacity + 1, dtype=torch.int64, device=piece.device)
    count = torch.zeros((), dtype=torch.int64, device=piece.device)
    previous = staged.view(-1)
    # Chunks of whole slices of dim 0, so a non-contiguous piece is flattened
    # one chunk at a time; a contiguous one is chunked by element.
    if piece.dim() == 0 or piece.is_contiguous():
        piece = piece.view(-1, 1)
    row_numel = math.prod(piece.shape[1:])
    rows = max(1, _CHUNK_NUMEL // max(row_numel, 1))
    for first in range(0, piece.shape[0], rows):
        flat = piece[first : first + rows].reshape(-1)
        start = first * row_numel
        old = previous[start : start + flat.numel()].to(flat.device, non_blocking=True)
        changed = flat.view(bits) != old.view(bits)
        size = min(capacity, flat.numel())
        index = torch.nonzero_static(changed, size=size, fill_value=-1).squeeze(1)
        slot = count + torch.arange(size, device=piece.device)
        slot = torch.where((index >= 0) & (slot < capacity), slot, capacity)
        values.scatter_(0, slot, flat[index.clamp(min=0)])
        indices.scatter_(0, slot, index + start)
        count += changed.sum()
    row.values.copy_(values[:capacity], non_blocking=True)
    row.indices.copy_(indices[:capacity].to(torch.int32), non_blocking=True)
    header = torch.stack([count, torch.full_like(count, version)])
    row.header.copy_(header, non_blocking=True)
    # Stream-ordered after the reads of the previous version above.
    staged.view(piece.shape).copy_(piece, non_blocking=True)


class _DeltaPiece(NamedTuple):
    """A generator rank's view of one staged piece's delta row."""

    row: _DeltaRow
    reads: list[int]
    """Indices of the piece's full reads."""
    destination: torch.Tensor
    """The flat local tensor of the piece's key."""
    shape: tuple[int, ...]
    """Shape of the piece."""
    offset: torch.Tensor | None
    """Offset of the piece in the local tensor, or None if the piece is the
    whole local tensor."""
    strides: torch.Tensor
    """Strides of the local tensor."""


def _apply_delta(piece: _DeltaPiece, count: int) -> None:
    """Scatter the first ``count`` changes of ``piece``'s row into its
    destination."""
    _, values, indices = piece.row
    index = indices[:count].long()
    if piece.offset is not None:
        position = torch.stack(torch.unravel_index(index, piece.shape), dim=1)
        index = ((position + piece.offset) * piece.strides).sum(dim=1)
    piece.destination[index] = values[:count]


def shard_layout(state_dict: dict[str, torch.Tensor]) -> dict[str, TensorShard]:
    """The layout of a generator rank's state dict, in its key order."""
    return {
        key: TensorShard(*_global_offset_and_local_shape(tensor), tensor.dtype)
        for key, tensor in state_dict.items()
    }


def shared_rank_layouts(
    generator_layouts: list[list[tuple[int, dict[str, TensorShard]]]],
) -> list[dict[str, TensorShard]]:
    """The ``shard_layout`` of each generator rank, in rank order.

    The trainer stages one copy that every generator reads, so every generator
    must have the same layout.

    Args:
        generator_layouts: Each generator's ``(rank, shard_layout)`` pairs, in any
            order.

    Raises:
        ValueError: if the generators' layouts differ.
    """
    layouts = []
    for pairs in generator_layouts:
        by_rank = dict(pairs)
        layouts.append([by_rank[rank] for rank in range(len(by_rank))])
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError("the generators' model state dict layouts differ")
    return layouts[0]


def _staged_pieces(
    trainer_boxes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    generator_rank_layouts: list[dict[str, TensorShard]],
) -> tuple[list[dict[str, StagedPiece]], int]:
    """Lay out a trainer rank's staging buffer: for each generator rank in order, the
    intersection of each of its boxes with the trainer rank's, in its key order.
    Generator ranks that receive the same piece (e.g. a tensor replicated over TP)
    share one copy.

    Returns the pieces per generator rank and the buffer size in bytes.
    """
    nbytes = 0
    pieces = []
    byte_offsets: dict[tuple, int] = {}
    for layout in generator_rank_layouts:
        rank_pieces = {}
        for key, shard in layout.items():
            if key not in trainer_boxes:
                continue
            trainer_offset, trainer_shape = trainer_boxes[key]
            lo = [max(a, b) for a, b in zip(trainer_offset, shard.offset)]
            hi = [
                min(a + n, b + m)
                for a, n, b, m in zip(
                    trainer_offset, trainer_shape, shard.offset, shard.shape
                )
            ]
            if any(h <= low for low, h in zip(lo, hi)):
                continue
            shape = tuple(h - low for low, h in zip(lo, hi))
            byte_offset = byte_offsets.setdefault((key, tuple(lo), shape), nbytes)
            rank_pieces[key] = StagedPiece(byte_offset, tuple(lo), shape)
            if byte_offset == nbytes:
                size = math.prod(shape) * shard.dtype.itemsize
                nbytes += -(-size // _ALIGNMENT) * _ALIGNMENT
        pieces.append(rank_pieces)
    return pieces, nbytes


def _check_disjoint(
    key: str, boxes: list[tuple[tuple[int, ...], tuple[int, ...]]]
) -> None:
    """Raise if any two ``(offset, shape)`` boxes overlap."""
    boxes = sorted(boxes)
    for i, (offset, shape) in enumerate(boxes):
        for other_offset, other_shape in boxes[i + 1 :]:
            # Sorted by offset: later boxes start past this one on dim 0.
            if other_offset[0] >= offset[0] + shape[0]:
                break
            if all(
                o < so + m and so < o + n
                for o, n, so, m in zip(offset, shape, other_offset, other_shape)
            ):
                raise ValueError(f"{key}: the trainer stages overlapping pieces")


def _plan_reads(
    sources: list[TrainerSource],
    layout: dict[str, TensorShard],
    *,
    rank: int,
    generator_index: int,
) -> list[_Read]:
    """Plan the reads that fill generator rank ``rank``'s shards from the trainer.

    When several trainer ranks stage the same piece (e.g. HSDP), generator ranks
    are spread across them round-robin, by generator then rank.

    Raises:
        ValueError: if the trainer does not cover a shard exactly once.
    """
    # Index of this rank across every generator's ranks.
    spread_index = generator_index * len(sources[0].pieces) + rank
    reads = []
    for key, shard in layout.items():
        # (offset, shape) -> the trainer ranks staging that piece.
        stagers: dict[tuple, list[tuple[int, StagedPiece]]] = {}
        for i, source in enumerate(sources):
            piece = source.pieces[rank].get(key)
            if piece is not None:
                stagers.setdefault((piece.offset, piece.shape), []).append((i, piece))

        _check_disjoint(key, list(stagers))
        numel = 0
        itemsize = shard.dtype.itemsize
        for piece_stagers in stagers.values():
            source, piece = piece_stagers[spread_index % len(piece_stagers)]
            if any(
                o < so or o + n > so + m
                for o, n, so, m in zip(
                    piece.offset, piece.shape, shard.offset, shard.shape
                )
            ):
                raise ValueError(
                    f"{key}: a staged piece is outside the generator shard"
                )
            numel += math.prod(piece.shape)
            # One contiguous run per index of the dims before the last one the
            # piece does not span (see _staged_placements).
            dim = max(
                (d for d, (n, m) in enumerate(zip(piece.shape, shard.shape)) if n != m),
                default=0,
            )
            strides = [math.prod(shard.shape[d + 1 :]) for d in range(len(shard.shape))]
            run = math.prod(piece.shape[dim:])
            for i, index in enumerate(
                itertools.product(*map(range, piece.shape[:dim]))
            ):
                start = sum(
                    (o - so + j) * stride
                    for o, so, j, stride in zip(
                        piece.offset, shard.offset, (*index, 0), strides
                    )
                )
                reads.append(
                    _Read(
                        source,
                        piece.byte_offset + i * run * itemsize,
                        run * itemsize,
                        key,
                        start * itemsize,
                    )
                )
        if numel != math.prod(shard.shape):
            raise ValueError(
                f"{key}: the trainer covers {numel} of the generator shard's "
                f"{math.prod(shard.shape)} elements"
            )
    return reads


class ModelStateDictTransport(Configurable):
    """Moves the model state dict from the trainer into the generators' prefetched
    model state dicts.

    The trainer and every generator rank each build one. The trainer stages its
    shards, cast to the generators' dtypes, in one registered pinned CPU buffer laid
    out per generator rank; each generator rank reads its pieces from every trainer
    rank directly into its own registered prefetch buffers.

    Trainer: ``register_trainer_source`` once, then ``stage`` per push.
    Generator: ``connect_generator`` once, then ``read`` per prefetch.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        backend: Literal["mooncake", "nixl"] = "mooncake"
        """The ``torch.distributed._transport`` backend."""

        read_timeout_s: float = 600.0
        """Seconds to wait for a generator rank's reads before raising. A pull
        moves tens of GB per generator rank at most, well under a minute at line
        rate; this fails a stuck transfer instead of hanging the controller."""

        delta_max_fraction: float | None = None
        """If set, a generator rank holding the previous version reads only the
        elements whose bits changed, for each staged piece where at most this
        fraction changed, and the whole piece otherwise. The generator logs the
        fraction each pull changes; set this just above it. A pull then reads
        about this fraction times ``1 + 4 / itemsize`` of the model state dict,
        plus the pieces that changed more. Costs pinned memory of the same size
        on both sides, and the trainer copies the previous version back to the
        device each push to compare. None reads every piece in full."""

        def __post_init__(self):
            if self.delta_max_fraction is not None and not (
                0 < self.delta_max_fraction < 1
            ):
                raise ValueError(
                    "delta_max_fraction must be in (0, 1), got "
                    f"{self.delta_max_fraction}"
                )

    def __init__(self, config: Config):
        self._transport = new_transport(config.backend)
        self._read_timeout_s = config.read_timeout_s
        self._delta_max_fraction = config.delta_max_fraction
        # Trainer: the staging buffer and, per key in model state dict order, the
        # dtype and placements it is staged in (None to keep its own) and a
        # (local index, staging view, delta row or None) per piece.
        self._staging: torch.Tensor | None = None
        self._staging_plan: dict[
            str,
            tuple[
                torch.dtype,
                tuple[Placement, ...] | None,
                list[tuple[tuple[slice, ...], torch.Tensor, _DeltaRow | None]],
            ],
        ] = {}
        # Trainer: the number of pushes staged.
        self._version = 0
        # Generator: (local view, remote buffer) per read, or None until connected.
        self._reads: list[tuple[Any, Any]] | None = None
        # Generator, with deltas: (local view, remote buffer) per delta row read,
        # the piece of each, and the version the prefetch buffers hold.
        self._row_reads: list[tuple[Any, Any]] = []
        self._delta_pieces: list[_DeltaPiece] = []
        self._version_held: int | None = None

    def register_trainer_source(
        self,
        model_state_dict: dict[str, torch.Tensor],
        *,
        generator_rank_layouts: list[dict[str, TensorShard]],
        transfer_dtype: torch.dtype | None,
        preserve_dtype_keys: frozenset[str],
    ) -> TrainerSource:
        """Allocate and register this trainer rank's staging buffer for the generator
        ranks' layouts, which every generator shares.

        Args:
            model_state_dict: This trainer rank's model state dict.
            generator_rank_layouts: ``shard_layout`` of each rank of a generator,
                in rank order.
            transfer_dtype: The dtype parameters are sent in, or None for their own.
            preserve_dtype_keys: Keys sent in their own dtype.

        Raises:
            ValueError: if a generator receives a key in a dtype other than the
                trainer sends it in.
        """
        dtypes = {}
        placements = {}
        boxes = {}
        for key, tensor in model_state_dict.items():
            if not any(key in layout for layout in generator_rank_layouts):
                continue
            dtypes[key] = (
                tensor.dtype
                if transfer_dtype is None or key in preserve_dtype_keys
                else transfer_dtype
            )
            for layout in generator_rank_layouts:
                if key in layout and layout[key].dtype != dtypes[key]:
                    raise ValueError(
                        f"{key}: the trainer sends {dtypes[key]}, the generator "
                        f"receives {layout[key].dtype}"
                    )
            boxes[key] = _global_offset_and_local_shape(tensor)
            if not isinstance(tensor, DTensor):
                continue
            staged = _staged_placements(
                tensor,
                [layout[key] for layout in generator_rank_layouts if key in layout],
            )
            if staged is not None:
                placements[key] = staged
                boxes[key] = _global_offset_and_local_shape(tensor, staged)

        pieces, nbytes = _staged_pieces(boxes, generator_rank_layouts)
        # One delta row per staged piece, after the pieces.
        delta_rows = {}
        for rank_pieces in pieces if self._delta_max_fraction is not None else ():
            for key, piece in rank_pieces.items():
                if piece.byte_offset in delta_rows:
                    continue
                numel = math.prod(piece.shape)
                if numel > torch.iinfo(torch.int32).max:
                    raise ValueError(
                        f"{key}: a staged piece of {numel} elements is too large "
                        "for int32 delta indices"
                    )
                if dtypes[key].itemsize not in _BITS_DTYPES:
                    raise ValueError(f"{key}: deltas do not support {dtypes[key]}")
                capacity = 8 * math.ceil(self._delta_max_fraction * numel / 8)
                delta_rows[piece.byte_offset] = (nbytes, capacity)
                nbytes += _delta_row_nbytes(capacity, dtypes[key])
        staging = _pinned_buffer(max(nbytes, 1))
        memory = self._transport.register_memory(staging)
        self._staging_plan = {
            key: (dtype, placements.get(key), []) for key, dtype in dtypes.items()
        }
        copied = set()
        for rank_pieces in pieces:
            for key, piece in rank_pieces.items():
                if piece.byte_offset in copied:
                    continue
                copied.add(piece.byte_offset)
                dtype = dtypes[key]
                start = piece.byte_offset
                end = start + math.prod(piece.shape) * dtype.itemsize
                index = tuple(
                    slice(o - to, o - to + n)
                    for o, to, n in zip(piece.offset, boxes[key][0], piece.shape)
                )
                view = staging[start:end].view(dtype).view(piece.shape)
                row = None
                if start in delta_rows:
                    row_start, capacity = delta_rows[start]
                    row_end = row_start + _delta_row_nbytes(capacity, dtype)
                    row = _delta_row(staging[row_start:row_end], capacity, dtype)
                self._staging_plan[key][2].append((index, view, row))
        # Keep restaged keys even without pieces on this rank: a redistribute is
        # a collective.
        self._staging_plan = {
            key: plan
            for key, plan in self._staging_plan.items()
            if plan[1] is not None or plan[2]
        }
        self._staging = staging
        return TrainerSource(
            pieces, memory.to_remote_buffer(), self._transport.bind(), delta_rows
        )

    def stage(self, model_state_dict: dict[str, torch.Tensor]) -> None:
        """Cast and reshard this trainer rank's shards and copy them into the
        staging buffer.

        A collective over the mesh of each restaged DTensor: every trainer rank
        must call it, in the same order relative to its other collectives. The
        copies are asynchronous; synchronize the current stream before generators
        read.

        With ``delta_max_fraction``, also writes each piece's changes since the
        previous ``stage`` into its delta row.
        """
        self._version += 1
        with torch.no_grad():
            for key, (dtype, placements, copies) in self._staging_plan.items():
                # Cast on the device: a casting device-to-host copy_ goes through
                # a pageable temporary and casts on the CPU.
                tensor = model_state_dict[key].to(dtype)
                if placements is not None:
                    tensor = tensor.redistribute(placements=placements)
                local = _local(tensor)
                for index, view, row in copies:
                    if row is None:
                        view.copy_(local[index], non_blocking=True)
                    else:
                        _stage_delta(local[index], view, row, self._version)

    def connect_generator(
        self,
        prefetched_model_state_dict: dict[str, torch.Tensor],
        *,
        sources: list[TrainerSource],
        rank: int,
        generator_index: int,
    ) -> None:
        """Register this generator rank's prefetch buffers, connect to every trainer
        rank and plan the reads.

        Args:
            prefetched_model_state_dict: The buffers ``read`` fills, whose
                ``shard_layout`` the trainer staged for.
            sources: Every trainer rank's ``register_trainer_source`` result.
            rank: This rank within its generator.
            generator_index: Index of this rank's generator.
        """
        layout = shard_layout(prefetched_model_state_dict)
        reads = _plan_reads(sources, layout, rank=rank, generator_index=generator_index)
        for source in sources:
            self._transport.connect(source.url)
        # Only keys with reads: backends reject registering empty tensors.
        memories = {
            key: self._transport.register_memory(
                _local(prefetched_model_state_dict[key]).view(-1)
            )
            for key in {r.key for r in reads}
        }
        self._reads = [
            (
                memories[r.key].to_mutable_view(r.local_byte_offset, r.nbytes),
                sources[r.source].remote_buffer.slice(r.byte_offset, r.nbytes),
            )
            for r in reads
        ]
        if self._delta_max_fraction is None or not reads:
            return
        # The reads of each (trainer rank, key) fill one staged piece.
        piece_reads: dict[tuple[int, str], list[int]] = {}
        for i, r in enumerate(reads):
            piece_reads.setdefault((r.source, r.key), []).append(i)
        rows = []
        nbytes = 0
        for source, key in piece_reads:
            piece = sources[source].pieces[rank][key]
            if piece.byte_offset not in sources[source].delta_rows:
                raise ValueError("the trainer stages no delta rows")
            row_offset, capacity = sources[source].delta_rows[piece.byte_offset]
            row_nbytes = _delta_row_nbytes(capacity, layout[key].dtype)
            rows.append((nbytes, row_offset, row_nbytes, capacity))
            nbytes += row_nbytes
        row_buffer = _pinned_buffer(nbytes)
        row_memory = self._transport.register_memory(row_buffer)
        self._row_reads, self._delta_pieces = [], []
        for ((source, key), indices), (start, row_offset, row_nbytes, capacity) in zip(
            piece_reads.items(), rows, strict=True
        ):
            self._row_reads.append(
                (
                    row_memory.to_mutable_view(start, row_nbytes),
                    sources[source].remote_buffer.slice(row_offset, row_nbytes),
                )
            )
            shard = layout[key]
            piece = sources[source].pieces[rank][key]
            whole = piece.offset == shard.offset and piece.shape == shard.shape
            self._delta_pieces.append(
                _DeltaPiece(
                    row=_delta_row(
                        row_buffer[start : start + row_nbytes], capacity, shard.dtype
                    ),
                    reads=indices,
                    destination=_local(prefetched_model_state_dict[key]).view(-1),
                    shape=piece.shape,
                    offset=None
                    if whole
                    else torch.tensor(
                        [o - so for o, so in zip(piece.offset, shard.offset)]
                    ),
                    strides=torch.tensor(
                        [
                            math.prod(shard.shape[d + 1 :])
                            for d in range(len(shard.shape))
                        ]
                    ),
                )
            )

    async def read(self) -> None:
        """Fill the prefetch buffers from the trainer's staging buffers.

        With ``delta_max_fraction``, reads the delta rows first, applies those on
        top of the version the prefetch buffers hold, and reads in full the
        pieces it cannot apply.

        A timeout leaves the transfers in flight: the prefetch buffers are
        undefined and the job must restart.
        """
        assert self._reads is not None, "connect_generator must be called first"
        if not self._delta_pieces:
            await self._issue(self._reads)
            return
        # Unknown until this read completes.
        held, self._version_held = self._version_held, None
        await self._issue(self._row_reads)
        version, full_reads = await asyncio.to_thread(self._apply_deltas, held)
        await self._issue(full_reads)
        self._version_held = version

    async def _issue(self, reads: list[tuple[Any, Any]]) -> None:
        """Issue ``reads`` and wait for them."""
        if not reads:
            return
        works = await asyncio.to_thread(
            lambda: [
                self._transport.read(local, remote, async_op=True)
                for local, remote in reads
            ]
        )
        await wait_all(works, timeout=self._read_timeout_s)

    def _apply_deltas(self, held: int | None) -> tuple[int, list[tuple[Any, Any]]]:
        """Apply the delta rows read on top of version ``held``.

        Returns the version the rows were staged at, and the full reads of the
        pieces whose rows overflowed or that do not hold the previous version.
        """
        headers = [piece.row.header.tolist() for piece in self._delta_pieces]
        versions = {version for _, version in headers}
        if len(versions) != 1:
            raise RuntimeError(f"delta rows of different versions: {versions}")
        [version] = versions
        full_reads = []
        changed = numel = full = 0
        largest = 0.0
        for piece, (count, _) in zip(self._delta_pieces, headers, strict=True):
            changed += count
            numel += math.prod(piece.shape)
            largest = max(largest, count / max(math.prod(piece.shape), 1))
            if held != version - 1 or count > piece.row.values.numel():
                full_reads.extend(self._reads[i] for i in piece.reads)
                full += 1
            elif count:
                _apply_delta(piece, count)
        logger.info(
            "Model state dict version %d: %.4f of the elements changed, at most "
            "%.4f of a piece's; %d of %d pieces read in full",
            version,
            changed / max(numel, 1),
            largest,
            full,
            len(self._delta_pieces),
        )
        return version, full_reads
