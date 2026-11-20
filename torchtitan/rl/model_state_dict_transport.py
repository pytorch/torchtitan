# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Moves the trainer's model state dict into the generators' prefetch buffers or
GPU weights with one-sided reads over a ``torch.distributed._transport`` backend.

See ``MODEL_STATE_DICT_TRANSPORT.md``.
"""

from __future__ import annotations

import asyncio
import math
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import torch

# TODO: torch.distributed._transport, torch.cuda._pin_memory_utils,
# torch.distributed.tensor._utils and placement_types._StridedShard are private
# upstream APIs; switch to public ones once they exist (no upstream issue yet).
import torch.cuda._pin_memory_utils as pin_memory_utils
from torch.distributed._transport import new_transport, wait_all
from torch.distributed.tensor import DTensor, Partial
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.config import Configurable

# Byte alignment of each staged piece, so it can be viewed as any dtype.
_ALIGNMENT = 16


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


class _Read(NamedTuple):
    source: int
    """Index of the trainer rank in the generator's sources."""
    byte_offset: int
    """Offset in the trainer rank's staging buffer."""
    nbytes: int
    key: str
    local_byte_offset: int
    """Offset in the generator's prefetch buffer of ``key``."""


class _PackedPiece(NamedTuple):
    """A piece read whole into a packed prefetch buffer, and where it goes in the
    generator rank's shard."""

    index: tuple[slice, ...]
    """The piece's index in the shard."""
    offset: int
    """Element offset in the prefetch buffer."""
    shape: tuple[int, ...]


def _global_offset_and_local_shape(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Global offset and shape of this rank's local part of ``tensor``."""
    if not isinstance(tensor, DTensor):
        return (0,) * tensor.ndim, tuple(tensor.shape)
    placements = tensor.placements
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


_GPU_STAGING_ALIGNMENT = 256
"""Byte alignment of each key in a GPU staging buffer, so every dtype view of it is
aligned."""


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


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
) -> tuple[list[_Read], dict[str, list[_PackedPiece]]]:
    """Plan the reads that fill generator rank ``rank``'s shards from the trainer.

    Each staged piece is read whole, into the key's prefetch buffer at the next
    offset in offset order. If every piece is one contiguous run of the shard,
    that is where it belongs; otherwise the key's buffer is packed and is copied
    into the shard piece by piece on load. E.g. a ``[8, 6]`` shard covered by
    ``[2, 6]`` row pieces is read in place; covered by ``[8, 3]`` column pieces,
    it is packed.

    When several trainer ranks stage the same piece (e.g. HSDP), generator ranks
    are spread across them round-robin, by generator then rank.

    Returns the reads and the pieces of each packed key.

    Raises:
        ValueError: if the trainer does not cover a shard exactly once.
    """
    # Index of this rank across every generator's ranks.
    spread_index = generator_index * len(sources[0].pieces) + rank
    reads = []
    packed = {}
    for key, shard in layout.items():
        # (offset, shape) -> the trainer ranks staging that piece.
        stagers: dict[tuple, list[tuple[int, StagedPiece]]] = {}
        for i, source in enumerate(sources):
            piece = source.pieces[rank].get(key)
            if piece is not None:
                stagers.setdefault((piece.offset, piece.shape), []).append((i, piece))

        _check_disjoint(key, list(stagers))
        pieces = []
        offset = 0
        contiguous = True
        for _, piece_stagers in sorted(stagers.items()):
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
            numel = math.prod(piece.shape)
            reads.append(
                _Read(
                    source,
                    piece.byte_offset,
                    numel * shard.dtype.itemsize,
                    key,
                    offset * shard.dtype.itemsize,
                )
            )
            pieces.append(
                _PackedPiece(
                    tuple(
                        slice(o - so, o - so + n)
                        for o, so, n in zip(piece.offset, shard.offset, piece.shape)
                    ),
                    offset,
                    piece.shape,
                )
            )
            offset += numel
            # Contiguous iff size 1 on every dim before the last one narrower
            # than the shard.
            dim = max(
                (d for d, (n, m) in enumerate(zip(piece.shape, shard.shape)) if n != m),
                default=0,
            )
            contiguous = contiguous and math.prod(piece.shape[:dim]) == 1
        if offset != math.prod(shard.shape):
            raise ValueError(
                f"{key}: the trainer covers {offset} of the generator shard's "
                f"{math.prod(shard.shape)} elements"
            )
        if not contiguous:
            packed[key] = pieces
    return reads, packed


class ModelStateDictTransport(Configurable):
    """Moves the model state dict from the trainer into the generators' prefetched
    model state dicts.

    The trainer and every generator rank each build one. The trainer stages its
    shards, cast to the generators' dtypes, in one registered pinned CPU buffer laid
    out per generator rank; each generator rank reads its pieces from every trainer
    rank directly into its own registered prefetch buffers.

    Trainer: ``register_trainer_source`` once, then ``stage`` per push.
    Generator: ``connect_generator`` once, then ``read`` per prefetch and
    ``copy_prefetched_`` per load; or, reading straight into GPU weights,
    ``connect_generator_gpu`` once, then ``read_into`` per load.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        backend: Literal["mooncake", "nixl"] = "nixl"
        """The ``torch.distributed._transport`` backend."""

        read_timeout_s: float = 600.0
        """Seconds to wait for a generator rank's reads before raising. A pull
        moves tens of GB per generator rank at most, well under a minute at line
        rate; this fails a stuck transfer instead of hanging the controller."""

    def __init__(self, config: Config):
        self._transport = new_transport(config.backend)
        self._read_timeout_s = config.read_timeout_s
        # Trainer: the staging buffer and, per key with pieces on this rank, the
        # dtype it is staged in and a (local index, staging view) per piece.
        self._staging: torch.Tensor | None = None
        self._staging_plan: dict[
            str, tuple[torch.dtype, list[tuple[tuple[slice, ...], torch.Tensor]]]
        ] = {}
        # Generator: (local view, remote buffer) per read, or None until connected;
        # the prefetch buffers; and (shard index, packed view) per piece of each
        # packed key.
        self._reads: list[tuple[Any, Any]] | None = None
        self._prefetched: dict[str, torch.Tensor] = {}
        self._packed: dict[str, list[tuple[tuple[slice, ...], torch.Tensor]]] = {}
        # Generator, GPU fetch: each key's (read, remote buffer) reads, or None until
        # connected; the pieces of each packed key; each load group's keys, in load
        # order, and each key's group and offset in a staging buffer; the two
        # registered staging buffers (tensor, memory, event marking the end of the
        # last copy out of it); and the group being prefetched as (group, staging
        # slot, works).
        self._gpu_reads: dict[str, list[tuple[_Read, Any]]] | None = None
        self._gpu_packed: dict[str, list[_PackedPiece]] = {}
        self._gpu_groups: list[frozenset[str]] = []
        self._gpu_group_of_key: dict[str, int] = {}
        self._gpu_staging_offsets: dict[str, int] = {}
        self._gpu_staging_nbytes = 0
        self._gpu_staging: list[tuple[torch.Tensor, Any, torch.Event | None]] = []
        self._gpu_prefetch: tuple[int, int, list[Any]] | None = None
        self._gpu_next_slot = 0

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

        pieces, nbytes = _staged_pieces(boxes, generator_rank_layouts)
        staging = _pinned_buffer(max(nbytes, 1))
        memory = self._transport.register_memory(staging)
        self._staging_plan = {}
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
                self._staging_plan.setdefault(key, (dtype, []))[1].append((index, view))
        self._staging = staging
        return TrainerSource(pieces, memory.to_remote_buffer(), self._transport.bind())

    def stage(self, model_state_dict: dict[str, torch.Tensor]) -> None:
        """Cast this trainer rank's shards and copy them into the staging buffer.

        The copies are asynchronous; synchronize the current stream before
        generators read.
        """
        with torch.no_grad():
            for key, (dtype, copies) in self._staging_plan.items():
                # Cast on the device: a casting device-to-host copy_ goes through
                # a pageable temporary and casts on the CPU.
                local = _local(model_state_dict[key]).to(dtype)
                for index, view in copies:
                    view.copy_(local[index], non_blocking=True)

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
                ``shard_layout`` the trainer staged for. Packed keys hold
                their pieces out of place; load them with ``copy_prefetched_``.
            sources: Every trainer rank's ``register_trainer_source`` result.
            rank: This rank within its generator.
            generator_index: Index of this rank's generator.
        """
        reads, packed = _plan_reads(
            sources,
            shard_layout(prefetched_model_state_dict),
            rank=rank,
            generator_index=generator_index,
        )
        for source in sources:
            self._transport.connect(source.url)
        locals_ = {
            key: _local(tensor).view(-1)
            for key, tensor in prefetched_model_state_dict.items()
        }
        # Only keys with reads: backends reject registering empty tensors.
        memories = {
            key: self._transport.register_memory(locals_[key])
            for key in {r.key for r in reads}
        }
        self._reads = [
            (
                memories[r.key].to_mutable_view(r.local_byte_offset, r.nbytes),
                sources[r.source].remote_buffer.slice(r.byte_offset, r.nbytes),
            )
            for r in reads
        ]
        self._prefetched = prefetched_model_state_dict
        self._packed = {
            key: [
                (
                    p.index,
                    locals_[key][p.offset : p.offset + math.prod(p.shape)].view(
                        p.shape
                    ),
                )
                for p in pieces
            ]
            for key, pieces in packed.items()
        }

    def connect_generator_gpu(
        self,
        layout: dict[str, TensorShard],
        *,
        load_groups: list[list[str]],
        sources: list[TrainerSource],
        rank: int,
        generator_index: int,
    ) -> None:
        """Connect to every trainer rank and plan reads into two GPU staging buffers,
        each sized for the largest load group.

        While one group is copied out of a staging buffer and loaded, the next group
        of the load is read into the other.

        Args:
            layout: ``shard_layout`` of the tensors loads read into, which the
                trainer staged for.
            load_groups: The keys of each ``read_into`` call of a load, in load
                order.
            sources: Every trainer rank's ``register_trainer_source`` result.
            rank: This rank within its generator.
            generator_index: Index of this rank's generator.

        Raises:
            ValueError: if ``load_groups`` does not partition ``layout``'s keys.
        """
        if sorted(key for group in load_groups for key in group) != sorted(layout):
            raise ValueError("the load groups must partition the layout's keys")
        reads, packed = _plan_reads(
            sources, layout, rank=rank, generator_index=generator_index
        )
        for source in sources:
            self._transport.connect(source.url)
        self._gpu_packed = packed
        self._gpu_groups = [frozenset(group) for group in load_groups]
        self._gpu_group_of_key = {
            key: index for index, group in enumerate(load_groups) for key in group
        }
        self._gpu_reads = {}
        for read in reads:
            self._gpu_reads.setdefault(read.key, []).append(
                (
                    read,
                    sources[read.source].remote_buffer.slice(
                        read.byte_offset, read.nbytes
                    ),
                )
            )
        for group in load_groups:
            offset = 0
            for key in group:
                self._gpu_staging_offsets[key] = offset
                shard = layout[key]
                nbytes = math.prod(shard.shape) * shard.dtype.itemsize
                offset += -(-nbytes // _GPU_STAGING_ALIGNMENT) * _GPU_STAGING_ALIGNMENT
            self._gpu_staging_nbytes = max(self._gpu_staging_nbytes, offset)

    def read_into(self, state_dict: dict[str, torch.Tensor]) -> Iterable[str]:
        """Read one load group's latest trainer values into its GPU tensors. Returns
        its keys.

        The group's reads land in a staging buffer, already in flight if the
        previous call prefetched this group. The next group of the load is then
        read into the other buffer while this group is copied out and loaded.
        """
        assert self._gpu_reads is not None, "connect_generator_gpu must be called first"
        tensors = {key: _local(tensor) for key, tensor in state_dict.items()}
        if not tensors:
            return state_dict.keys()
        if not self._gpu_staging:
            self._allocate_gpu_staging(next(iter(tensors.values())).device)
        group = self._gpu_group_of_key[next(iter(tensors))]
        prefetch = self._gpu_prefetch
        if prefetch is not None and prefetch[0] == group:
            self._gpu_prefetch = None
            _, slot, works = prefetch
        else:
            # E.g. left by a load that failed part way.
            self._finish_gpu_prefetch()
            slot = self._gpu_next_slot
            works = self._read_gpu_group(group, slot)
        for work in works:
            work.wait()
        # Prefetch only within this load: the trainer may restage after it.
        if group + 1 < len(self._gpu_groups):
            self._gpu_prefetch = (
                group + 1,
                1 - slot,
                self._read_gpu_group(group + 1, 1 - slot),
            )
        self._gpu_next_slot = 1 - slot

        buffer, _, event = self._gpu_staging[slot]
        with torch.no_grad():
            for key, tensor in tensors.items():
                start = self._gpu_staging_offsets[key]
                staged = buffer[start : start + tensor.nbytes].view(tensor.dtype)
                pieces = self._gpu_packed.get(key)
                if pieces is None:
                    tensor.copy_(staged.view(tensor.shape), non_blocking=True)
                    continue
                for p in pieces:
                    tensor[p.index].copy_(
                        staged[p.offset : p.offset + math.prod(p.shape)].view(p.shape),
                        non_blocking=True,
                    )
        if event is not None:
            event.record()
        return state_dict.keys()

    def _allocate_gpu_staging(self, device: torch.device) -> None:
        """Allocate and register the two staging buffers on ``device``."""
        for _ in range(2):
            buffer = torch.empty(
                max(self._gpu_staging_nbytes, 1), dtype=torch.uint8, device=device
            )
            event = torch.Event(device=device) if device.type != "cpu" else None
            self._gpu_staging.append(
                (buffer, self._transport.register_memory(buffer), event)
            )

    def _read_gpu_group(self, group: int, slot: int) -> list[Any]:
        """Start reading every key of load group ``group`` into staging buffer ``slot``."""
        assert self._gpu_reads is not None
        _, memory, event = self._gpu_staging[slot]
        if event is not None:
            # Copies of the previous group out of this buffer must finish first.
            event.synchronize()
        works = []
        for key in self._gpu_groups[group]:
            start = self._gpu_staging_offsets[key]
            for read, remote in self._gpu_reads.get(key, ()):
                local = memory.to_mutable_view(
                    start + read.local_byte_offset, read.nbytes
                )
                works.append(
                    self._transport.read(
                        local, remote, async_op=True, timeout=self._read_timeout_s
                    )
                )
        return works

    def _finish_gpu_prefetch(self) -> None:
        """Wait for and drop the prefetched group, if any."""
        if self._gpu_prefetch is not None:
            for work in self._gpu_prefetch[2]:
                work.wait()
            self._gpu_prefetch = None

    async def read(self) -> None:
        """Fill the prefetch buffers from the trainer's staging buffers.

        A timeout leaves the transfers in flight: the prefetch buffers are
        undefined and the job must restart.
        """
        assert self._reads is not None, "connect_generator must be called first"
        if not self._reads:
            return
        works = await asyncio.to_thread(
            lambda: [
                self._transport.read(local, remote, async_op=True)
                for local, remote in self._reads
            ]
        )
        await wait_all(works, timeout=self._read_timeout_s)

    def copy_prefetched_(self, state_dict: dict[str, torch.Tensor]) -> Iterable[str]:
        """Copy ``state_dict`` in place from the prefetch buffers, unpacking packed
        keys. Returns its keys."""
        unpacked = False
        with torch.no_grad():
            for key, tensor in state_dict.items():
                pieces = self._packed.get(key)
                if pieces is None:
                    tensor.copy_(self._prefetched[key])
                    continue
                unpacked = True
                local = _local(tensor)
                for index, view in pieces:
                    local[index].copy_(view, non_blocking=True)
        # Packed copies are async; finish them before the next read reuses
        # the prefetch buffers.
        if unpacked and torch.accelerator.is_available():
            torch.accelerator.current_stream().synchronize()
        return state_dict.keys()
