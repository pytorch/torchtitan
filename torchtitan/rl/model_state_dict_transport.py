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
import math
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple

import numpy as np
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
# Delta sync compares each push with the preceding one per window of this many
# staging buffer bytes, as 2-byte words: the unit the trainer diffs on its
# device and a generator rank reads and applies.
_DELTA_WINDOW_NBYTES = 64 << 20
# A window sends at most 1 / _DELTA_CAPACITY_DIVISOR of its words; a push with
# more changes in a window is read whole. This bounds the trainer's buffer of
# changes and the delta and receive buffers.
_DELTA_CAPACITY_DIVISOR = 16
# zstd's fastest default level.
_DELTA_ZSTD_LEVEL = 1
# Threads per process that compress (trainer) or decompress (generator) the
# records. zstd releases the GIL.
_DELTA_CODEC_THREADS = 32


class TensorShard(NamedTuple):
    """A generator rank's box of one tensor and the dtype it receives it in."""

    offset: tuple[int, ...]
    """Global offset of the box."""
    shape: tuple[int, ...]
    """Shape of the box, i.e. of the rank's local tensor."""
    dtype: torch.dtype
    """Dtype of the generator rank's local tensor."""


class StagedPiece(NamedTuple):
    """The part of a generator rank's shard one trainer rank holds, and where that
    trainer rank stages it."""

    byte_offset: int
    """Offset in the trainer rank's staging buffer."""
    offset: tuple[int, ...]
    """Global offset of the piece."""
    shape: tuple[int, ...]


class DeltaRecord(NamedTuple):
    """Where one staged piece's record, its changes since the preceding push, is in
    a trainer rank's delta buffer.

    The record is the piece's changed words, as one zstd frame: the gap before each
    (the unchanged words since the previous changed word, or since the piece's
    start) as ``gap_itemsize``-byte unsigned integers, then each word XOR its
    previous value.
    """

    offset: int
    nbytes: int
    gap_itemsize: int
    """2, or 4 if a gap does not fit in 2 bytes."""


@dataclass(frozen=True, slots=True)
class ModelStateDictTransferMetadata:
    """Small control-plane metadata returned by one trainer rank's push."""

    delta_records: dict[int, DeltaRecord] | None = None
    """Delta sync: the record of each staged piece that changed since the preceding
    push, by its ``StagedPiece.byte_offset``. None to read the push whole: full
    sync, the first push, or a push with more changes than a window or the delta
    buffer holds."""
    push: int = 0
    """Index of the push, from 0."""


@dataclass(frozen=True, slots=True)
class TrainerSource:
    """What a generator rank needs to read one trainer rank's staging buffer."""

    pieces: list[dict[str, StagedPiece]]
    """``pieces[rank][key]``: the staged part of generator rank ``rank``'s shard of
    ``key``, for each key this trainer rank holds part of."""
    remote_buffer: Any
    """The staging buffer's ``torch.distributed._transport.RemoteBuffer``."""
    url: bytes


@dataclass(frozen=True, slots=True)
class DeltaTrainerSource(TrainerSource):
    """A ``TrainerSource`` that also stages each push's changes, for delta sync
    (``ModelStateDictDeltaTransport``)."""

    delta_remote_buffer: Any
    """The delta buffer's ``torch.distributed._transport.RemoteBuffer``: the
    records of the latest push."""


class _Read(NamedTuple):
    source: int
    """Index of the trainer rank in the generator's sources."""
    byte_offset: int
    """Offset in the trainer rank's staging buffer."""
    nbytes: int
    """Number of bytes in this contiguous transfer."""
    key: str
    """Key of the generator tensor that receives the transfer."""
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


def _delta_windows(staging_nbytes: int) -> list[tuple[int, int]]:
    """The ``[start, end)`` staging buffer bytes of each delta window."""
    return [
        (start, min(start + _DELTA_WINDOW_NBYTES, staging_nbytes))
        for start in range(0, staging_nbytes, _DELTA_WINDOW_NBYTES)
    ]


def _windows_of(start: int, nbytes: int) -> range:
    """The delta windows that the staging buffer bytes ``[start, start + nbytes)``
    overlap."""
    return range(
        start // _DELTA_WINDOW_NBYTES, -(-(start + nbytes) // _DELTA_WINDOW_NBYTES)
    )


def _delta_capacity(window_nbytes: int) -> int:
    """The most changed words a window of ``window_nbytes`` bytes sends."""
    return -(-window_nbytes // 2 // _DELTA_CAPACITY_DIVISOR)


def _delta_capacity_nbytes(sizes: Iterable[int]) -> int:
    """Bytes reserved for the compressed changes of windows (trainer) or staged
    pieces (generator) of these sizes: 4 per changed word (a 2-byte gap and the
    word)."""
    return sum(4 * _delta_capacity(n) for n in sizes)


def _diff_window(
    new: torch.Tensor,
    old: torch.Tensor,
    *,
    count: torch.Tensor,
    positions_out: torch.Tensor,
    words_out: torch.Tensor,
) -> None:
    """Stage the words of ``new`` that differ from ``old`` in host
    ``positions_out`` (their positions in the window) and ``words_out`` (each word
    XOR its old value), and their number in ``count``, without host
    synchronization. Past ``positions_out.numel()`` changed words (the push is
    then read whole), the outputs are unspecified."""
    changed = new != old
    count.copy_(changed.sum())
    positions = torch.nonzero_static(
        changed, size=positions_out.numel(), fill_value=0
    ).view(-1)
    # Cast on the device: a casting device-to-host copy_ goes through a pageable
    # temporary and casts on the CPU.
    positions_out.copy_(positions.int(), non_blocking=True)
    words_out.copy_(
        new.index_select(0, positions).bitwise_xor_(old.index_select(0, positions)),
        non_blocking=True,
    )


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


def _zstd() -> Any:
    # TODO: zstandard is not a PyTorch library; switch to the standard library's
    # compression.zstd once torchtitan requires Python 3.14.
    import zstandard

    return zstandard


def _unpack_(
    local: torch.Tensor, flat: torch.Tensor, pieces: list[_PackedPiece]
) -> None:
    """Copy each packed piece from ``flat`` into its index in ``local``,
    asynchronously."""
    for p in pieces:
        numel = math.prod(p.shape)
        local[p.index].copy_(
            flat[p.offset : p.offset + numel].view(p.shape), non_blocking=True
        )


def _xor_at_(words: torch.Tensor, positions: torch.Tensor, xors: torch.Tensor) -> None:
    """XOR ``xors`` into int16 ``words`` at ``positions`` in place."""
    words[positions] ^= xors


class _Window(NamedTuple):
    """One delta window of a trainer rank's staging buffer: its ``[lo, hi)``
    bytes; the (key, local index, staging byte offset, bytes) of each staged piece
    in it; and the host buffers its changes are staged in: up to its capacity of
    changed words, the int32 position of each in the window and its int16 XOR."""

    lo: int
    hi: int
    pieces: list[tuple[str, tuple[slice, ...], int, int]]
    positions: torch.Tensor
    xors: torch.Tensor


class _TrainerDelta:
    """A trainer rank's delta sync: compares each push with the staging buffer on
    the shards' device per window, and compresses each staged piece's changes into
    a record in a registered delta buffer.

    Args:
        transport: The trainer rank's transport.
        staging: The staging buffer.
        pieces: (key, index in the local tensor, staging byte offset, bytes) of
            each distinct staged piece.
    """

    def __init__(
        self,
        transport: Any,
        staging: torch.Tensor,
        pieces: list[tuple[str, tuple[slice, ...], int, int]],
    ) -> None:
        self._staging = staging
        self._pieces = sorted(pieces, key=lambda piece: piece[2])
        # An empty staging buffer is still allocated with one byte.
        bounds = _delta_windows(staging.numel() if pieces else 0)
        window_pieces: list[list[tuple[str, tuple[slice, ...], int, int]]] = [
            [] for _ in bounds
        ]
        for piece in self._pieces:
            for w in _windows_of(piece[2], piece[3]):
                window_pieces[w].append(piece)
        self._windows = [
            _Window(
                lo,
                hi,
                window_piece_list,
                _pinned_buffer(4 * _delta_capacity(hi - lo)).view(torch.int32),
                _pinned_buffer(2 * _delta_capacity(hi - lo)).view(torch.int16),
            )
            for (lo, hi), window_piece_list in zip(bounds, window_pieces)
        ]
        self._counts = _pinned_buffer(max(4 * len(bounds), 4)).view(torch.int32)
        self._delta = _pinned_buffer(
            max(_delta_capacity_nbytes(hi - lo for lo, hi in bounds), 1)
        )
        self.delta_buffer = transport.register_memory(self._delta).to_remote_buffer()
        self._num_pushes = 0
        self._executor = ThreadPoolExecutor(
            _DELTA_CODEC_THREADS, thread_name_prefix="model_state_dict_delta"
        )

    def stage(self, local: Callable[[str], torch.Tensor]) -> None:
        """Assemble each window of the staging buffer on the shards' device and,
        after the first push, stage its changes before overwriting it.

        ``local(key)`` is this rank's shard of ``key`` in the dtype it is staged in.
        """
        if not self._windows:
            return
        diff = self._num_pushes > 0
        device = local(self._pieces[0][0]).device
        counts = torch.zeros(len(self._windows), dtype=torch.int32, device=device)
        for w, window in enumerate(self._windows):
            host = self._staging[window.lo : window.hi]
            # The previous push, read back before it is overwritten below; both
            # copies run in stream order. It also fills the padding between pieces.
            old = host.to(device, non_blocking=True)
            new = old.clone()
            for key, index, start, nbytes in window.pieces:
                piece = local(key)[index].contiguous().view(-1).view(torch.uint8)
                a, b = max(start, window.lo), min(start + nbytes, window.hi)
                new[a - window.lo : b - window.lo].copy_(piece[a - start : b - start])
            if diff:
                # Windows hold a multiple of _ALIGNMENT bytes.
                _diff_window(
                    new.view(torch.int16),
                    old.view(torch.int16),
                    count=counts[w : w + 1],
                    positions_out=window.positions,
                    words_out=window.xors,
                )
            host.copy_(new, non_blocking=True)
        if diff:
            self._counts[: counts.numel()].copy_(counts, non_blocking=True)

    def finish(self) -> ModelStateDictTransferMetadata:
        """After the first push, compress each changed piece's record into the delta
        buffer and report where each landed."""
        push = self._num_pushes
        self._num_pushes += 1
        if push == 0 or not self._windows:
            return ModelStateDictTransferMetadata(push=push)
        counts = self._counts[: len(self._windows)].tolist()
        if any(
            count > window.positions.numel()
            for count, window in zip(counts, self._windows)
        ):
            return ModelStateDictTransferMetadata(push=push)

        def compress(
            piece: tuple[str, tuple[slice, ...], int, int],
        ) -> tuple[int, bytes, int] | None:
            _, _, start, nbytes = piece
            positions, xors = [], []
            for w in _windows_of(start, nbytes):
                window = self._windows[w]
                window_positions = window.positions[: counts[w]].numpy()
                # The piece's part of the window, as words of the window.
                lo, hi = np.searchsorted(
                    window_positions,
                    [
                        (max(start, window.lo) - window.lo) // 2,
                        (min(start + nbytes, window.hi) - window.lo) // 2,
                    ],
                )
                positions.append(
                    window_positions[lo:hi].astype(np.int64) + (window.lo - start) // 2
                )
                xors.append(window.xors[lo:hi].numpy())
            positions = np.concatenate(positions)
            if not positions.size:
                return None
            gaps = np.diff(positions, prepend=-1) - 1
            gaps = gaps.astype(np.uint16 if gaps.max() < 1 << 16 else np.uint32)
            record = np.concatenate(
                [gaps.view(np.uint8), *(x.view(np.uint8) for x in xors)]
            )
            data = _zstd().ZstdCompressor(level=_DELTA_ZSTD_LEVEL).compress(record)
            return start, data, gaps.itemsize

        compressed = [
            record
            for record in self._executor.map(compress, self._pieces)
            if record is not None
        ]
        if sum(len(data) for _, data, _ in compressed) > self._delta.numel():
            return ModelStateDictTransferMetadata(push=push)
        delta = self._delta.numpy()
        records = {}
        offset = 0
        for start, data, gap_itemsize in compressed:
            delta[offset : offset + len(data)] = np.frombuffer(data, dtype=np.uint8)
            records[start] = DeltaRecord(offset, len(data), gap_itemsize)
            offset += len(data)
        return ModelStateDictTransferMetadata(records, push=push)


# A record's target (``_DeltaPlan.records``), the record and its offset in the
# receive buffer.
_RecordRead = tuple[tuple[str, int], DeltaRecord, int]


class _DeltaPlan(NamedTuple):
    """How a generator rank reads one trainer rank's records."""

    records: dict[int, tuple[str, int]]
    """For each piece this rank reads, by its ``StagedPiece.byte_offset``: its key
    and word offset in that key's prefetch buffer."""
    delta_buffer: Any
    receive_offset: int
    """Where in the receive buffer this trainer rank's records go."""
    receive_capacity: int


def _decompress_record(
    compressed: np.ndarray, delta_record: DeltaRecord
) -> tuple[np.ndarray, np.ndarray]:
    """Decompress one piece's record into its gaps and XORs."""
    out = np.frombuffer(_zstd().ZstdDecompressor().decompress(compressed), np.uint8)
    split = out.size // (delta_record.gap_itemsize + 2) * delta_record.gap_itemsize
    gaps = out[:split].view(np.uint16 if delta_record.gap_itemsize == 2 else np.uint32)
    return gaps, out[split:].view(np.int16)


@dataclass(slots=True, eq=False)
class _Group:
    """The changes of one load group's keys (the keys one ``copy_prefetched_``
    call of a load gets, e.g. an FSDP group), packed so a load computes their
    positions with one copy to the accelerator and one running sum."""

    gaps: torch.Tensor
    """int32, pinned: every changed piece's gaps, by key."""
    xors: torch.Tensor
    """int16, pinned, in the same order."""
    starts: torch.Tensor
    """int64, pinned: the index in ``gaps`` of each piece's first change."""
    shifts: torch.Tensor
    """int64, pinned: added at each piece's first change, so the running sum of
    (gap + 1) restarts at the piece's offset in its key."""
    keys: dict[str, tuple[int, int]]
    """Each key's ``[start, end)`` changes in ``gaps`` and ``xors``."""
    unloaded: set[str]
    """The keys whose load has not applied the changes yet."""
    on_device: tuple[torch.Tensor, torch.Tensor] | None = None
    """The positions and XORs on the load's device while keys are unloaded."""

    @classmethod
    def pack(
        cls,
        key_changes: list[tuple[str, list[tuple[int, np.ndarray, np.ndarray]]]],
        *,
        pin_memory: bool,
    ) -> _Group:
        """Pack the (word offset in the key's prefetch buffer, gaps, XORs) of each
        changed piece of each key."""
        total = sum(gaps.size for _, changes in key_changes for _, gaps, _ in changes)
        gaps = torch.empty(total, dtype=torch.int32, pin_memory=pin_memory)
        xors = torch.empty(total, dtype=torch.int16, pin_memory=pin_memory)
        gaps_np, xors_np = gaps.numpy(), xors.numpy()
        starts, shifts, keys = [], [], {}
        start = 0
        previous_end = 0  # The running sum before a piece: past the last change.
        for key, changes in key_changes:
            first = start
            for offset, piece_gaps, piece_xors in changes:
                end = start + piece_gaps.size
                gaps_np[start:end] = piece_gaps
                xors_np[start:end] = piece_xors
                starts.append(start)
                shifts.append(offset - previous_end)
                previous_end = (
                    offset + int(piece_gaps.sum(dtype=np.int64)) + (end - start)
                )
                start = end
            keys[key] = (first, start)
        return cls(
            gaps,
            xors,
            torch.tensor(starts, dtype=torch.long, pin_memory=pin_memory),
            torch.tensor(shifts, dtype=torch.long, pin_memory=pin_memory),
            keys,
            set(keys),
        )

    def positions_on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Each change's word position in its key, and its XOR, on ``device``."""
        non_blocking = device.type != "cpu"
        positions = self.gaps.to(device, non_blocking=non_blocking).long().add_(1)
        positions.index_add_(
            0,
            self.starts.to(device, non_blocking=non_blocking),
            self.shifts.to(device, non_blocking=non_blocking),
        )
        return (
            positions.cumsum_(0).sub_(1),
            self.xors.to(device, non_blocking=non_blocking),
        )


class _GeneratorDelta:
    """A generator rank's delta sync: reads and decompresses the records of its
    pieces, applies them to the weights a load copies, and to the prefetch buffers
    after the load.

    Args:
        transport: The generator rank's transport, connected to every source.
        reads: The rank's planned reads (``_plan_reads``).
        sources: Every trainer rank's ``register_trainer_source`` result.
        prefetched_bytes: Each key's prefetch buffer, as flat bytes.
        load_groups: The keys of each ``copy_prefetched_`` call of a load.

    Raises:
        ValueError: if the trainer stages no deltas, or a read is not whole 2-byte
            words, which the changes are.
    """

    def __init__(
        self,
        transport: Any,
        reads: list[_Read],
        sources: list[TrainerSource],
        prefetched_bytes: dict[str, torch.Tensor],
        load_groups: list[list[str]],
    ) -> None:
        if not all(isinstance(source, DeltaTrainerSource) for source in sources):
            raise ValueError("sync_mode='delta', but the trainer stages no deltas")
        targets: list[dict[int, tuple[str, int]]] = [{} for _ in sources]
        piece_nbytes: list[list[int]] = [[] for _ in sources]
        # Each read is one staged piece.
        for read in reads:
            if read.local_byte_offset % 2 or read.nbytes % 2:
                raise ValueError(
                    f"{read.key}: delta sync applies 2-byte words; the piece has "
                    f"{read.nbytes} bytes at prefetch byte {read.local_byte_offset}"
                )
            targets[read.source][read.byte_offset] = (
                read.key,
                read.local_byte_offset // 2,
            )
            piece_nbytes[read.source].append(read.nbytes)
        self._plans = []
        offset = 0
        for source, records, nbytes in zip(sources, targets, piece_nbytes):
            # The trainer's own bound for these records.
            capacity = _delta_capacity_nbytes(nbytes)
            self._plans.append(
                _DeltaPlan(
                    dict(sorted(records.items())),
                    source.delta_remote_buffer,
                    offset,
                    capacity,
                )
            )
            offset += capacity
        receive = _pinned_buffer(max(offset, 1))
        self._receive = (receive, transport.register_memory(receive))
        self._prefetched_bytes = prefetched_bytes
        self._executor = ThreadPoolExecutor(
            _DELTA_CODEC_THREADS, thread_name_prefix="model_state_dict_delta"
        )
        # The load group of each key; the read changes not yet loaded, by key;
        # the loaded groups not yet written into the prefetch buffers; and the
        # background writes.
        self._key_groups = {
            key: i for i, group in enumerate(load_groups) for key in group
        }
        assert self._key_groups.keys() >= prefetched_bytes.keys()
        self._pending: dict[str, _Group] = {}
        self._loaded: list[_Group] = []
        self._writebacks: list[Future] = []

    def plan_pull(
        self, metadata_by_trainer_rank: tuple[ModelStateDictTransferMetadata, ...]
    ) -> tuple[list[tuple[Any, Any]], list[_RecordRead]] | None:
        """The reads of the records this rank applies, one per run of records
        adjacent in a delta buffer, and the (target, record, receive buffer offset)
        of each record; None to read the push whole."""
        memory = self._receive[1]
        reads = []
        records = []
        for plan, metadata in zip(self._plans, metadata_by_trainer_rank):
            needed = [
                (target, metadata.delta_records[byte_offset])
                for byte_offset, target in plan.records.items()
                if byte_offset in metadata.delta_records
            ]
            if (
                sum(delta_record.nbytes for _, delta_record in needed)
                > plan.receive_capacity
            ):
                return None
            local = plan.receive_offset
            runs: list[list[int]] = []  # [remote offset, local offset, bytes]
            for target, delta_record in needed:
                if runs and runs[-1][0] + runs[-1][2] == delta_record.offset:
                    runs[-1][2] += delta_record.nbytes
                else:
                    runs.append([delta_record.offset, local, delta_record.nbytes])
                records.append((target, delta_record, local))
                local += delta_record.nbytes
            reads += [
                (memory.to_mutable_view(lo, n), plan.delta_buffer.slice(remote, n))
                for remote, lo, n in runs
            ]
        return reads, records

    def decode(self, records: list[_RecordRead]) -> None:
        """Decompress the read records on the codec threads, and pack their changes
        by load group for the load."""
        receive = self._receive[0].numpy()
        futures = [
            (
                target,
                self._executor.submit(
                    _decompress_record,
                    receive[offset : offset + delta_record.nbytes],
                    delta_record,
                ),
            )
            for target, delta_record, offset in records
        ]
        changes = defaultdict(list)
        for (key, offset), future in futures:
            changes[key].append((offset, *future.result()))
        groups = defaultdict(list)
        for key, key_changes in changes.items():
            groups[self._key_groups[key]].append((key, key_changes))
        pin_memory = torch.accelerator.is_available()
        packed = self._executor.map(
            lambda group: _Group.pack(group, pin_memory=pin_memory), groups.values()
        )
        self._pending = {key: group for group in packed for key in group.keys}

    def take(
        self, key: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The word positions and XORs of ``key``'s read changes on ``device``, or
        None if it has none, and mark them loaded."""
        group = self._pending.pop(key, None)
        if group is None:
            return None
        if group.on_device is None:
            group.on_device = group.positions_on(device)
        positions, xors = group.on_device
        start, end = group.keys[key]
        group.unloaded.discard(key)
        if not group.unloaded:
            group.on_device = None
            self._loaded.append(group)
        return positions[start:end], xors[start:end]

    def end_load(self) -> None:
        """Once every read change is loaded, start writing them into the prefetch
        buffers in the background; writing them during the load would slow its
        copies."""
        if self._loaded and not self._pending:
            self._writebacks += [
                self._executor.submit(self._write_back, group) for group in self._loaded
            ]
            self._loaded = []

    def write_back_rest(self) -> None:
        """Start writing the changes no load applied (e.g. of a read that was
        never loaded) into the prefetch buffers in the background too."""
        self._writebacks += [
            self._executor.submit(self._write_back, group)
            for group in {*self._loaded, *self._pending.values()}
        ]
        self._loaded = []
        self._pending = {}

    def wait_write_backs(self) -> None:
        """Wait for the background writes into the prefetch buffers."""
        for future in self._writebacks:
            future.result()
        self._writebacks = []

    def _write_back(self, group: _Group) -> None:
        """XOR one group's changes into its keys' prefetch buffers on the CPU
        (PyTorch ops release the GIL; NumPy fancy indexing would hold it)."""
        positions, xors = group.positions_on(torch.device("cpu"))
        for key, (start, end) in group.keys.items():
            _xor_at_(
                self._prefetched_bytes[key].view(torch.int16),
                positions[start:end],
                xors[start:end],
            )


class ModelStateDictTransport(Configurable):
    """Moves the model state dict from the trainer into the generators' prefetched
    model state dicts.

    The trainer and every generator rank each build one. The trainer stages its
    shards, cast to the generators' dtypes, in one registered pinned CPU buffer laid
    out per generator rank; each generator rank reads its pieces from every trainer
    rank directly into its own registered prefetch buffers.

    ``Config.build`` builds a ``ModelStateDictDeltaTransport`` for
    ``sync_mode="delta"``.

    Trainer: ``register_trainer_source`` once, then ``stage`` and ``finish_stage``
    per push.
    Generator: ``connect_generator`` once, then ``read`` per prefetch and
    ``copy_prefetched_`` per load.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        backend: Literal["mooncake", "nixl"] = "nixl"
        """The ``torch.distributed._transport`` backend."""

        sync_mode: Literal["full", "delta"] = "full"
        """Whether every pull reads the full state or only the words changed since
        the preceding push (``ModelStateDictDeltaTransport``)."""

        read_timeout_s: float = 600.0
        """Seconds to wait for a generator rank's reads before raising. A pull
        moves tens of GB per generator rank at most, well under a minute at line
        rate; this fails a stuck transfer instead of hanging the controller."""

        def __post_init__(self) -> None:
            if self.sync_mode not in ("full", "delta"):
                raise ValueError(f"Invalid sync mode: {self.sync_mode}")

        def build(self, **kwargs) -> ModelStateDictTransport:
            """Build the transport for ``sync_mode``."""
            if self.sync_mode == "delta":
                return ModelStateDictDeltaTransport(config=replace(self), **kwargs)
            return Configurable.Config.build(self, **kwargs)

    def __init__(self, config: Config):
        if type(self) is ModelStateDictTransport and config.sync_mode != "full":
            raise ValueError(
                f"sync_mode={config.sync_mode!r}: build the transport with "
                "Config.build()"
            )
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

    def finish_stage(self) -> ModelStateDictTransferMetadata:
        """Finish the push staged by ``stage`` once its copies have completed, and
        return the metadata generators read it with."""
        return ModelStateDictTransferMetadata()

    def connect_generator(
        self,
        prefetched_model_state_dict: dict[str, torch.Tensor],
        *,
        load_groups: list[list[str]],
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
            load_groups: The keys of each ``copy_prefetched_`` call of a load
                (e.g. each FSDP group's), covering every prefetched key. Delta
                sync applies each group's changes at once.
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

    async def read(
        self, metadata_by_trainer_rank: tuple[ModelStateDictTransferMetadata, ...]
    ) -> None:
        """Fill the prefetch buffers from the trainer's staging buffers.

        ``metadata_by_trainer_rank`` is returned by the same trainer ranks whose
        ``TrainerSource`` values were passed to ``connect_generator``.

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


class ModelStateDictDeltaTransport(ModelStateDictTransport):
    """A ``ModelStateDictTransport`` whose pulls after the first read only the
    words that changed since the preceding push, as zstd-compressed gaps and
    XORs, which the trainer stages in a second, smaller buffer
    (``_TrainerDelta``) and generators apply when they load
    (``_GeneratorDelta``). Built by ``Config.build`` for ``sync_mode="delta"``.
    """

    def __init__(self, config: ModelStateDictTransport.Config):
        super().__init__(config)
        # Trainer or generator rank's delta state; the pieces of each packed key,
        # which a load unpacks from a copy with changes; and the last push read.
        self._trainer_delta: _TrainerDelta | None = None
        self._generator_delta: _GeneratorDelta | None = None
        self._delta_packed: dict[str, list[_PackedPiece]] = {}
        self._last_push: int | None = None

    def register_trainer_source(
        self,
        model_state_dict: dict[str, torch.Tensor],
        *,
        generator_rank_layouts: list[dict[str, TensorShard]],
        transfer_dtype: torch.dtype | None,
        preserve_dtype_keys: frozenset[str],
    ) -> TrainerSource:
        """Register the staging buffer, then the delta buffer."""
        source = super().register_trainer_source(
            model_state_dict,
            generator_rank_layouts=generator_rank_layouts,
            transfer_dtype=transfer_dtype,
            preserve_dtype_keys=preserve_dtype_keys,
        )
        staging = self._staging
        assert staging is not None
        # (key, local index, staging byte offset, bytes) of each distinct piece.
        pieces = [
            (key, index, view.data_ptr() - staging.data_ptr(), view.nbytes)
            for key, (_, copies) in self._staging_plan.items()
            for index, view in copies
        ]
        self._trainer_delta = _TrainerDelta(self._transport, staging, pieces)
        return DeltaTrainerSource(
            source.pieces,
            source.remote_buffer,
            source.url,
            self._trainer_delta.delta_buffer,
        )

    def stage(self, model_state_dict: dict[str, torch.Tensor]) -> None:
        """Assemble each window of the staging buffer on the shards' device and,
        after the first push, compare it with the window's previous contents before
        overwriting them, staging the changed words' positions and XORs for
        ``finish_stage``. The copies and the comparison are
        asynchronous, with no host synchronization; synchronize the current stream
        before ``finish_stage`` and before generators read.
        """
        assert self._trainer_delta is not None
        with torch.no_grad():
            # Cast on the device, as ``stage`` does.
            self._trainer_delta.stage(
                lambda key: _local(model_state_dict[key]).to(
                    self._staging_plan[key][0]
                ),
            )

    def finish_stage(self) -> ModelStateDictTransferMetadata:
        """After the first push, compress each changed piece's record into the delta
        buffer and report where each landed."""
        assert self._trainer_delta is not None
        return self._trainer_delta.finish()

    def connect_generator(
        self,
        prefetched_model_state_dict: dict[str, torch.Tensor],
        *,
        load_groups: list[list[str]],
        sources: list[TrainerSource],
        rank: int,
        generator_index: int,
    ) -> None:
        """Plan the reads, then which of each trainer rank's records this rank
        applies."""
        super().connect_generator(
            prefetched_model_state_dict,
            load_groups=load_groups,
            sources=sources,
            rank=rank,
            generator_index=generator_index,
        )
        reads, self._delta_packed = _plan_reads(
            sources,
            shard_layout(prefetched_model_state_dict),
            rank=rank,
            generator_index=generator_index,
        )
        self._generator_delta = _GeneratorDelta(
            self._transport,
            reads,
            sources,
            {
                key: _local(tensor).view(-1).view(torch.uint8)
                for key, tensor in prefetched_model_state_dict.items()
            },
            load_groups,
        )

    async def read(
        self, metadata_by_trainer_rank: tuple[ModelStateDictTransferMetadata, ...]
    ) -> None:
        """If the prefetch buffers hold the preceding push and every trainer rank's
        records fit the receive buffer, read and decompress the records of this
        rank's pieces, which ``copy_prefetched_`` applies; otherwise read every
        piece whole."""
        delta = self._generator_delta
        assert delta is not None, "connect_generator must be called first"
        delta.write_back_rest()
        pushes = {metadata.push for metadata in metadata_by_trainer_rank}
        assert len(pushes) <= 1, f"trainer ranks are on different pushes: {pushes}"
        push = pushes.pop() if pushes else 0
        planned = None
        if (
            self._last_push is not None
            and push == self._last_push + 1
            and all(
                metadata.delta_records is not None
                for metadata in metadata_by_trainer_rank
            )
        ):
            planned = delta.plan_pull(metadata_by_trainer_rank)
        if planned is None:
            # Whole reads overwrite the prefetch buffers: finish writing the last
            # push's changes into them first.
            await asyncio.to_thread(delta.wait_write_backs)
            await super().read(metadata_by_trainer_rank)
        else:
            reads, records = planned
            if reads:
                works = await asyncio.to_thread(
                    lambda: [
                        self._transport.read(local, remote, async_op=True)
                        for local, remote in reads
                    ]
                )
                await wait_all(works, timeout=self._read_timeout_s)
                await asyncio.to_thread(delta.decode, records)
            # Delta reads only write the receive buffer, so the last push's writes
            # overlap them; the load copies from the prefetch buffers.
            await asyncio.to_thread(delta.wait_write_backs)
        self._last_push = push

    def copy_prefetched_(self, state_dict: dict[str, torch.Tensor]) -> Iterable[str]:
        """Copy ``state_dict`` in place from the prefetch buffers, applying the
        changes of a delta ``read``. Returns its keys.

        After a delta ``read``, the prefetch buffers still hold the preceding push:
        each key with changes is copied to ``state_dict``'s device, its changed
        words are XORed in there, and packed keys are unpacked from that copy. Once
        every change is loaded, CPU threads XOR them into the prefetch buffers in
        the background, which the next ``read`` waits for.
        """
        delta = self._generator_delta
        assert delta is not None, "connect_generator must be called first"
        unchanged = {}
        changed = False
        with torch.no_grad():
            for key, tensor in state_dict.items():
                local = _local(tensor)
                change = delta.take(key, local.device)
                if change is None:
                    unchanged[key] = tensor
                    continue
                changed = True
                pieces = self._delta_packed.get(key)
                source = _local(self._prefetched[key])
                if pieces is None and type(local) is torch.Tensor:
                    # The prefetch buffer has the tensor's layout: XOR in place.
                    local.copy_(source, non_blocking=True)
                    _xor_at_(local.view(-1).view(torch.int16), *change)
                    continue
                # Packed, or a tensor subclass (e.g. a quantized parameter): XOR a
                # contiguous copy.
                source = source.to(local.device, non_blocking=True, copy=True)
                _xor_at_(source.view(-1).view(torch.int16), *change)
                if pieces is None:
                    local.copy_(source)
                else:
                    _unpack_(local, source.view(-1), pieces)
        super().copy_prefetched_(unchanged)
        # Finish the copies out of the prefetch buffers before the writes below
        # change them.
        if changed and torch.accelerator.is_available():
            torch.accelerator.current_stream().synchronize()
        delta.end_load()
        return state_dict.keys()
