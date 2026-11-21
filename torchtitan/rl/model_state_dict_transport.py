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
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import timedelta
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
# windows. zstd releases the GIL.
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
    """Where one window's record, its changes since the preceding push, is in a
    trainer rank's delta buffer, and how to split it.

    The window's changed words are one zstd-compressed record: the gap before
    each as ``gap_itemsize``-byte unsigned integers, then each word XOR its
    previous value. The words are grouped by the window's segments
    (``DeltaTrainerSource.delta_segments``), and a gap counts the unchanged words
    since the previous changed word of its segment, or since the segment's start.
    """

    offset: int
    nbytes: int
    """Compressed size; 0 if the window did not change."""
    gap_itemsize: int
    """2, or 4 if a gap does not fit in 2 bytes."""
    segment_counts: tuple[int, ...]
    """The number of changed words in each segment of the window."""


@dataclass(frozen=True, slots=True)
class ModelStateDictTransferMetadata:
    """Small control-plane metadata returned by one trainer rank's push."""

    delta_records: tuple[DeltaRecord, ...]
    """Delta sync: for each window of the trainer rank's staging buffer
    (``_delta_windows``), where its record is in its delta buffer. Empty to read
    the push whole: full sync, the first push, or a
    push with more changes than a window or the delta buffer holds."""
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
    compressed changes of the latest push."""
    staging_nbytes: int
    """The staging buffer's size, which sets its windows (``_delta_windows``)."""
    delta_segments: list[list[tuple[int, int]]]
    """For each window, the ``[start, end)`` staging buffer bytes of each staged
    piece in it, by offset: the units its changes are addressed in, so a generator
    rank applies them to its tensors without translating."""


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


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def _delta_windows(staging_nbytes: int) -> list[tuple[int, int]]:
    """The ``[start, end)`` staging buffer bytes of each delta window."""
    return [
        (start, min(start + _DELTA_WINDOW_NBYTES, staging_nbytes))
        for start in range(0, staging_nbytes, _DELTA_WINDOW_NBYTES)
    ]


def _delta_capacity(window_nbytes: int) -> int:
    """The most changed words a window of ``window_nbytes`` bytes sends."""
    return -(-window_nbytes // 2 // _DELTA_CAPACITY_DIVISOR)


def _delta_capacity_nbytes(window_nbytes: Iterable[int]) -> int:
    """Bytes reserved for the compressed changes of these windows: 4 per changed
    word (a 2-byte gap and the word), aligned."""
    nbytes = sum(4 * _delta_capacity(n) for n in window_nbytes)
    return max(_round_up(nbytes, _ALIGNMENT), _ALIGNMENT)


def _diff_window(
    new: torch.Tensor,
    old: torch.Tensor,
    *,
    segment_starts: torch.Tensor,
    count: torch.Tensor,
    gaps_out: torch.Tensor,
    words_out: torch.Tensor,
    segment_counts_out: torch.Tensor,
) -> None:
    """Stage the words of ``new`` that differ from ``old`` in host ``gaps_out``
    (the unchanged words since the previous changed word of the same segment, or
    since the segment's start; segments start at word ``segment_starts``) and
    ``words_out`` (each word XOR its old value), their number in ``count`` and
    per segment in ``segment_counts_out``, without host synchronization. Past
    ``gaps_out.numel()`` changed words (the push is then read whole), the outputs
    are unspecified."""
    capacity = gaps_out.numel()
    changed = new != old
    num_changed = changed.sum()
    count.copy_(num_changed)
    positions = torch.nonzero_static(changed, size=capacity, fill_value=0).view(-1)
    # Changes lie in staged pieces, at or past the first segment's start; the
    # padding entries past the changes are clamped into the first segment.
    segments = (torch.searchsorted(segment_starts, positions, right=True) - 1).clamp_(
        min=0
    )
    first = torch.ones_like(segments, dtype=torch.bool)
    first[1:] = segments[1:] != segments[:-1]
    previous = torch.cat([positions.new_full((1,), -1), positions[:-1]])
    gaps = torch.where(
        first, positions - segment_starts[segments], positions - previous - 1
    )
    # Cast on the device: a casting device-to-host copy_ goes through a pageable
    # temporary and casts on the CPU.
    gaps_out.copy_(gaps.int(), non_blocking=True)
    words_out.copy_(
        new.index_select(0, positions).bitwise_xor_(old.index_select(0, positions)),
        non_blocking=True,
    )
    valid = (torch.arange(capacity, device=new.device) < num_changed).int()
    segment_counts = torch.zeros(
        segment_starts.numel(), dtype=torch.int32, device=new.device
    )
    segment_counts_out.copy_(
        segment_counts.scatter_add_(0, segments, valid), non_blocking=True
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


def _new_codec_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        _DELTA_CODEC_THREADS, thread_name_prefix="model_state_dict_delta"
    )


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
    words.index_copy_(0, positions, words.index_select(0, positions).bitwise_xor_(xors))


@dataclass(slots=True)
class _Window:
    """One delta window of a trainer rank's staging buffer: its ``[lo, hi)``
    bytes; the (key, local index, staging byte offset, bytes) of each staged piece
    in it, by offset, which are its segments; each segment's first word in the
    window (on the shards' device once staged); and the host buffers its changes
    are staged in: up to its capacity of changed words, the int32 gap before each
    and its int16 XOR, and the int32 number of them per segment."""

    lo: int
    hi: int
    pieces: list[tuple[str, tuple[slice, ...], int, int]]
    segment_starts: torch.Tensor
    gaps: torch.Tensor
    xors: torch.Tensor
    segment_counts: torch.Tensor


class _TrainerDelta:
    """A trainer rank's delta sync: compares each push with the staging buffer on
    the shards' device and compresses the changes into a registered delta buffer.

    Args:
        transport: The trainer rank's transport.
        staging: The staging buffer.
        nbytes: Its size.
        pieces: (key, index in the local tensor, staging byte offset, bytes) of
            each distinct staged piece.
    """

    def __init__(
        self,
        transport: Any,
        staging: torch.Tensor,
        nbytes: int,
        pieces: list[tuple[str, tuple[slice, ...], int, int]],
    ) -> None:
        self._staging = staging
        bounds = _delta_windows(nbytes)
        window_pieces: list[list[tuple[str, tuple[slice, ...], int, int]]] = [
            [] for _ in bounds
        ]
        for piece in sorted(pieces, key=lambda piece: piece[2]):
            _, _, start, piece_nbytes = piece
            for w in range(
                start // _DELTA_WINDOW_NBYTES,
                -(-(start + piece_nbytes) // _DELTA_WINDOW_NBYTES),
            ):
                window_pieces[w].append(piece)
        self.segments = [
            [(max(start, lo), min(start + n, hi)) for _, _, start, n in pieces]
            for pieces, (lo, hi) in zip(window_pieces, bounds)
        ]
        # Per window: an int32 count per segment, capacity int32 gaps, then
        # capacity int16 XORs, the int32 parts 4-byte aligned.
        sizes = [
            _round_up(4 * len(segments) + 6 * _delta_capacity(hi - lo), 8)
            for segments, (lo, hi) in zip(self.segments, bounds)
        ]
        found = _pinned_buffer(max(sum(sizes), 1))
        self._windows = []
        o = 0
        for (lo, hi), pieces, segments, size in zip(
            bounds, window_pieces, self.segments, sizes
        ):
            capacity = _delta_capacity(hi - lo)
            g = o + 4 * len(segments)
            self._windows.append(
                _Window(
                    lo,
                    hi,
                    pieces,
                    torch.tensor(
                        [(a - lo) // 2 for a, _ in segments], dtype=torch.long
                    ),
                    found[g : g + 4 * capacity].view(torch.int32),
                    found[g + 4 * capacity : g + 6 * capacity].view(torch.int16),
                    found[o:g].view(torch.int32),
                )
            )
            o += size
        self._counts = _pinned_buffer(max(4 * len(bounds), 4)).view(torch.int32)
        self._delta = _pinned_buffer(
            _delta_capacity_nbytes(hi - lo for lo, hi in bounds)
        )
        self.delta_buffer = transport.register_memory(self._delta).to_remote_buffer()
        self._num_pushes = 0
        self._executor = _new_codec_executor()

    def stage(
        self,
        model_state_dict: dict[str, torch.Tensor],
        local: Callable[[str], torch.Tensor],
    ) -> None:
        """Assemble each window of the staging buffer on the shards' device and,
        after the first push, stage its changes before overwriting it.

        ``local(key)`` is this rank's shard of ``key`` in the dtype it is staged in.
        """
        if not self._windows:
            return
        diff = self._num_pushes > 0
        device = _local(next(iter(model_state_dict.values()))).device
        counts = torch.zeros(len(self._windows), dtype=torch.int32, device=device)
        for w, window in enumerate(self._windows):
            window.segment_starts = window.segment_starts.to(device)
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
                    segment_starts=window.segment_starts,
                    count=counts[w : w + 1],
                    gaps_out=window.gaps,
                    words_out=window.xors,
                    segment_counts_out=window.segment_counts,
                )
            host.copy_(new, non_blocking=True)
        if diff:
            self._counts[: counts.numel()].copy_(counts, non_blocking=True)

    def finish(self) -> ModelStateDictTransferMetadata:
        """After the first push, compress each changed window's record into the
        delta buffer and report where each landed."""
        push = self._num_pushes
        self._num_pushes += 1
        if push == 0 or not self._windows:
            return ModelStateDictTransferMetadata((), push=push)
        counts = self._counts[: len(self._windows)].tolist()
        changed = [w for w, count in enumerate(counts) if count]
        if any(counts[w] > self._windows[w].gaps.numel() for w in changed):
            return ModelStateDictTransferMetadata((), push=push)

        def compress(w: int) -> tuple[bytes, int]:
            window = self._windows[w]
            gaps = window.gaps[: counts[w]].numpy()
            # Gaps are below the window's word count: int32 bits are uint32 ones.
            gaps = (
                gaps.astype(np.uint16) if gaps.max() < 1 << 16 else gaps.view(np.uint32)
            )
            compressor = _zstd().ZstdCompressor(level=_DELTA_ZSTD_LEVEL).compressobj()
            data = compressor.compress(gaps)
            data += compressor.compress(window.xors[: counts[w]].numpy())
            return data + compressor.flush(), gaps.itemsize

        compressed = list(self._executor.map(compress, changed))
        if sum(len(data) for data, _ in compressed) > self._delta.numel():
            return ModelStateDictTransferMetadata((), push=push)
        delta = self._delta.numpy()
        table = [
            DeltaRecord(0, 0, 2, (0,) * window.segment_counts.numel())
            for window in self._windows
        ]
        offset = 0
        for w, (data, gap_itemsize) in zip(changed, compressed):
            delta[offset : offset + len(data)] = np.frombuffer(data, dtype=np.uint8)
            table[w] = DeltaRecord(
                offset,
                len(data),
                gap_itemsize,
                tuple(self._windows[w].segment_counts.tolist()),
            )
            offset += len(data)
        return ModelStateDictTransferMetadata(tuple(table), push=push)


# A window's segments (``_DeltaPlan.windows``), its changes and their offset in the
# receive buffer.
_WindowApply = tuple[list[tuple[str, int] | None], DeltaRecord, int]


class _DeltaPlan(NamedTuple):
    """How a generator rank reads one trainer rank's changes."""

    windows: dict[int, list[tuple[str, int] | None]]
    """For each window this rank's pieces touch: per segment, the key and word
    offset in its prefetch buffer if this rank reads it, else None."""
    delta_buffer: Any
    receive_offset: int
    """Where in the receive buffer this trainer rank's windows go."""
    receive_capacity: int


class _KeyChanges(NamedTuple):
    """The changes of one segment, in one key's prefetch buffer."""

    offset: int
    """Word offset of the segment in the key's prefetch buffer."""
    gaps: np.ndarray
    """uint16 or int32."""
    xors: np.ndarray
    span: int
    """The sum of (gap + 1): the word just past the segment's last change, from
    its start."""


def _decompress_window(
    compressed: np.ndarray,
    delta_record: DeltaRecord,
    segments: list[tuple[str, int] | None],
) -> list[tuple[str, _KeyChanges]]:
    """Decompress one window's record and slice it by segment: (key, changes) for
    each segment this rank reads that changed."""
    num_changed = sum(delta_record.segment_counts)
    out = np.empty(num_changed * (delta_record.gap_itemsize + 2), dtype=np.uint8)
    reader = _zstd().ZstdDecompressor().stream_reader(compressed)
    start = 0
    while start < out.size:
        n = reader.readinto(out[start:])
        if n == 0:
            raise RuntimeError("a delta window's record is truncated")
        start += n
    split = num_changed * delta_record.gap_itemsize
    gaps = out[:split].view(np.uint16 if delta_record.gap_itemsize == 2 else np.int32)
    xors = out[split:].view(np.int16)
    changes = []
    start = 0
    for count, target in zip(delta_record.segment_counts, segments, strict=True):
        if count and target is not None:
            key, offset = target
            end = start + count
            span = int(gaps[start:end].sum(dtype=np.int64)) + count
            changes.append(
                (key, _KeyChanges(offset, gaps[start:end], xors[start:end], span))
            )
        start += count
    return changes


@dataclass(slots=True)
class _Group:
    """The changes of one load group's keys (the keys one ``copy_prefetched_``
    call of a load gets, e.g. an FSDP group), packed so a load computes their
    positions with one copy to the accelerator and one running sum."""

    gaps: torch.Tensor
    """int32, pinned: every segment's gaps, by key."""
    xors: torch.Tensor
    """int16, pinned, in the same order."""
    starts: torch.Tensor
    """int64, pinned: the index in ``gaps`` of each segment's first change."""
    shifts: torch.Tensor
    """int64, pinned: added at each segment's first change, so the running sum of
    (gap + 1) restarts at the segment's offset in its key."""
    keys: dict[str, tuple[int, int]]
    """Each key's ``[start, end)`` changes in ``gaps`` and ``xors``."""
    unloaded: set[str]
    """The keys whose load has not applied the changes yet."""
    positions: tuple[torch.Tensor, torch.Tensor] | None = None
    """The positions and XORs on the load's device while keys are unloaded."""

    @classmethod
    def pack(
        cls, key_changes: list[tuple[str, list[_KeyChanges]]], *, pin_memory: bool
    ) -> _Group:
        total = sum(c.gaps.size for _, changes in key_changes for c in changes)
        gaps = torch.empty(total, dtype=torch.int32, pin_memory=pin_memory)
        xors = torch.empty(total, dtype=torch.int16, pin_memory=pin_memory)
        gaps_np, xors_np = gaps.numpy(), xors.numpy()
        starts, shifts, keys = [], [], {}
        start = 0
        previous_end = 0  # The running sum before a segment: past the last change.
        for key, changes in key_changes:
            first = start
            for c in changes:
                end = start + c.gaps.size
                gaps_np[start:end] = c.gaps
                xors_np[start:end] = c.xors
                starts.append(start)
                shifts.append(c.offset - previous_end)
                previous_end = c.offset + c.span
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
    """A generator rank's delta sync: reads and decompresses the changes of the
    windows covering its pieces, applies them to the weights a load copies, and
    to the prefetch buffers after the load.

    Args:
        transport: The generator rank's transport, connected to every source.
        reads: The rank's planned reads (``_plan_reads``).
        sources: Every trainer rank's ``register_trainer_source`` result.
        prefetched_bytes: Each key's prefetch buffer, as flat bytes.
        load_groups: The keys of each ``copy_prefetched_`` call of a load.
        read_timeout_s: Seconds to wait for each read.

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
        *,
        read_timeout_s: float,
    ) -> None:
        if not all(isinstance(source, DeltaTrainerSource) for source in sources):
            raise ValueError("sync_mode='delta', but the trainer stages no deltas")
        # (source, segment start byte) -> (window, segment index).
        segment_of = {
            (s, start): (w, i)
            for s, source in enumerate(sources)
            for w, segments in enumerate(source.delta_segments)
            for i, (start, _) in enumerate(segments)
        }
        targets: list[dict[int, list[tuple[str, int] | None]]] = [{} for _ in sources]
        for read in reads:
            if read.local_byte_offset % 2 or read.nbytes % 2:
                raise ValueError(
                    f"{read.key}: delta sync applies 2-byte words; the piece has "
                    f"{read.nbytes} bytes at prefetch byte {read.local_byte_offset}"
                )
            end = read.byte_offset + read.nbytes
            for w in range(
                read.byte_offset // _DELTA_WINDOW_NBYTES,
                (end - 1) // _DELTA_WINDOW_NBYTES + 1,
            ):
                lo = max(read.byte_offset, w * _DELTA_WINDOW_NBYTES)
                window, i = segment_of[(read.source, lo)]
                assert window == w
                segments = targets[read.source].setdefault(
                    w, [None] * len(sources[read.source].delta_segments[w])
                )
                segments[i] = (
                    read.key,
                    (read.local_byte_offset + lo - read.byte_offset) // 2,
                )
        self._plans = []
        offset = 0
        for source, windows in zip(sources, targets):
            bounds = _delta_windows(source.staging_nbytes)
            # The trainer's own bound for these windows.
            capacity = _delta_capacity_nbytes(
                bounds[w][1] - bounds[w][0] for w in windows
            )
            self._plans.append(
                _DeltaPlan(
                    dict(sorted(windows.items())),
                    source.delta_remote_buffer,
                    offset,
                    capacity,
                )
            )
            offset += capacity
        receive = _pinned_buffer(max(offset, 1))
        self._receive = (receive, transport.register_memory(receive))
        self._prefetched_bytes = prefetched_bytes
        self._timeout = timedelta(seconds=read_timeout_s)
        self._executor = _new_codec_executor()
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
    ) -> tuple[list[tuple[Any, Any]], list[list[_WindowApply]]] | None:
        """The reads of the changes this rank applies, one per run of windows
        adjacent in a delta buffer, and per read the (window's segments, its
        changes, receive buffer offset) of each window in it; None to read the
        push whole."""
        memory = self._receive[1]
        reads = []
        applies = []
        for plan, metadata in zip(self._plans, metadata_by_trainer_rank):
            needed = [
                (segments, metadata.delta_records[w])
                for w, segments in plan.windows.items()
                if metadata.delta_records[w].nbytes
            ]
            if (
                sum(delta_record.nbytes for _, delta_record in needed)
                > plan.receive_capacity
            ):
                return None
            local = plan.receive_offset
            runs: list[list[int]] = []  # [remote offset, local offset, bytes]
            for segments, delta_record in needed:
                if runs and runs[-1][0] + runs[-1][2] == delta_record.offset:
                    runs[-1][2] += delta_record.nbytes
                else:
                    runs.append([delta_record.offset, local, delta_record.nbytes])
                    applies.append([])
                applies[-1].append((segments, delta_record, local))
                local += delta_record.nbytes
            reads += [
                (memory.to_mutable_view(lo, n), plan.delta_buffer.slice(remote, n))
                for remote, lo, n in runs
            ]
        return reads, applies

    def decode(
        self,
        works: list[Any],
        applies: list[list[_WindowApply]],
    ) -> None:
        """Decompress each read's windows on the codec threads once the read
        completes, and pack their changes by load group for the load."""
        receive = self._receive[0].numpy()
        futures = []
        for work, read_applies in zip(works, applies, strict=True):
            # A timeout leaves the transfers in flight: the job must restart.
            work.wait(self._timeout)
            futures += [
                self._executor.submit(
                    _decompress_window,
                    receive[offset : offset + delta_record.nbytes],
                    delta_record,
                    segments,
                )
                for segments, delta_record, offset in read_applies
            ]
        changes: dict[str, list[_KeyChanges]] = {}
        for future in futures:
            for key, key_changes in future.result():
                changes.setdefault(key, []).append(key_changes)
        groups: dict[int, list[tuple[str, list[_KeyChanges]]]] = {}
        for key, key_changes in changes.items():
            groups.setdefault(self._key_groups[key], []).append((key, key_changes))
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
        if group.positions is None:
            group.positions = group.positions_on(device)
        positions, xors = group.positions
        start, end = group.keys[key]
        group.unloaded.discard(key)
        if not group.unloaded:
            group.positions = None
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
        unloaded = {id(group): group for group in self._pending.values()}
        self._writebacks += [
            self._executor.submit(self._write_back, group)
            for group in [*self._loaded, *unloaded.values()]
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
        return ModelStateDictTransferMetadata(())

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
        self,
        metadata_by_trainer_rank: tuple[ModelStateDictTransferMetadata, ...],
        *,
        full: bool,
    ) -> None:
        """Fill the prefetch buffers from the trainer's staging buffers.

        ``metadata_by_trainer_rank`` is returned by the same trainer ranks whose
        ``TrainerSource`` values were passed to ``connect_generator``.
        ``full`` is set by a generator whose prefetch buffers do not yet contain
        a complete model state dict.

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
        if config.sync_mode != "delta":
            raise ValueError(f"Expected sync_mode='delta', got {config.sync_mode!r}")
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
        """Register the staging buffer, then the delta buffer and its windows."""
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
        nbytes = max(
            (_round_up(start + n, _ALIGNMENT) for _, _, start, n in pieces), default=0
        )
        self._trainer_delta = _TrainerDelta(self._transport, staging, nbytes, pieces)
        return DeltaTrainerSource(
            source.pieces,
            source.remote_buffer,
            source.url,
            self._trainer_delta.delta_buffer,
            nbytes,
            self._trainer_delta.segments,
        )

    def stage(self, model_state_dict: dict[str, torch.Tensor]) -> None:
        """Assemble each window of the staging buffer on the shards' device and,
        after the first push, compare it with the window's previous contents
        before overwriting them, staging the changed words' gaps and XORs, grouped
        by segment, for ``finish_stage``. The copies and the comparison are
        asynchronous, with no host synchronization; synchronize the current stream
        before ``finish_stage`` and before generators read.
        """
        assert self._trainer_delta is not None
        with torch.no_grad():
            # Cast on the device, as ``stage`` does.
            self._trainer_delta.stage(
                model_state_dict,
                lambda key: _local(model_state_dict[key]).to(
                    self._staging_plan[key][0]
                ),
            )

    def finish_stage(self) -> ModelStateDictTransferMetadata:
        """After the first push, compress each changed window's record into the
        delta buffer and report where each landed."""
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
        """Plan the reads, then which windows of each trainer rank's changes this
        rank applies."""
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
            read_timeout_s=self._read_timeout_s,
        )

    async def read(
        self,
        metadata_by_trainer_rank: tuple[ModelStateDictTransferMetadata, ...],
        *,
        full: bool,
    ) -> None:
        """If the prefetch buffers hold the preceding push and every trainer rank's
        changes fit the receive buffer, read and decompress the changes of the
        windows covering this rank's pieces, which ``copy_prefetched_`` applies;
        otherwise read every piece whole."""
        delta = self._generator_delta
        assert delta is not None, "connect_generator must be called first"
        delta.write_back_rest()
        pushes = {metadata.push for metadata in metadata_by_trainer_rank}
        assert len(pushes) <= 1, f"trainer ranks are on different pushes: {pushes}"
        push = pushes.pop() if pushes else 0
        planned = None
        if (
            not full
            and self._last_push is not None
            and push == self._last_push + 1
            and all(metadata.delta_records for metadata in metadata_by_trainer_rank)
        ):
            planned = delta.plan_pull(metadata_by_trainer_rank)
        if planned is None:
            # Whole reads overwrite the prefetch buffers: finish writing the last
            # push's changes into them first.
            await asyncio.to_thread(delta.wait_write_backs)
            await super().read(metadata_by_trainer_rank, full=True)
        else:
            reads, applies = planned
            if reads:
                works = await asyncio.to_thread(
                    lambda: [
                        self._transport.read(local, remote, async_op=True)
                        for local, remote in reads
                    ]
                )
                await asyncio.to_thread(delta.decode, works, applies)
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
