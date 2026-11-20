# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delta sync for ``ModelStateDictTransport``: after the first pull, each pull
reads only the words that changed since the preceding push.

See ``MODEL_STATE_DICT_DELTA_TRANSPORT.md``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
from torch.distributed._transport import wait_all

from torchtitan.rl.model_state_dict_transport import (
    local_tensor,
    ModelStateDictTransferMetadata,
    ModelStateDictTransport,
    pinned_buffer,
    Read,
    TensorShard,
    TrainerSource,
)

# Delta sync compares each push with the preceding one per window of this many
# staging buffer bytes, as 2-byte words: the unit the trainer diffs on its
# device and a generator rank reads and applies.
_DELTA_WINDOW_NBYTES = 64 << 20
# zstd's fastest default level.
_DELTA_ZSTD_LEVEL = 1


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
class ModelStateDictDeltaTransferMetadata(ModelStateDictTransferMetadata):
    """A delta sync push's metadata."""

    delta_records: dict[int, DeltaRecord] | None = None
    """The record of each staged piece that changed since the preceding push, by
    its ``StagedPiece.byte_offset``. None to read the push whole: the first push,
    or a push with more changes than a window or the delta buffer holds."""
    push: int = 0
    """Index of the push, from 0."""


@dataclass(frozen=True, slots=True)
class DeltaTrainerSource(TrainerSource):
    """A ``TrainerSource`` that also stages each push's changes, for delta sync
    (``ModelStateDictDeltaTransport``)."""

    delta_remote_buffer: Any
    """The delta buffer's ``torch.distributed._transport.RemoteBuffer``: the
    records of the latest push."""


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


def _delta_capacity(window_nbytes: int, capacity_divisor: int) -> int:
    """The most changed words a window of ``window_nbytes`` bytes sends."""
    return -(-window_nbytes // 2 // capacity_divisor)


def _delta_capacity_nbytes(sizes: Iterable[int], capacity_divisor: int) -> int:
    """Bytes reserved for the compressed changes of windows (trainer) or staged
    pieces (generator) of these sizes: 4 per changed word (a 2-byte gap and the
    word)."""
    return sum(4 * _delta_capacity(n, capacity_divisor) for n in sizes)


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


def _zstd() -> Any:
    # TODO: zstandard is not a PyTorch library; switch to the standard library's
    # compression.zstd once torchtitan requires Python 3.14.
    import zstandard

    return zstandard


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
        capacity_divisor: ``ModelStateDictDeltaTransport.Config.capacity_divisor``.
        codec_threads: ``ModelStateDictDeltaTransport.Config.codec_threads``.
    """

    def __init__(
        self,
        transport: Any,
        staging: torch.Tensor,
        pieces: list[tuple[str, tuple[slice, ...], int, int]],
        *,
        capacity_divisor: int,
        codec_threads: int,
    ) -> None:
        self._staging = staging
        self._pieces = sorted(pieces, key=lambda piece: piece[2])
        # An empty staging buffer is still allocated with one byte.
        bounds = _delta_windows(staging.numel() if pieces else 0)
        # window_pieces[w]: (key, index in the trainer's local tensor, byte_offset,
        # bytes) of each piece that overlaps window w, in staging order. A piece
        # that spans windows is in each.
        window_pieces: list[list[tuple[str, tuple[slice, ...], int, int]]] = [
            [] for _ in bounds
        ]
        for piece in self._pieces:
            for w in _windows_of(piece[2], piece[3]):
                window_pieces[w].append(piece)
        self._windows = []
        for (lo, hi), window_piece_list in zip(bounds, window_pieces):
            capacity = _delta_capacity(hi - lo, capacity_divisor)
            positions = pinned_buffer(capacity * torch.int32.itemsize)
            xors = pinned_buffer(capacity * torch.int16.itemsize)
            self._windows.append(
                _Window(
                    lo,
                    hi,
                    window_piece_list,
                    positions.view(torch.int32),
                    xors.view(torch.int16),
                )
            )
        # Each window's number of changed words in this push, copied from the GPU
        # by ``stage``; ``compress_records`` reads it to split the positions and
        # XORs and to check the capacity.
        counts = pinned_buffer(max(len(bounds), 1) * torch.int32.itemsize)
        self._counts = counts.view(torch.int32)
        self._delta = pinned_buffer(
            max(
                _delta_capacity_nbytes(
                    (hi - lo for lo, hi in bounds), capacity_divisor
                ),
                1,
            )
        )
        self.delta_buffer = transport.register_memory(self._delta).to_remote_buffer()
        self._num_pushes = 0
        self._executor = ThreadPoolExecutor(
            codec_threads, thread_name_prefix="model_state_dict_delta"
        )

    def stage(self, piece_of: Callable[[str, tuple[slice, ...]], torch.Tensor]) -> None:
        """Assemble each window of the staging buffer on the shards' device and,
        after the first push, stage its changes before overwriting it.

        ``piece_of(key, index)`` is the piece at ``index`` of this rank's shard of
        ``key``, in the dtype it is staged in.
        """
        if not self._windows:
            return
        diff = self._num_pushes > 0
        device = piece_of(*self._pieces[0][:2]).device
        counts = torch.zeros(len(self._windows), dtype=torch.int32, device=device)
        for w, window in enumerate(self._windows):
            staging_window = self._staging[window.lo : window.hi]
            # The previous push, read back before it is overwritten below; both
            # copies run in stream order. It also fills the padding between pieces.
            old = staging_window.to(device, non_blocking=True)
            new = old.clone()
            for key, index, start, nbytes in window.pieces:
                # The raw bytes of the piece, to cut at the window's byte bounds.
                piece = piece_of(key, index).contiguous().view(-1).view(torch.uint8)
                # [a, b): the staging buffer bytes of the piece in this window,
                # shifted into the window's and the piece's own byte offsets below.
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
            staging_window.copy_(new, non_blocking=True)
        if diff:
            self._counts[: counts.numel()].copy_(counts, non_blocking=True)

    def compress_records(self) -> ModelStateDictDeltaTransferMetadata:
        """After the first push, compress each changed piece's record into the delta
        buffer and report where each landed."""
        push = self._num_pushes
        self._num_pushes += 1
        if push == 0:
            return ModelStateDictDeltaTransferMetadata(push=push)
        if not self._windows:
            # Nothing staged on this rank: an empty delta, not a whole read.
            return ModelStateDictDeltaTransferMetadata({}, push=push)
        counts = self._counts[: len(self._windows)].tolist()
        # A window with more changed words than its capacity overflowed its
        # positions and XORs buffers (nonzero_static kept only the first
        # capacity): send no delta, so generators read the push whole.
        if any(
            count > window.positions.numel()
            for count, window in zip(counts, self._windows)
        ):
            return ModelStateDictDeltaTransferMetadata(push=push)

        def compress(
            piece: tuple[str, tuple[slice, ...], int, int],
        ) -> tuple[int, bytes, int] | None:
            """One piece's record: take the piece's slice of each window it
            overlaps (``searchsorted`` over the window's sorted positions by the
            piece's word range), make the positions relative to the piece, turn
            them into gaps (the unchanged words before each changed word), and
            zstd-compress ``[gaps | XORs]``. Returns (``byte_offset``, the
            compressed record, ``gap_itemsize``), or None if the piece did not
            change."""
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
        # The records do not fit the delta buffer (sized for 4 bytes per changed
        # word at every window's capacity; e.g. 4-byte gaps that compress
        # poorly): send no delta, so generators read the push whole.
        if sum(len(data) for _, data, _ in compressed) > self._delta.numel():
            return ModelStateDictDeltaTransferMetadata(push=push)
        delta = self._delta.numpy()
        records = {}
        offset = 0
        # Pack the records back to back into the registered delta buffer, which
        # generators RDMA-read, and note where each landed (by its piece's
        # byte_offset) in the metadata they plan those reads with.
        for start, data, gap_itemsize in compressed:
            delta[offset : offset + len(data)] = np.frombuffer(data, dtype=np.uint8)
            records[start] = DeltaRecord(offset, len(data), gap_itemsize)
            offset += len(data)
        return ModelStateDictDeltaTransferMetadata(records, push=push)


# A record to decode: its piece's Read, its DeltaRecord, and its compressed
# bytes in the receive buffer.
_RecordRead = tuple[Read, DeltaRecord, np.ndarray]


class _DeltaPlan(NamedTuple):
    """How a generator rank reads one trainer rank's records."""

    records: dict[int, Read]
    """The read of each piece this rank reads, by its ``StagedPiece.byte_offset``.
    """
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

    @classmethod
    def pack(
        cls,
        changes: list[tuple[Read, np.ndarray, np.ndarray]],
        *,
        pin_memory: bool,
    ) -> _Group:
        """Pack the changes of the group's pieces, by key.

        Args:
            changes: One entry per changed piece of the group: its ``Read`` (its
                key, and its byte offset in that key's prefetch buffer); its gaps,
                the unchanged words before each changed word (uint16 or uint32);
                and each changed word's XOR (int16).
            pin_memory: Whether to pin the packed buffers.
        """
        num_changes = sum(piece_gaps.size for _, piece_gaps, _ in changes)
        gaps = torch.empty(num_changes, dtype=torch.int32, pin_memory=pin_memory)
        xors = torch.empty(num_changes, dtype=torch.int16, pin_memory=pin_memory)
        gaps_np, xors_np = gaps.numpy(), xors.numpy()
        # Per piece: the index in gaps of its first change (starts), and the jump
        # added there so the running sum restarts at its word offset in its key
        # (shifts). Per key: its [first, end) changes in gaps and xors (keys).
        starts, shifts, keys = [], [], {}
        # Where the next piece's changes go in gaps and xors.
        start = 0
        previous_end = 0  # The running sum before a piece: past the last change.
        for read, piece_gaps, piece_xors in sorted(changes, key=lambda c: c[0].key):
            end = start + piece_gaps.size
            gaps_np[start:end] = piece_gaps
            xors_np[start:end] = piece_xors
            # The piece's word offset in its key's prefetch buffer.
            offset = read.local_byte_offset // 2
            starts.append(start)
            shifts.append(offset - previous_end)
            previous_end = offset + int(piece_gaps.sum(dtype=np.int64)) + (end - start)
            first, _ = keys.get(read.key, (start, end))
            keys[read.key] = (first, end)
            start = end
        return cls(
            gaps,
            xors,
            torch.tensor(starts, dtype=torch.long, pin_memory=pin_memory),
            torch.tensor(shifts, dtype=torch.long, pin_memory=pin_memory),
            keys,
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
        reads: The rank's planned reads (``ModelStateDictTransport._read_plan``).
        sources: Every trainer rank's ``register_trainer_source`` result.
        prefetched: The prefetch buffers (``connect_generator``'s
            ``prefetched_model_state_dict``).
        load_groups: The keys of each ``copy_prefetched_`` call of a load.
        capacity_divisor: ``ModelStateDictDeltaTransport.Config.capacity_divisor``,
            the trainer's.
        codec_threads: ``ModelStateDictDeltaTransport.Config.codec_threads``.

    Raises:
        ValueError: if a read is not whole 2-byte words, which the changes are.
    """

    def __init__(
        self,
        transport: Any,
        reads: list[Read],
        sources: list[DeltaTrainerSource],
        prefetched: dict[str, torch.Tensor],
        load_groups: list[list[str]],
        *,
        capacity_divisor: int,
        codec_threads: int,
    ) -> None:
        # Each read is one staged piece.
        for read in reads:
            if read.local_byte_offset % 2 or read.nbytes % 2:
                raise ValueError(
                    f"{read.key}: delta sync applies 2-byte words; the piece has "
                    f"{read.nbytes} bytes at prefetch byte {read.local_byte_offset}"
                )
        self._plans = []
        offset = 0
        for i, source in enumerate(sources):
            # This rank's pieces from trainer rank i, in staging order, so records
            # adjacent in its delta buffer merge into one read.
            source_reads = sorted(
                (read for read in reads if read.source == i),
                key=lambda read: read.byte_offset,
            )
            # The trainer's own bound for these records.
            capacity = _delta_capacity_nbytes(
                (read.nbytes for read in source_reads), capacity_divisor
            )
            self._plans.append(
                _DeltaPlan(
                    {read.byte_offset: read for read in source_reads},
                    source.delta_remote_buffer,
                    offset,
                    capacity,
                )
            )
            offset += capacity
        receive = pinned_buffer(max(offset, 1))
        self._receive = (receive, transport.register_memory(receive))
        self._prefetched = prefetched
        self._executor = ThreadPoolExecutor(
            codec_threads, thread_name_prefix="model_state_dict_delta"
        )
        # The index in load_groups of each key, to pack each group's changes
        # together.
        self._key_groups = {
            key: i for i, group in enumerate(load_groups) for key in group
        }
        # The changes of the last delta read that no load has applied yet, by
        # index in load_groups. Set by decode, popped by take_group.
        self._pending: dict[int, _Group] = {}
        # Groups whose keys were all loaded, not yet written into the prefetch
        # buffers; start_write_back_after_load starts their write-back once
        # nothing is pending.
        self._loaded: list[_Group] = []
        # The background write-backs into the prefetch buffers, which a read waits
        # for before the next load copies from them (wait_write_backs).
        self._writebacks: list[Future] = []

    def plan_pull(
        self,
        metadata_by_trainer_rank: tuple[ModelStateDictDeltaTransferMetadata, ...],
    ) -> tuple[list[tuple[Any, Any]], list[_RecordRead]] | None:
        """The reads of the records this rank applies, one per run of records
        adjacent in a delta buffer, and each record to decode once they land; None
        to read the push whole."""
        # The receive buffer, and its registration the reads write into.
        receive_buffer = self._receive[0].numpy()
        receive_memory = self._receive[1]
        # Each RDMA read as (destination: bytes of this generator rank's receive
        # buffer, source: bytes of the trainer rank's delta buffer).
        reads = []
        # Per record, for decode: its piece's Read (the _plan_reads entry: key,
        # prefetch buffer offset, ...), its DeltaRecord, and its bytes in the
        # receive buffer.
        records = []
        for plan, metadata in zip(self._plans, metadata_by_trainer_rank):
            # (Read, DeltaRecord) of each piece this rank reads that changed, in
            # staging order: the trainer rank's records minus other ranks' pieces
            # and unchanged ones.
            needed = [
                (read, metadata.delta_records[byte_offset])
                for byte_offset, read in plan.records.items()
                if byte_offset in metadata.delta_records
            ]
            # This rank's records from this trainer rank do not fit its share of
            # the receive buffer (its changes are concentrated in this rank's
            # pieces): read the push whole.
            if (
                sum(delta_record.nbytes for _, delta_record in needed)
                > plan.receive_capacity
            ):
                return None
            receive_offset = plan.receive_offset
            # [delta buffer offset, receive buffer offset, bytes] of each run of
            # adjacent records: one RDMA read.
            runs: list[list[int]] = []
            for read, delta_record in needed:
                # A record that starts where the last run ends in the delta buffer
                # extends that run; otherwise it starts a new run. Records are
                # placed back to back in the receive buffer, so a run is
                # contiguous there too.
                if runs and runs[-1][0] + runs[-1][2] == delta_record.offset:
                    runs[-1][2] += delta_record.nbytes
                else:
                    runs.append(
                        [delta_record.offset, receive_offset, delta_record.nbytes]
                    )
                compressed = receive_buffer[
                    receive_offset : receive_offset + delta_record.nbytes
                ]
                records.append((read, delta_record, compressed))
                receive_offset += delta_record.nbytes
            reads += [
                (
                    receive_memory.to_mutable_view(receive_offset, nbytes),
                    plan.delta_buffer.slice(delta_offset, nbytes),
                )
                for delta_offset, receive_offset, nbytes in runs
            ]
        return reads, records

    def decode(self, records: list[_RecordRead]) -> None:
        """Decompress the read records on the codec threads, and pack their changes
        by load group for the load."""
        futures = [
            (read, self._executor.submit(_decompress_record, compressed, delta_record))
            for read, delta_record, compressed in records
        ]
        groups = defaultdict(list)
        for read, future in futures:
            groups[self._key_groups[read.key]].append((read, *future.result()))
        pin_memory = torch.accelerator.is_available()
        packed = self._executor.map(
            lambda group: _Group.pack(group, pin_memory=pin_memory), groups.values()
        )
        self._pending = dict(zip(groups, packed))

    def take_group(self, keys: Iterable[str]) -> _Group | None:
        """The read changes of the load group whose ``keys`` one
        ``copy_prefetched_`` call gets, or None if it has none; marks them loaded.
        """
        key = next(iter(keys), None)
        group = None if key is None else self._pending.pop(self._key_groups[key], None)
        if group is not None:
            self._loaded.append(group)
        return group

    def start_write_back_after_load(self) -> None:
        """Once every read change is loaded, start writing them into the prefetch
        buffers in the background; writing them during the load would slow its
        copies."""
        if not self._pending:
            self._writebacks += [
                self._executor.submit(self._write_back, group) for group in self._loaded
            ]
            self._loaded = []

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
                local_tensor(self._prefetched[key]).view(-1).view(torch.int16),
                positions[start:end],
                xors[start:end],
            )


class ModelStateDictDeltaTransport(ModelStateDictTransport):
    """A ``ModelStateDictTransport`` whose pulls after the first read only the
    words that changed since the preceding push, as zstd-compressed gaps and
    XORs, which the trainer stages in a second, smaller buffer
    (``_TrainerDelta``) and generators apply when they load
    (``_GeneratorDelta``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelStateDictTransport.Config):
        capacity_divisor: int = 16
        """Each 64 MiB window of the staging buffer sends at most 1 /
        ``capacity_divisor`` of its words; a push with more changes in a window is
        read whole. It sizes the trainer's pinned buffers of changes and the
        registered delta and receive buffers (1 / (2 x ``capacity_divisor``) of the
        bytes they cover). 16 keeps the fallback rare at RL learning rates (about 2%
        of bf16 words change per push on Qwen3.5-27B); lower it if pushes read
        whole."""

        codec_threads: int = 32
        """Threads per process that compress (trainer) or decompress and write back
        (generator) the records; zstd and PyTorch ops release the GIL. 32 was
        measured on GB300 hosts (144 CPUs, 4 ranks per host, each with a 72-CPU
        affinity), where the generator's write-back keeps all 32 busy; lower it
        on hosts with more ranks per CPU."""

        def __post_init__(self) -> None:
            if self.capacity_divisor < 1:
                raise ValueError(
                    f"capacity_divisor must be at least 1, got {self.capacity_divisor}"
                )
            if self.codec_threads < 1:
                raise ValueError(
                    f"codec_threads must be at least 1, got {self.codec_threads}"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self._capacity_divisor = config.capacity_divisor
        self._codec_threads = config.codec_threads
        # Trainer or generator rank's delta state, and the last push read.
        self._trainer_delta: _TrainerDelta | None = None
        self._generator_delta: _GeneratorDelta | None = None
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
        self._trainer_delta = _TrainerDelta(
            self._transport,
            staging,
            pieces,
            capacity_divisor=self._capacity_divisor,
            codec_threads=self._codec_threads,
        )
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
        ``prepare_transfer``. The copies and the comparison are asynchronous, with
        no host synchronization; synchronize the current stream before
        ``prepare_transfer`` and before generators read.
        """
        assert self._trainer_delta is not None
        with torch.no_grad():
            # Cast each piece on the device, as ``stage`` does.
            self._trainer_delta.stage(
                lambda key, index: local_tensor(model_state_dict[key])[index].to(
                    self._staging_plan[key][0]
                ),
            )

    def prepare_transfer(self) -> ModelStateDictDeltaTransferMetadata:
        """After the first push, compress each changed piece's record into the delta
        buffer and report where each landed."""
        assert self._trainer_delta is not None
        return self._trainer_delta.compress_records()

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
        applies.

        Raises:
            ValueError: if the trainer does not use delta sync.
        """
        if not all(isinstance(source, DeltaTrainerSource) for source in sources):
            raise ValueError("delta sync, but the trainer stages no deltas")
        super().connect_generator(
            prefetched_model_state_dict,
            load_groups=load_groups,
            sources=sources,
            rank=rank,
            generator_index=generator_index,
        )
        self._generator_delta = _GeneratorDelta(
            self._transport,
            self._read_plan,
            sources,
            prefetched_model_state_dict,
            load_groups,
            capacity_divisor=self._capacity_divisor,
            codec_threads=self._codec_threads,
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
        # Every pull loads its read before the next read.
        assert not delta._pending, "the previous read was never loaded"
        pushes = {metadata.push for metadata in metadata_by_trainer_rank}
        assert len(pushes) == 1, f"trainer ranks are on different pushes: {pushes}"
        push = pushes.pop()
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
        group = delta.take_group(state_dict.keys())
        if group is None:
            return super().copy_prefetched_(state_dict)
        # The group's changes on the load's device: one copy and one running sum.
        device = local_tensor(state_dict[next(iter(group.keys))]).device
        positions, xors = group.positions_on(device)
        unchanged = {}
        with torch.no_grad():
            for key, tensor in state_dict.items():
                if key not in group.keys:
                    unchanged[key] = tensor
                    continue
                local = local_tensor(tensor)
                start, end = group.keys[key]
                change = positions[start:end], xors[start:end]
                pieces = self._packed.get(key)
                prefetched = local_tensor(self._prefetched[key])
                if pieces is None and type(local) is torch.Tensor:
                    # The prefetch buffer has the tensor's layout: XOR in place.
                    local.copy_(prefetched, non_blocking=True)
                    _xor_at_(local.view(-1).view(torch.int16), *change)
                    continue
                # Packed, or a tensor subclass (e.g. a quantized parameter): XOR a
                # contiguous copy.
                source = prefetched.to(local.device, non_blocking=True, copy=True)
                _xor_at_(source.view(-1).view(torch.int16), *change)
                if pieces is None:
                    local.copy_(source)
                    continue
                # Unpack from the copy as ModelStateDictTransport.copy_prefetched_
                # does from the prefetch buffer: each piece's view of the prefetch
                # buffer gives its offset in it.
                source = source.view(-1)
                for index, view in pieces:
                    offset = view.storage_offset() - prefetched.storage_offset()
                    local[index].copy_(
                        source[offset : offset + view.numel()].view(view.shape),
                        non_blocking=True,
                    )
        super().copy_prefetched_(unchanged)
        # Finish the copies out of the prefetch buffers before the writes below
        # change them.
        if torch.accelerator.is_available():
            torch.accelerator.current_stream().synchronize()
        delta.start_write_back_after_load()
        return state_dict.keys()
