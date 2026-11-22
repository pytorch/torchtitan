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
import contextlib
import logging
import math
import time
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

logger = logging.getLogger(__name__)

# Byte alignment of each staged piece, so it can be viewed as any dtype.
_ALIGNMENT = 16

# XOR deltas (``Config.xor_delta_codec``) are encoded per window of this many
# staging buffer bytes, the unit a generator rank decompresses. A window's chunks
# are what nvCOMP runs in parallel: on B200, zstd encodes BF16 XOR deltas with
# 2.6% of elements changed at 6 / 23 GB/s and decodes them at 23 / 81 GB/s in
# 64 / 256 MiB windows, and the copies run at ~50 GB/s per direction. GPU scratch
# is four windows on the trainer and three on a generator.
_DELTA_WINDOW_NBYTES = 256 << 20
# nvCOMP compresses each window as independent chunks of this size. On B200, zstd
# decodes 256 KiB chunks 2.6x faster than 1 MiB ones, with the same ratio; 64 KiB
# chunks crash its compressor ("scratch buffer allocation exceeded").
_DELTA_CHUNK_NBYTES = 256 << 10
# A push whose delta compresses to more than 1 / _DELTA_MIN_RATIO of the bytes it
# covers is read whole instead, which bounds the delta buffers to that fraction of
# the staging buffer. Measured BF16 deltas compress 27-100x (MODEL_STATE_DICT_TRANSPORT.md).
_DELTA_MIN_RATIO = 4


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


class DeltaSource(NamedTuple):
    """Where a generator rank reads one trainer rank's XOR deltas."""

    remote_buffer: Any
    """The delta buffer's ``RemoteBuffer``: a header (``_delta_header_nbytes``),
    then each window's compressed XOR."""
    staging_nbytes: int
    """Size of the staging buffer the deltas cover, in windows of
    ``_DELTA_WINDOW_NBYTES``."""


@dataclass(frozen=True, slots=True)
class TrainerSource:
    """What a generator rank needs to read one trainer rank's staging buffer."""

    pieces: list[dict[str, StagedPiece]]
    """``pieces[rank][key]``: the staged part of generator rank ``rank``'s shard of
    ``key``, for each key this trainer rank holds part of."""
    remote_buffer: Any
    """The staging buffer's ``torch.distributed._transport.RemoteBuffer``."""
    url: bytes
    delta: DeltaSource | None = None
    """The XOR deltas, or None without ``Config.xor_delta_codec``."""


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


class _DeltaPlan(NamedTuple):
    """How a generator rank reads one trainer rank's pushes, whole or as XOR
    deltas."""

    reads: list[tuple[Any, Any]]
    """(local view, remote buffer) per piece, to read the push whole."""
    windows: dict[int, list[tuple[int, torch.Tensor]]]
    """The windows the pieces of non-resident keys touch, in order, each with the
    (offset in the window, prefetch buffer bytes) of every part of a piece in
    it."""
    resident_windows: dict[int, list[tuple[int, str, int, int]]]
    """The same for resident keys, which loads XOR on the GPU: the (offset in
    the window, key, byte offset in the key's tensor, size) of every part."""
    staging_nbytes: int
    remote_buffer: Any
    """The delta buffer."""
    header: torch.Tensor
    """The received header, an int64 view of the receive buffer."""
    header_read: tuple[Any, Any]
    receive_memory: Any
    """The registered receive buffer."""
    receive_offset: int
    """Where in the receive buffer this source's compressed windows go."""
    receive_capacity: int


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


def _delta_windows(staging_nbytes: int) -> list[tuple[int, int]]:
    """The ``[start, end)`` staging buffer bytes of each delta window."""
    return [
        (start, min(start + _DELTA_WINDOW_NBYTES, staging_nbytes))
        for start in range(0, staging_nbytes, _DELTA_WINDOW_NBYTES)
    ]


def _delta_header_nbytes(num_windows: int) -> int:
    """Size of a delta buffer's header: int64 push index, whether the push has a
    delta, then the (offset after the header, size) of each window's compressed
    XOR."""
    return -(-8 * (2 + 2 * num_windows) // _ALIGNMENT) * _ALIGNMENT


class _CachingAllocation:
    """nvCOMP device memory from PyTorch's caching allocator, on the current
    stream.

    nvCOMP's own allocations would sit outside the caching allocator's pool, and
    freeing one could synchronize the device under the trainer's forward/backward
    or the generator's decoding. Freed blocks return to the pool of the stream
    they were allocated on, which is the codec's stream (``_DeltaCodec.on_stream``),
    so work queued on it before the free still runs before any reuse.
    """

    def __init__(self, nbytes: int, stream: Any) -> None:
        self.tensor = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        self.ptr = self.tensor.data_ptr()


class _PipelineSlot:
    """GPU buffers that windows take turns in on their way through the H2D, codec
    and D2H streams, and the events that order a window's stages."""

    def __init__(self, device: torch.device, *nbytes: int) -> None:
        self.buffers = [
            torch.empty(n, dtype=torch.uint8, device=device) for n in nbytes
        ]
        self.loaded = torch.cuda.Event()
        """The window's inputs are on the GPU."""
        self.computed = torch.cuda.Event()
        """The codec stream is done with the window."""
        self.done = torch.cuda.Event()
        """The window's outputs left the GPU: the slot is free. Waiting before
        the first record is a no-op."""


class _DeltaCodec:
    """nvCOMP zstd for XOR delta windows on its own stream of ``device``, with
    an H2D and a D2H stream to overlap copies with it.

    Encodes and decodes run on worker threads, so the actor's event loop keeps
    serving; ``on_stream`` sets the device and codec stream there.
    """

    def __init__(self, device: torch.device) -> None:
        # TODO: nvCOMP is not a PyTorch library; switch to a PyTorch-native GPU
        # codec if one lands (no upstream issue yet).
        from nvidia import nvcomp

        # Process-wide; this transport is nvCOMP's only user in the process.
        nvcomp.set_device_allocator(_CachingAllocation)
        self._nvcomp = nvcomp
        self.device = device
        self.stream = torch.cuda.Stream(device)
        self.h2d = torch.cuda.Stream(device)
        self.d2h = torch.cuda.Stream(device)
        with self.on_stream():
            self._codec = nvcomp.Codec(
                algorithm="Zstd",
                device_id=device.index,
                cuda_stream=self.stream.cuda_stream,
                uncomp_chunk_size=_DELTA_CHUNK_NBYTES,
            )
        # Per uncompressed size. Configs built from a size, not from a compressed
        # buffer, spare encodes and decodes a host sync each.
        self._compression_configs: dict[int, Any] = {}
        self._decompression_configs: dict[int, Any] = {}

    def on_stream(self) -> contextlib.ExitStack:
        """Make the codec's device and stream current on this thread."""
        stack = contextlib.ExitStack()
        stack.enter_context(torch.cuda.device(self.device))
        stack.enter_context(torch.cuda.stream(self.stream))
        return stack

    def _array(self, tensor: torch.Tensor) -> Any:
        return self._nvcomp.as_array(tensor, self.stream.cuda_stream)

    def encode(self, data: torch.Tensor, out: torch.Tensor) -> int | None:
        """Compress the uint8 ``data`` and copy the compressed bytes to the start of
        ``out``, on the stream. Returns their size, or None if they do not fit.
        Waits for the stream, to learn the size."""
        nbytes = data.numel()
        config = self._compression_configs.get(nbytes)
        if config is None:
            config = self._codec.compression_config(nbytes)
            self._compression_configs[nbytes] = config
        encoded = self._codec.encode(self._array(data), compression_config=config)
        size = encoded.buffer_size
        if size > out.numel():
            return None
        # The DLPack view spans the output's capacity. It does not keep
        # ``encoded`` alive, so copy before returning; freeing ``encoded`` after
        # the copy is queued is stream-ordered (_CachingAllocation).
        out[:size].copy_(
            torch.from_dlpack(encoded).view(torch.uint8)[:size], non_blocking=True
        )
        return size

    def decode(self, data: torch.Tensor, out: torch.Tensor) -> None:
        """Decompress ``data`` into the uint8 ``out`` on the stream."""
        nbytes = out.numel()
        config = self._decompression_configs.get(nbytes)
        if config is None:
            config = self._codec.decompression_config(
                self._codec.compression_config(nbytes)
            )
            self._decompression_configs[nbytes] = config
        self._codec.decode(self._array(data), out=out, decompression_config=config)


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

    With ``Config.xor_delta_codec``, each trainer rank also encodes every push
    after the first as the XOR of its staging buffer with the previous push's,
    compressed on its GPU; a generator rank whose prefetch buffers hold the
    previous push reads only the compressed windows covering its pieces, then
    decompresses them and XORs them into its prefetch buffers on its GPU.

    Trainer: ``register_trainer_source`` once, then ``stage`` and ``finish_stage``
    per push.
    Generator: ``connect_generator`` once, then ``read`` per prefetch and
    ``copy_prefetched_`` per load.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        backend: Literal["mooncake", "nixl"] = "nixl"
        """The ``torch.distributed._transport`` backend."""

        read_timeout_s: float = 600.0
        """Seconds to wait for a generator rank's reads before raising. A pull
        moves tens of GB per generator rank at most, well under a minute at line
        rate; this fails a stuck transfer instead of hanging the controller."""

        xor_delta_codec: Literal["zstd"] | None = None
        """Send each push after the first as the XOR of its bytes with the
        previous push's, compressed with this nvCOMP codec, or None to send every
        push whole. An optimizer step flips few bits: on Qwen3-30B-A3B,
        0.4% of BF16 elements at lr 1e-6 and 2.6% at 5e-6, which zstd compresses
        64-100x and 27-53x. Lossless: generators end up with the same bytes. Needs
        a CUDA GPU on every trainer and generator rank and the
        ``nvidia-nvcomp-cu13`` package."""

    def __init__(self, config: Config):
        self._transport = new_transport(config.backend)
        self._read_timeout_s = config.read_timeout_s
        self._xor_delta_codec = config.xor_delta_codec
        # Trainer: the staging buffer and, per key with pieces on this rank, the
        # dtype it is staged in and a (local index, staging view) per piece.
        self._staging: torch.Tensor | None = None
        self._staging_plan: dict[
            str, tuple[torch.dtype, list[tuple[tuple[slice, ...], torch.Tensor]]]
        ] = {}
        # Trainer with XOR deltas: the previous push's staging buffer; the delta
        # buffer generators read (header, then compressed windows); and the
        # number of pushes so far.
        self._baseline: torch.Tensor | None = None
        self._delta: torch.Tensor | None = None
        self._num_pushes = 0
        # Generator: (local view, remote buffer) per read, or None until connected;
        # the prefetch buffers; and (shard index, packed view) per piece of each
        # packed key.
        self._reads: list[tuple[Any, Any]] | None = None
        self._prefetched: dict[str, torch.Tensor] = {}
        self._packed: dict[str, list[tuple[tuple[slice, ...], torch.Tensor]]] = {}
        # Generator with XOR deltas: a plan per source; the registered buffer that
        # receives headers and compressed windows; and the push index the prefetch
        # buffers hold, None before the first read.
        self._delta_plans: list[_DeltaPlan] = []
        self._receive: torch.Tensor | None = None
        self._version: int | None = None
        # Generator: the keys loads XOR on the GPU (connect_generator), and the
        # (plan, decodes) of the last read's windows that cover them.
        self._resident_keys: frozenset[str] = frozenset()
        self._resident_deltas: list[tuple[_DeltaPlan, list[tuple[int, int, int]]]] = []
        # Either side with XOR deltas: the GPU codec.
        self._codec: _DeltaCodec | None = None

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
            transfer_dtype: The dtype floating-point tensors are sent in, or None for
                their own.
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
                if transfer_dtype is None
                or key in preserve_dtype_keys
                or not tensor.is_floating_point()
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
        delta = None
        if self._xor_delta_codec is not None:
            # The device the shards live on, which encodes the deltas.
            device = next(iter(model_state_dict.values())).device
            self._codec = _DeltaCodec(device)
            self._baseline = _pinned_buffer(max(nbytes, 1))
            header_nbytes = _delta_header_nbytes(len(_delta_windows(nbytes)))
            self._delta = _pinned_buffer(
                header_nbytes + nbytes // _DELTA_MIN_RATIO // _ALIGNMENT * _ALIGNMENT
            )
            # Until the first push, the header says there is no delta.
            self._delta[:header_nbytes].zero_()
            delta_memory = self._transport.register_memory(self._delta)
            delta = DeltaSource(delta_memory.to_remote_buffer(), nbytes)
        self._staging_nbytes = nbytes
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
        return TrainerSource(
            pieces, memory.to_remote_buffer(), self._transport.bind(), delta
        )

    def stage(self, model_state_dict: dict[str, torch.Tensor]) -> None:
        """Cast this trainer rank's shards and copy them into the staging buffer.

        The copies are asynchronous; synchronize the current stream, then call
        ``finish_stage``, before generators read.
        """
        with torch.no_grad():
            for key, (dtype, copies) in self._staging_plan.items():
                # Cast on the device: a casting device-to-host copy_ goes through
                # a pageable temporary and casts on the CPU.
                local = _local(model_state_dict[key]).to(dtype)
                for index, view in copies:
                    view.copy_(local[index], non_blocking=True)

    async def finish_stage(self) -> None:
        """Finish a push once ``stage``'s copies have completed: with XOR deltas,
        encode it against the previous push, off the event loop."""
        if self._codec is not None:
            await asyncio.to_thread(self._encode_delta)

    def _encode_delta(self) -> None:
        """Write this push's delta to the delta buffer and copy the staging buffer
        to the baseline, window by window on the GPU.

        The first push, and a push whose delta outgrows the delta buffer, get a
        header without a delta: generators read them whole.
        """
        codec = self._codec
        staging, baseline, delta = self._staging, self._baseline, self._delta
        assert codec is not None and baseline is not None and delta is not None
        windows = _delta_windows(self._staging_nbytes)
        header_nbytes = _delta_header_nbytes(len(windows))
        encode = self._num_pushes > 0
        table = []
        size = 0
        start_s = time.perf_counter()
        with codec.on_stream(), torch.no_grad():
            # Window i loads into slot i % 2 on the H2D stream while window i - 1
            # encodes on the codec stream and stores the baseline on the D2H
            # stream. Every buffer outlives the final synchronize.
            slots = [
                _PipelineSlot(codec.device, _DELTA_WINDOW_NBYTES, _DELTA_WINDOW_NBYTES)
                for _ in range(2)
            ]

            def load(i: int) -> None:
                slot = slots[i % 2]
                lo, hi = windows[i]
                # Free once window i - 2 stored its baseline and was encoded.
                codec.h2d.wait_event(slot.done)
                codec.h2d.wait_event(slot.computed)
                with torch.cuda.stream(codec.h2d):
                    slot.buffers[0][: hi - lo].copy_(staging[lo:hi], non_blocking=True)
                    if encode:
                        slot.buffers[1][: hi - lo].copy_(
                            baseline[lo:hi], non_blocking=True
                        )
                    slot.loaded.record()

            if windows:
                load(0)
            for i, (lo, hi) in enumerate(windows):
                slot = slots[i % 2]
                if i + 1 < len(windows):
                    load(i + 1)
                new, xor = slot.buffers[0][: hi - lo], slot.buffers[1][: hi - lo]
                codec.d2h.wait_event(slot.loaded)
                with torch.cuda.stream(codec.d2h):
                    baseline[lo:hi].copy_(new, non_blocking=True)
                    slot.done.record()
                if not encode:
                    continue
                codec.stream.wait_event(slot.loaded)
                # Windows and staging buffer sizes are multiples of _ALIGNMENT.
                xor.view(torch.int64).bitwise_xor_(new.view(torch.int64))
                nbytes = codec.encode(xor, delta[header_nbytes + size :])
                slot.computed.record()
                if nbytes is None:
                    # Too large to be worth it: the remaining windows only update
                    # the baseline.
                    encode = False
                    continue
                table.append((size, nbytes))
                size += nbytes
            codec.d2h.synchronize()
            codec.stream.synchronize()
        header = delta[:header_nbytes].view(torch.int64)
        header[0] = self._num_pushes
        header[1] = int(encode)
        if table:
            header[2 : 2 + 2 * len(table)] = torch.tensor(table).view(-1)
        logger.info(
            f"Push {self._num_pushes}: "
            + (
                f"XOR delta of {self._staging_nbytes / 1e9:.2f} GB compressed to "
                f"{size / 1e6:.1f} MB ({self._staging_nbytes / max(size, 1):.0f}x)"
                if encode
                else "no XOR delta, generators read it whole"
            )
            + f" in {time.perf_counter() - start_s:.2f} s"
        )
        self._num_pushes += 1

    def connect_generator(
        self,
        prefetched_model_state_dict: dict[str, torch.Tensor],
        *,
        sources: list[TrainerSource],
        rank: int,
        generator_index: int,
        resident_keys: frozenset[str] = frozenset(),
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
            resident_keys: Keys whose load destination keeps what the last load
                wrote until the next load (e.g. quantized operands loaded in place),
                all loaded by one ``copy_prefetched_`` call. With XOR deltas, that
                call XORs them into the destination on the GPU, and their prefetch
                buffers go stale until a push is read whole.
        """
        self._resident_keys = resident_keys
        reads, packed = _plan_reads(
            sources,
            shard_layout(prefetched_model_state_dict),
            rank=rank,
            generator_index=generator_index,
        )
        if packed_resident := sorted(packed.keys() & resident_keys):
            # Their deltas are laid out like the prefetch buffers, not like the
            # destination.
            raise ValueError(
                f"Resident keys must arrive in place, but {packed_resident} are packed"
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
        if self._xor_delta_codec is not None:
            self._plan_deltas(sources, reads, locals_)

    def _plan_deltas(
        self,
        sources: list[TrainerSource],
        reads: list[_Read],
        locals_: dict[str, torch.Tensor],
    ) -> None:
        """Plan each source's delta reads and allocate and register the buffer that
        receives them."""
        if any(source.delta is None for source in sources):
            raise ValueError(
                "xor_delta_codec is set, but the trainer does not encode deltas"
            )
        self._codec = _DeltaCodec(
            torch.device("cuda", torch.accelerator.current_device_index())
        )
        source_reads: list[list[tuple[Any, Any]]] = [[] for _ in sources]
        windows: list[dict[int, list[tuple[int, torch.Tensor]]]] = [{} for _ in sources]
        resident_windows: list[dict[int, list[tuple[int, str, int, int]]]] = [
            {} for _ in sources
        ]
        for r, pair in zip(reads, self._reads, strict=True):
            source_reads[r.source].append(pair)
            view = locals_[r.key].view(torch.uint8)
            view = view[r.local_byte_offset : r.local_byte_offset + r.nbytes]
            end = r.byte_offset + r.nbytes
            for w in range(
                r.byte_offset // _DELTA_WINDOW_NBYTES,
                (end - 1) // _DELTA_WINDOW_NBYTES + 1,
            ):
                lo = max(r.byte_offset, w * _DELTA_WINDOW_NBYTES)
                hi = min(end, (w + 1) * _DELTA_WINDOW_NBYTES)
                start = lo - w * _DELTA_WINDOW_NBYTES
                if r.key in self._resident_keys:
                    resident_windows[r.source].setdefault(w, []).append(
                        (
                            start,
                            r.key,
                            r.local_byte_offset + lo - r.byte_offset,
                            hi - lo,
                        )
                    )
                else:
                    windows[r.source].setdefault(w, []).append(
                        (start, view[lo - r.byte_offset : hi - r.byte_offset])
                    )
        # Receive buffer: per source, its header, then room for its windows'
        # compressed bytes up to the trainer's own limit (_DELTA_MIN_RATIO).
        sizes = []
        for source, cpu_windows, gpu_windows in zip(sources, windows, resident_windows):
            staging_nbytes = source.delta.staging_nbytes
            covered = sum(
                min((w + 1) * _DELTA_WINDOW_NBYTES, staging_nbytes)
                - w * _DELTA_WINDOW_NBYTES
                for w in cpu_windows.keys() | gpu_windows.keys()
            )
            sizes.append(
                (
                    _delta_header_nbytes(len(_delta_windows(staging_nbytes))),
                    -(-covered // _DELTA_MIN_RATIO // _ALIGNMENT) * _ALIGNMENT,
                )
            )
        self._receive = _pinned_buffer(sum(h + c for h, c in sizes))
        memory = self._transport.register_memory(self._receive)
        offset = 0
        for s, (source, (header_nbytes, capacity)) in enumerate(zip(sources, sizes)):
            self._delta_plans.append(
                _DeltaPlan(
                    reads=source_reads[s],
                    windows=dict(sorted(windows[s].items())),
                    resident_windows=dict(sorted(resident_windows[s].items())),
                    staging_nbytes=source.delta.staging_nbytes,
                    remote_buffer=source.delta.remote_buffer,
                    header=self._receive[offset : offset + header_nbytes].view(
                        torch.int64
                    ),
                    header_read=(
                        memory.to_mutable_view(offset, header_nbytes),
                        source.delta.remote_buffer.slice(0, header_nbytes),
                    ),
                    receive_memory=memory,
                    receive_offset=offset + header_nbytes,
                    receive_capacity=capacity,
                )
            )
            offset += header_nbytes + capacity

    async def _transfer(self, reads: list[tuple[Any, Any]]) -> None:
        """Run ``(local view, remote buffer)`` reads and wait for them."""
        if not reads:
            return
        works = await asyncio.to_thread(
            lambda: [
                self._transport.read(local, remote, async_op=True)
                for local, remote in reads
            ]
        )
        await wait_all(works, timeout=self._read_timeout_s)

    async def read(self) -> None:
        """Fill the prefetch buffers from the trainer's staging buffers.

        With XOR deltas, read each trainer rank's header first. If the prefetch
        buffers (and resident keys' destinations) hold the push before this one and
        every trainer rank encoded a delta that fits the receive buffer, read the
        compressed windows covering this rank's pieces and XOR them into the
        prefetch buffers, leaving resident keys to ``copy_prefetched_``; otherwise
        read every piece whole.

        A timeout leaves the transfers in flight: the prefetch buffers are
        undefined and the job must restart.
        """
        assert self._reads is not None, "connect_generator must be called first"
        if self._codec is None:
            await self._transfer(self._reads)
            return
        start_s = time.perf_counter()
        await self._transfer([plan.header_read for plan in self._delta_plans])
        versions = {int(plan.header[0]) for plan in self._delta_plans}
        assert len(versions) == 1, f"trainer ranks are on different pushes: {versions}"
        (version,) = versions
        planned = [
            (plan, *self._plan_delta_reads(plan, version))
            for plan in self._delta_plans
            if plan.windows or plan.resident_windows
        ]
        # One trainer rank without a usable delta reads the push whole: a resident
        # key's prefetch buffer is stale, so its pieces cannot mix the two.
        whole = any(decodes is None for _, _, decodes in planned)
        transfers = self._reads if whole else [t for _, ts, _ in planned for t in ts]
        await self._transfer(transfers)
        read_s = time.perf_counter() - start_s
        self._resident_deltas = []
        delta_nbytes = 0
        if not whole and planned:
            deltas = []
            for plan, _, decodes in planned:
                delta_nbytes += sum(n for _, _, n in decodes)
                if cpu := [d for d in decodes if d[0] in plan.windows]:
                    deltas.append((plan, cpu))
                if gpu := [d for d in decodes if d[0] in plan.resident_windows]:
                    self._resident_deltas.append((plan, gpu))
            if deltas:
                await asyncio.to_thread(self._apply_deltas, deltas)
        self._version = version
        logger.info(
            f"Pull of push {version}: "
            + ("whole" if whole else f"XOR deltas ({delta_nbytes / 1e6:.1f} MB)")
            + f", read in {read_s:.2f} s, applied in "
            f"{time.perf_counter() - start_s - read_s:.2f} s"
        )

    def _plan_delta_reads(
        self, plan: _DeltaPlan, version: int
    ) -> tuple[list[tuple[Any, Any]], list[tuple[int, int, int]] | None]:
        """The reads of a source's compressed windows and the (window, receive
        buffer offset, size) of each, or no decodes to read its pieces whole."""
        has_delta = bool(plan.header[1])
        if not has_delta or self._version is None or version != self._version + 1:
            return [], None
        num_windows = len(_delta_windows(plan.staging_nbytes))
        table = plan.header[2 : 2 + 2 * num_windows].view(-1, 2).tolist()
        windows = sorted(plan.windows.keys() | plan.resident_windows.keys())
        if sum(table[w][1] for w in windows) > plan.receive_capacity:
            return [], None
        header_nbytes = plan.header.numel() * plan.header.itemsize
        # Windows are encoded back to back, so consecutive ones are one read.
        runs: list[list[int]] = []
        decodes = []
        offset = plan.receive_offset
        previous = None
        for w in windows:
            start, nbytes = table[w]
            if previous == w - 1:
                runs[-1][2] += nbytes
            else:
                runs.append([header_nbytes + start, offset, nbytes])
            decodes.append((w, offset, nbytes))
            offset += nbytes
            previous = w
        transfers = [
            (
                plan.receive_memory.to_mutable_view(local, nbytes),
                plan.remote_buffer.slice(remote, nbytes),
            )
            for remote, local, nbytes in runs
        ]
        return transfers, decodes

    def _apply_deltas(
        self, deltas: list[tuple[_DeltaPlan, list[tuple[int, int, int]]]]
    ) -> None:
        """Decompress each received window on the GPU and XOR it into the parts of
        the prefetch buffers it covers."""
        codec = self._codec
        receive = self._receive
        assert codec is not None and receive is not None
        jobs = [
            (plan, w, offset, nbytes)
            for plan, decodes in deltas
            for w, offset, nbytes in decodes
        ]
        max_compressed = max(nbytes for *_, nbytes in jobs)
        with codec.on_stream(), torch.no_grad():
            # Window i loads into slot i % 2 on the H2D stream while window i - 1
            # decodes on the codec stream and window i - 2 stores on the D2H
            # stream. Every buffer outlives the final synchronize.
            slots = [
                _PipelineSlot(codec.device, max_compressed, _DELTA_WINDOW_NBYTES)
                for _ in range(2)
            ]
            xor_scratch = torch.empty(
                _DELTA_WINDOW_NBYTES, dtype=torch.uint8, device=codec.device
            )
            for i, (plan, w, offset, nbytes) in enumerate(jobs):
                slot = slots[i % 2]
                size = (
                    min((w + 1) * _DELTA_WINDOW_NBYTES, plan.staging_nbytes)
                    - w * _DELTA_WINDOW_NBYTES
                )
                parts = plan.windows[w]
                compressed = slot.buffers[0][:nbytes]
                data = slot.buffers[1][:size]
                # The slot is free once window i - 2 stored, which was after its
                # decode read the compressed bytes.
                codec.h2d.wait_event(slot.done)
                with torch.cuda.stream(codec.h2d):
                    compressed.copy_(
                        receive[offset : offset + nbytes], non_blocking=True
                    )
                    for start, view in parts:
                        data[start : start + view.numel()].copy_(
                            view, non_blocking=True
                        )
                    slot.loaded.record()
                codec.stream.wait_event(slot.loaded)
                xor = xor_scratch[:size]
                codec.decode(compressed, xor)
                # Bytes outside the parts are garbage and stay on the GPU.
                data.view(torch.int64).bitwise_xor_(xor.view(torch.int64))
                slot.computed.record()
                codec.d2h.wait_event(slot.computed)
                with torch.cuda.stream(codec.d2h):
                    for start, view in parts:
                        view.copy_(
                            data[start : start + view.numel()], non_blocking=True
                        )
                    slot.done.record()
            codec.d2h.synchronize()

    def copy_prefetched_(self, state_dict: dict[str, torch.Tensor]) -> Iterable[str]:
        """Copy ``state_dict`` in place from the prefetch buffers, unpacking packed
        keys; or, for resident keys after a delta read, XOR the delta into them on
        the GPU. Returns its keys."""
        unpacked = False
        resident = {k: t for k, t in state_dict.items() if k in self._resident_keys}
        if resident and self._resident_deltas:
            assert (
                resident.keys() == self._resident_keys
            ), "resident keys load in one call"
            self._apply_resident_deltas(resident)
        with torch.no_grad():
            for key, tensor in state_dict.items():
                if key in resident and self._resident_deltas:
                    continue
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
        if resident:
            self._resident_deltas = []
        return state_dict.keys()

    def _apply_resident_deltas(self, targets: dict[str, torch.Tensor]) -> None:
        """Decompress the windows of the last read that cover resident keys and XOR
        them into ``targets``, which hold the push before it, on the GPU."""
        codec = self._codec
        receive = self._receive
        assert codec is not None and receive is not None
        start_s = time.perf_counter()
        jobs = [
            (plan, w, offset, nbytes)
            for plan, decodes in self._resident_deltas
            for w, offset, nbytes in decodes
        ]
        views = {k: _local(t).view(-1).view(torch.uint8) for k, t in targets.items()}
        # The caller's queued work, e.g. the last forward, may still read the
        # targets.
        caller = torch.cuda.current_stream(codec.device)
        with codec.on_stream(), torch.no_grad():
            codec.stream.wait_stream(caller)
            compressed_scratch = torch.empty(
                max(nbytes for *_, nbytes in jobs),
                dtype=torch.uint8,
                device=codec.device,
            )
            xor_scratch = torch.empty(
                _DELTA_WINDOW_NBYTES, dtype=torch.uint8, device=codec.device
            )
            for plan, w, offset, nbytes in jobs:
                size = (
                    min((w + 1) * _DELTA_WINDOW_NBYTES, plan.staging_nbytes)
                    - w * _DELTA_WINDOW_NBYTES
                )
                compressed = compressed_scratch[:nbytes]
                compressed.copy_(receive[offset : offset + nbytes], non_blocking=True)
                xor = xor_scratch[:size]
                codec.decode(compressed, xor)
                for start, key, byte_offset, n in plan.resident_windows[w]:
                    views[key][byte_offset : byte_offset + n].bitwise_xor_(
                        xor[start : start + n]
                    )
            codec.stream.synchronize()
        logger.info(
            f"XORed push {self._version} into {len(targets)} resident tensors on "
            f"the GPU in {time.perf_counter() - start_s:.3f} s"
        )
