# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model state dict deltas on top of the public TorchStore state dict API.

Every version, the trainer puts the full model state dict, then a second state
dict of fixed-size uint8 rows, one per local shard, holding only the elements
whose wire bytes changed since the previous version:

    [header: int64 x 4][values: wire dtype x cap][indices: int32 x cap]

``indices`` are flat offsets into the trainer's local shard. The header holds
the number of changed elements (-1 when they do not fit in ``cap``, and on the
first version), the number of put versions, a random epoch per trainer
rank, and the index of the wire dtype in ``_WIRE_DTYPES``. A row is keyed by
its tensor's flat key and the global offsets of the shard, so replicas put the
same row key. Their epochs differ: this relies on TorchStore reading a key
from the same replica every time; otherwise generators read full tensors.

A generator looks up the shards the trainer published, reads the rows of those
overlapping its own slice, scatters the changed values, and reads the full
tensors whose rows overflowed or that it does not hold at the previous version.
Values are absolute, so applying a delta is bitwise exact.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torchstore as ts
from torch.distributed.tensor import DTensor, Replicate

# TODO: private; TorchStore computes shard offsets with it as well.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_utils.py
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset

logger = logging.getLogger(__name__)

# Row header: [count, version, epoch, index of the wire dtype in _WIRE_DTYPES].
_HEADER_DTYPE = torch.int64
_HEADER_BYTES = 4 * _HEADER_DTYPE.itemsize
_INDEX_DTYPE = torch.int32
_OVERFLOW = -1
# Elements compared per step; bounds accelerator temporaries to a few hundred MB.
_CHUNK_NUMEL = 1 << 24
_WIRE_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
)
# Bitwise comparison dtype per element size.
_BITS_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _delta_key(key: str) -> str:
    """TorchStore key of the rows of state dict ``key``."""
    return f"{key}_delta"


def _row_key(flat_key: str, offsets: tuple[int, ...]) -> str:
    return f"{flat_key}@{','.join(map(str, offsets))}"


def _new_epoch() -> int:
    """Random epoch distinguishing trainer instances that reuse versions."""
    return int.from_bytes(os.urandom(8), "little") >> 1


def _local(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def _offsets(value: torch.Tensor) -> tuple[int, ...]:
    """Global offsets of this rank's elements, as TorchStore registers them."""
    if not isinstance(value, DTensor):
        return (0,) * value.ndim
    mesh = value.device_mesh
    # TorchStore stores fully local DTensors as plain tensors.
    if mesh.size() == 1 or all(isinstance(p, Replicate) for p in value.placements):
        return (0,) * value.ndim
    _, offsets = _compute_local_shape_and_global_offset(
        value.shape, mesh.shape, mesh.get_coordinate(), value.placements
    )
    return tuple(offsets)


@dataclass
class _Row:
    """Trainer state for one local shard."""

    wire_dtype: torch.dtype
    capacity: int
    # Previous version in the wire dtype, in host memory.
    previous: torch.Tensor
    # Row that is put, in host memory.
    packed: torch.Tensor


class ModelStateDictDeltaPublisher:
    """Puts the full model state dict and its delta rows every version.

    Keeps a copy of the previous version in the wire dtype and the rows in
    host memory. Accelerator tensors are compared on their device, copying the
    previous version there in chunks of ``_CHUNK_NUMEL`` elements.

    Args:
        state_dict: Flat template with the same keys, shapes, placements and
            devices as every put version.
        key: TorchStore key of the full state dict.
        max_fraction: Fraction of a shard's elements that fit in its row. A
            version changing more elements sends the full tensor instead.
        transfer_dtype: Wire dtype of floating-point tensors, as in
            ``ts.put_state_dict``.
        preserve_dtype_keys: Keys that keep their own dtype, as in
            ``ts.put_state_dict``.
    """

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        key: str,
        max_fraction: float,
        transfer_dtype: torch.dtype | None,
        preserve_dtype_keys: frozenset[str],
    ) -> None:
        self._key = key
        self._transfer_dtype = transfer_dtype
        self._preserve_dtype_keys = preserve_dtype_keys
        self._version = 0
        self._epoch = _new_epoch()
        # Flat key -> (row key, row).
        self._rows: dict[str, tuple[str, _Row]] = {}
        for flat_key, value in state_dict.items():
            local = _local(value)
            if local.numel() == 0:
                continue
            if local.numel() > torch.iinfo(_INDEX_DTYPE).max:
                raise ValueError(f"{flat_key!r} has too many elements per shard")
            wire_dtype = local.dtype
            if (
                transfer_dtype is not None
                and flat_key not in preserve_dtype_keys
                and local.is_floating_point()
            ):
                wire_dtype = transfer_dtype
            if wire_dtype not in _WIRE_DTYPES:
                raise ValueError(f"unsupported dtype {wire_dtype} for {flat_key!r}")
            # Multiple of 8 keeps each region of a row aligned.
            capacity = 8 * max(1, math.ceil(max_fraction * local.numel() / 8))
            nbytes = _HEADER_BYTES + capacity * (
                wire_dtype.itemsize + _INDEX_DTYPE.itemsize
            )
            pin_memory = local.device.type != "cpu"
            self._rows[flat_key] = (
                _row_key(flat_key, _offsets(value)),
                _Row(
                    wire_dtype=wire_dtype,
                    capacity=capacity,
                    previous=torch.empty(
                        local.shape, dtype=wire_dtype, pin_memory=pin_memory
                    ),
                    packed=torch.zeros(
                        nbytes, dtype=torch.uint8, pin_memory=pin_memory
                    ),
                ),
            )

    async def put(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Put ``state_dict`` in full, then its delta against the previous put.

        Encoding runs synchronously on the calling thread and current stream.
        """
        self._encode(state_dict)
        # Full first: a generator that sees a row's version reads full tensors
        # of at least that version.
        await ts.put_state_dict(
            dict(state_dict),
            self._key,
            transfer_dtype=self._transfer_dtype,
            preserve_dtype_keys=self._preserve_dtype_keys,
        )
        await ts.put_state_dict(
            {row_key: row.packed for row_key, row in self._rows.values()},
            _delta_key(self._key),
        )

    @torch.no_grad()
    def _encode(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        try:
            devices = set()
            for flat_key, (_row_key, row) in self._rows.items():
                local = _local(state_dict[flat_key])
                self._encode_row(row, local.reshape(-1))
                if local.device.type != "cpu":
                    devices.add(local.device)
            # Not the whole device: callers may overlap work on other streams.
            for device in devices:
                torch.accelerator.current_stream(device).synchronize()
        except BaseException:
            # Some rows and previous copies may hold the new version: a new
            # epoch makes generators read full tensors.
            self._version = 0
            self._epoch = _new_epoch()
            raise
        self._version += 1

    def _encode_row(self, row: _Row, source: torch.Tensor) -> None:
        """Encode the flat ``source`` into ``row.packed``.

        Syncs the stream once per chunk to size the changed elements.
        """
        capacity = row.capacity
        wire_dtype = row.wire_dtype
        values_end = _HEADER_BYTES + capacity * wire_dtype.itemsize
        header = row.packed[:_HEADER_BYTES].view(_HEADER_DTYPE)
        values = row.packed[_HEADER_BYTES:values_end].view(wire_dtype)
        indices = row.packed[values_end:].view(_INDEX_DTYPE)
        bits_dtype = _BITS_DTYPES[wire_dtype.itemsize]
        previous = row.previous.view(-1)
        # Overflow on the first version.
        count = 0 if self._version > 0 else capacity + 1
        for start in range(0, source.numel(), _CHUNK_NUMEL):
            wire = source[start : start + _CHUNK_NUMEL].to(wire_dtype)
            host_previous = previous[start : start + wire.numel()]
            if count <= capacity:
                old = host_previous.to(wire.device, non_blocking=True)
                found = torch.nonzero(
                    wire.view(bits_dtype) != old.view(bits_dtype)
                ).squeeze(1)
                end = count + found.numel()
                if end <= capacity:
                    values[count:end].copy_(wire[found], non_blocking=True)
                    indices[count:end].copy_(
                        (found + start).to(_INDEX_DTYPE), non_blocking=True
                    )
                count = end
            # Stream-ordered after the read of the previous version above.
            host_previous.copy_(wire, non_blocking=True)
        header[0] = count if count <= capacity else _OVERFLOW
        header[1] = self._version + 1
        header[2] = self._epoch
        header[3] = _WIRE_DTYPES.index(wire_dtype)


@dataclass
class _Shard:
    """One published shard overlapping a generator's slice."""

    row_key: str
    offsets: tuple[int, ...]
    shape: tuple[int, ...]
    # Receive buffer, reused so transports register it once.
    row: torch.Tensor
    # (epoch, version) the generator holds from this shard, None if unknown.
    held: tuple[int, int] | None = None


class ModelStateDictDeltaReader:
    """Updates a state dict in place from the previous to the latest version.

    Pass the same destination tensors every call and write them only through
    this reader: the delta applies on top of what the previous call wrote.
    Pulls must not overlap puts.

    Args:
        key: TorchStore key of the full state dict.
    """

    def __init__(self, key: str) -> None:
        self._key = key
        # Flat key -> published shards overlapping the own slice, planned on
        # the first pull.
        self._shards: dict[str, list[_Shard]] | None = None

    async def pull(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Read the latest version into ``state_dict``; unpublished keys stay as is."""
        if self._shards is None:
            self._shards = await self._plan(state_dict)
        shards = self._shards
        rows = await ts.get_state_dict(
            _delta_key(self._key),
            {
                shard.row_key: shard.row
                for flat_shards in shards.values()
                for shard in flat_shards
            },
        )
        full = {}
        full_headers = {}
        changed = total = num_overflowed = 0
        for flat_key, flat_shards in shards.items():
            # Rows of one tensor may be at different versions, e.g. after one
            # trainer replica failed a put; each is checked separately.
            headers = [
                rows[shard.row_key][:_HEADER_BYTES].view(_HEADER_DTYPE).tolist()
                for shard in flat_shards
            ]
            if all(
                count != _OVERFLOW and shard.held == (epoch, version - 1)
                for shard, (count, version, epoch, _dtype) in zip(flat_shards, headers)
            ):
                self._apply(state_dict[flat_key], flat_shards, headers)
                changed += sum(count for count, *_ in headers)
                total += sum(math.prod(shard.shape) for shard in flat_shards)
                for shard, (_count, version, epoch, _dtype) in zip(
                    flat_shards, headers
                ):
                    shard.held = (epoch, version)
            else:
                num_overflowed += any(count == _OVERFLOW for count, *_ in headers)
                full[flat_key] = state_dict[flat_key]
                full_headers[flat_key] = headers
                # Unknown until the full read succeeds.
                for shard in flat_shards:
                    shard.held = None
        if total or num_overflowed:
            # Guides max_fraction: rows that fit changed this fraction.
            logger.info(
                "Applied deltas of %r: %.4f of elements changed, %d tensors overflowed",
                self._key,
                changed / max(total, 1),
                num_overflowed,
            )
        if not full:
            return
        logger.info(
            "Reading %d of %d tensors of %r in full", len(full), len(shards), self._key
        )
        await ts.get_state_dict(self._key, full, strict=False)
        for flat_key in full:
            for shard, header in zip(shards[flat_key], full_headers[flat_key]):
                shard.held = (header[2], header[1])

    async def _plan(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, list[_Shard]]:
        layout = await ts.get_state_dict_layout(self._key)
        row_layout = await ts.get_state_dict_layout(_delta_key(self._key))
        shards = {}
        for flat_key, value in state_dict.items():
            local = _local(value)
            if flat_key not in layout or local.numel() == 0:
                continue
            if not local.is_contiguous():
                raise ValueError(f"destination {flat_key!r} must be contiguous")
            own_offsets = _offsets(value)
            own_end = [o + s for o, s in zip(own_offsets, local.shape)]
            flat_shards = []
            for tensor_slice in layout[flat_key]:
                offsets = tuple(tensor_slice.offsets)
                shape = tuple(tensor_slice.local_shape)
                if math.prod(shape) == 0 or any(
                    o >= e or o + s <= own
                    for o, s, own, e in zip(offsets, shape, own_offsets, own_end)
                ):
                    continue
                row_key = _row_key(flat_key, offsets)
                [row_slice] = row_layout[row_key]
                flat_shards.append(
                    _Shard(
                        row_key=row_key,
                        offsets=offsets,
                        shape=shape,
                        row=torch.empty(row_slice.local_shape, dtype=torch.uint8),
                    )
                )
            if flat_shards:
                shards[flat_key] = flat_shards
        return shards

    def _apply(
        self,
        destination: torch.Tensor,
        shards: list[_Shard],
        headers: list[list[int]],
    ) -> None:
        own_offsets = _offsets(destination)
        destination = _local(destination)
        flat_destination = destination.view(-1)
        for shard, (count, _version, _epoch, dtype_index) in zip(shards, headers):
            if count == 0:
                continue
            wire_dtype = _WIRE_DTYPES[dtype_index]
            capacity = (shard.row.numel() - _HEADER_BYTES) // (
                wire_dtype.itemsize + _INDEX_DTYPE.itemsize
            )
            values_end = _HEADER_BYTES + capacity * wire_dtype.itemsize
            values = shard.row[_HEADER_BYTES:values_end].view(wire_dtype)[:count]
            indices = shard.row[values_end:].view(_INDEX_DTYPE)[:count].long()
            indices, values = _shard_to_destination(
                indices,
                values,
                shard.offsets,
                shard.shape,
                own_offsets,
                tuple(destination.shape),
            )
            flat_destination[indices.to(destination.device)] = values.to(
                device=destination.device, dtype=destination.dtype
            )


def _shard_to_destination(
    indices: torch.Tensor,
    values: torch.Tensor,
    shard_offsets: tuple[int, ...],
    shard_shape: tuple[int, ...],
    own_offsets: tuple[int, ...],
    own_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map shard-local flat ``indices`` to destination-local ones.

    Drops elements outside the destination.
    """
    if shard_offsets == own_offsets and shard_shape == own_shape:
        return indices, values
    # Shard-local flat offsets -> global coordinates -> own-local offsets.
    position = torch.stack(torch.unravel_index(indices, shard_shape), dim=1)
    position += torch.tensor(shard_offsets) - torch.tensor(own_offsets)
    inside = ((position >= 0) & (position < torch.tensor(own_shape))).all(dim=1)
    strides = torch.tensor(torch.empty(own_shape, device="meta").stride())
    return (position[inside] * strides).sum(dim=1), values[inside]
