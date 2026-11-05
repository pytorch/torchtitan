# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch's `RDMAObject` (`rl/actors/rdma.py`): the owner registers a value's large tensors,
arrays and bytes as Monarch RDMA buffers, and a reader reads them all in one RDMA action. The
batcher sends trainer microbatches this way (`Batcher.Config.rdma_min_tensor_bytes`), and the
rollout workers send prompts (`RolloutWorker.Config.rdma_min_prompt_bytes`).
"""

from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import logging
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from monarch.rdma import RDMAAction, RDMABuffer
from torch.utils import _pytree as pytree

logger = logging.getLogger(__name__)

_ValueT = TypeVar("_ValueT")


@dataclasses.dataclass(frozen=True, slots=True)
class MonarchRDMAObject(Generic[_ValueT]):
    """A value whose large tensors, arrays and bytes stay in its owner, the process that
    registered it.

    Example (a microbatch: its 4 KiB ``input`` stays in the batcher, its 8-byte token counts go
    by value):
        rdma_microbatch = MonarchRDMAObject.register(microbatch, min_bytes=1024)  # owner
        microbatch = await rdma_microbatch.read()  # reader: one RDMA action reads every buffer
        await rdma_microbatch.drop()               # owner, after every reader has read it
    """

    value: _ValueT
    """The value, with each tensor, array and bytes leaf of at least `min_bytes` replaced by its
    buffer reference. `register` looks for leaves in dataclass fields and in the containers pytree
    flattens (dict, list, tuple, ...); a leaf inside any other object goes by value."""

    @classmethod
    def register(cls, value: _ValueT, *, min_bytes: int) -> MonarchRDMAObject[_ValueT]:
        def register_leaf(leaf: torch.Tensor | np.ndarray | bytes) -> Any:
            if isinstance(leaf, torch.Tensor):
                return _TensorRef.register(leaf) if leaf.nbytes >= min_bytes else leaf
            if isinstance(leaf, np.ndarray):
                return _ArrayRef.register(leaf) if leaf.nbytes >= min_bytes else leaf
            return _BytesRef.register(leaf) if len(leaf) >= min_bytes else leaf

        return cls(
            value=_map_leaves(value, (torch.Tensor, np.ndarray, bytes), register_leaf)
        )

    async def read(self) -> _ValueT:
        action = RDMAAction()
        # Tensors and arrays are filled in place when the action runs; bytes are copied out after.
        value = _map_leaves(
            self.value, _BUFFER_REFS, lambda ref: ref.queue_read(action)
        )
        await action.submit()
        return _map_leaves(value, _BytesRead, _BytesRead.to_bytes)

    async def drop(self) -> None:
        refs: list[_TensorRef | _ArrayRef | _BytesRef] = []
        _map_leaves(self.value, _BUFFER_REFS, refs.append)
        await asyncio.gather(*(ref.buffer.drop() for ref in refs))


@dataclasses.dataclass(frozen=True, slots=True)
class _TensorRef:
    """A tensor as its RDMA buffer, plus the shape and dtype to rebuild it."""

    buffer: RDMABuffer
    shape: torch.Size
    dtype: torch.dtype

    @classmethod
    def register(cls, tensor: torch.Tensor) -> _TensorRef:
        # A view, not a copy: the buffer keeps the tensor's storage alive until drop().
        # view(-1) needs a contiguous tensor; the batcher builds only contiguous ones.
        return cls(
            buffer=RDMABuffer(tensor.view(-1).view(torch.uint8)),
            shape=tensor.shape,
            dtype=tensor.dtype,
        )

    def queue_read(self, action: RDMAAction) -> torch.Tensor:
        # TODO: allocate pinned tensors here. With max_num_documents also set, that removed the
        # ~4 ms per microbatch in `TrainingEngine._forward_backward_body`'s TODO; pinned alone did not.
        tensor = torch.empty(self.shape, dtype=self.dtype)
        action.read_remote(tensor.view(-1).view(torch.uint8), self.buffer)
        return tensor


@dataclasses.dataclass(frozen=True, slots=True)
class _ArrayRef:
    """A numpy array as its RDMA buffer, plus the shape and dtype to rebuild it."""

    buffer: RDMABuffer
    shape: tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def register(cls, array: np.ndarray) -> _ArrayRef:
        # A view like a tensor's: from_numpy shares the array's memory. It needs a writable,
        # contiguous array, e.g. one np.asarray built from a list.
        return cls(
            buffer=RDMABuffer(torch.from_numpy(array).view(-1).view(torch.uint8)),
            shape=array.shape,
            dtype=array.dtype,
        )

    def queue_read(self, action: RDMAAction) -> np.ndarray:
        array = np.empty(self.shape, dtype=self.dtype)
        action.read_remote(
            torch.from_numpy(array).view(-1).view(torch.uint8), self.buffer
        )
        return array


@dataclasses.dataclass(frozen=True, slots=True)
class _BytesRef:
    """A bytes object as its RDMA buffer."""

    buffer: RDMABuffer

    @classmethod
    def register(cls, data: bytes) -> _BytesRef:
        # A copy: RDMABuffer takes a writable tensor, and bytes are read-only. The buffer keeps it
        # alive until drop().
        return cls(
            buffer=RDMABuffer(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        )

    def queue_read(self, action: RDMAAction) -> _BytesRead:
        destination = torch.empty(self.buffer.size(), dtype=torch.uint8)
        action.read_remote(destination, self.buffer)
        return _BytesRead(destination=destination)


@dataclasses.dataclass(frozen=True, slots=True)
class _BytesRead:
    """A bytes object being read into `destination`; `to_bytes` once the action has run."""

    destination: torch.Tensor

    def to_bytes(self) -> bytes:
        return self.destination.numpy().tobytes()


_BUFFER_REFS = (_TensorRef, _ArrayRef, _BytesRef)


def _map_leaves(
    value: Any, leaf_types: type | tuple[type, ...], fn: Callable[[Any], Any]
) -> Any:
    """Return a copy of `value` with each `leaf_types` instance replaced by `fn(instance)`,
    looking in dataclass fields and in the containers pytree flattens."""

    def map_leaf(leaf: Any) -> Any:
        if isinstance(leaf, leaf_types):
            return fn(leaf)
        if dataclasses.is_dataclass(leaf) and not isinstance(leaf, type):
            return dataclasses.replace(
                leaf,
                **{
                    field.name: _map_leaves(getattr(leaf, field.name), leaf_types, fn)
                    for field in dataclasses.fields(leaf)
                },
            )
        return leaf

    return pytree.tree_map(map_leaf, value)


# The CUDA driver's `CU_LIMIT_STACK_SIZE` (cuda.h).
_CU_LIMIT_STACK_SIZE = 0


def create_cuda_context_to_silence_monarch_log() -> None:
    """Make a CUDA context current on GPU 0, so Monarch stops printing `[RdmaXcel] No CUDA context ...`
    for every registered buffer (its CUDA driver check on each pointer needs a context).

    The context is current only on the calling thread: call it on the thread that registers.
    """
    # TODO: this costs ~256 MiB on GPU 0 (H100). Remove it when Monarch prints this line once, not per
    # buffer (https://github.com/meta-pytorch/monarch/issues/5049; per buffer in torchmonarch 0.7.0).
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError as error:  # e.g. GPU 0 is full or in exclusive-process mode
        logger.warning(
            f"No CUDA context, so Monarch logs one line per RDMA buffer: {error}"
        )
        return
    # The driver raises the stack limit when a kernel needs more, so 0 halves the context (520 -> 256 MiB).
    ctypes.CDLL("libcuda.so.1").cuCtxSetLimit(_CU_LIMIT_STACK_SIZE, ctypes.c_size_t(0))
