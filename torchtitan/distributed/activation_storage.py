# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager execution of a CheckpointPolicy for each tensor autograd saves."""

from __future__ import annotations

import contextlib
import weakref
from collections import defaultdict
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch_remat as remat
from torch.utils.checkpoint import CheckpointPolicy

# (pipeline stage, micro-batch, layer): the unit that is saved together and read back together.
ChunkKey = tuple[int, int, int]
SavePolicy = Callable[[torch.Tensor, ChunkKey], CheckpointPolicy]

_SAVE = frozenset({CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE})
_OFFLOAD = frozenset(
    {CheckpointPolicy.MUST_CPU_OFFLOAD, CheckpointPolicy.PREFER_CPU_OFFLOAD}
)


class StorageBackend(Protocol):
    """Where an offloaded save waits for backward."""

    def put(self, tensor: torch.Tensor, stream: torch.cuda.Stream | None) -> Any:
        """Start copying ``tensor`` out; ``None`` when there is no room for it."""
        ...

    def get(
        self, payload: Any, out: torch.Tensor, stream: torch.cuda.Stream | None
    ) -> None:
        ...

    def free(self, payload: Any) -> None:
        ...


class HostBackend:
    """Pinned host memory, copied on the storage stream, holding at most ``capacity_bytes``."""

    def __init__(self, capacity_bytes: int | None = None) -> None:
        self._capacity = capacity_bytes
        self._held = 0

    def put(
        self, tensor: torch.Tensor, stream: torch.cuda.Stream | None
    ) -> torch.Tensor | None:
        nbytes = tensor.numel() * tensor.element_size()
        if self._capacity is not None and self._held + nbytes > self._capacity:
            return None
        self._held += nbytes
        pinned = tensor.is_cuda
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=pinned)
        with torch.cuda.stream(
            stream
        ) if stream is not None else contextlib.nullcontext():
            host.copy_(tensor, non_blocking=pinned)
        return host

    def get(
        self, payload: torch.Tensor, out: torch.Tensor, stream: torch.cuda.Stream | None
    ) -> None:
        with torch.cuda.stream(
            stream
        ) if stream is not None else contextlib.nullcontext():
            out.copy_(payload, non_blocking=out.is_cuda)

    def free(self, payload: torch.Tensor) -> None:
        # The caching host allocator keeps a pinned block until the copies that read it finish.
        self._held -= payload.numel() * payload.element_size()


def cpu_offload_all(skip_layers: Collection[int] = ()) -> SavePolicy:
    """``MUST_CPU_OFFLOAD`` for every save outside ``skip_layers``."""

    def policy(tensor: torch.Tensor, chunk: ChunkKey) -> CheckpointPolicy:
        if chunk[2] in skip_layers:
            return CheckpointPolicy.PREFER_SAVE
        return CheckpointPolicy.MUST_CPU_OFFLOAD

    return policy


@dataclass
class _Saved:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    source: weakref.ref
    tensor: torch.Tensor | None = None
    backend: str | None = None
    payload: Any = None
    ready: torch.cuda.Event | None = None
    readers: int = 0


@dataclass
class _Ref:
    saved: _Saved
    chunk: ChunkKey


class ActivationStorage:
    """Applies ``policy`` to the tensors a stage's forward saves and reads the offloaded ones
    back ``prefetch_n_layers`` layers ahead in backward.

    A ``{MUST,PREFER}_SAVE`` save stays with autograd; a ``{MUST,PREFER}_CPU_OFFLOAD`` save goes
    to the backend ``route`` names for its chunk, host memory by default, and stays on the device
    when that backend is full. A pinned storage stays on the device whatever the policy says.
    Every device allocation happens on the compute stream; a tensor copied out stays referenced
    until its copy finishes instead of being handed to the storage stream's allocator.
    """

    def __init__(
        self,
        device: torch.device,
        policy: SavePolicy,
        backends: dict[str, StorageBackend] | None = None,
        *,
        route: Callable[[ChunkKey], str] | None = None,
        min_tensor_bytes: int = 1 << 20,
        prefetch_n_layers: int = 1,
    ) -> None:
        if prefetch_n_layers < 1:
            raise ValueError(
                f"prefetch_n_layers must be positive, got {prefetch_n_layers}"
            )
        # pyrefly: ignore [read-only]
        self._device = device
        self._policy = policy
        self._backends: dict[str, StorageBackend] = (
            backends if backends is not None else {"host": HostBackend()}
        )
        self._route = route or (lambda chunk: "host")
        self._min_bytes = min_tensor_bytes
        self._lookahead = prefetch_n_layers
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._stage_mb: tuple[int, int] | None = None
        self._layer: int | None = None
        self._keep: set[int] = set()
        self._by_id: dict[int, _Saved] = {}
        self._chunks: dict[ChunkKey, list[_Ref]] = defaultdict(list)
        self._layers: dict[int, list[int]] = {}
        self._started: set[ChunkKey] = set()
        self._outgoing: list[tuple[torch.cuda.Event | None, torch.Tensor]] = []
        self._pending: list[tuple[_Saved, ChunkKey]] = []
        self._forward_refs: list[_Ref] = []
        self._pinned: dict[int, int] = {}
        self.stats: dict[str, float] = defaultdict(float)

    def pin(self, tensor: torch.Tensor) -> None:
        """Keep ``tensor``'s storage on the device until ``unpin``."""
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in self._pinned:
            self._pinned[storage.data_ptr()] = storage.nbytes()
            self.stats["pinned_bytes"] += storage.nbytes()
            self.stats["pinned_peak_bytes"] = max(
                self.stats["pinned_peak_bytes"], self.stats["pinned_bytes"]
            )

    def unpin(self, tensor: torch.Tensor) -> None:
        nbytes = self._pinned.pop(tensor.untyped_storage().data_ptr(), None)
        if nbytes is not None:
            self.stats["pinned_bytes"] -= nbytes

    def register_stage(self, stage: int, layers: torch.nn.Module) -> None:
        """Track which layer of ``stage`` is running by a pre-hook on each of its blocks."""
        ids = []
        for name, block in layers.named_children():
            layer = int(name)
            ids.append(layer)
            block.register_forward_pre_hook(self._make_layer_hook(layer))
        self._layers[stage] = sorted(ids)

    def _make_layer_hook(self, layer: int):
        def hook(module, args):
            self._layer = layer

        return hook

    @contextlib.contextmanager
    def forward(self, stage: int, mb: int, keep: Sequence[Any] = ()) -> Iterator[None]:
        """Run ``stage``'s forward of ``mb`` with its saves routed; ``keep`` stay on the device."""
        self._stage_mb = (stage, mb)
        self._layer = None
        self._keep = {id(t) for t in keep if isinstance(t, torch.Tensor)}
        self._by_id = {}
        try:
            with remat.saved_tensors_hooks(
                self._pack, self._unpack, capture_context=self._capture
            ):
                yield
            self._route_pending()
        finally:
            self._pending = []
            self._forward_refs = []
            self._stage_mb = None
            self._layer = None
            self._keep = set()
            self._by_id = {}

    def _capture(self) -> ChunkKey | None:
        if self._stage_mb is None or self._layer is None:
            return None
        return (*self._stage_mb, self._layer)

    def _pack(self, tensor: torch.Tensor) -> Any:
        self._drain()
        chunk = None
        with contextlib.suppress(RuntimeError):
            chunk = remat.current_saved_tensor_info().context
        chunk = chunk if chunk is not None else self._capture()
        if (
            chunk is None
            or id(tensor) in self._keep
            or tensor.device.type != self._device.type
            or tensor.untyped_storage().data_ptr() in self._pinned
            or not tensor.is_contiguous()
            or tensor.numel() == 0
            or tensor.numel() * tensor.element_size() < self._min_bytes
            # a view leaves its base allocated: moving it copies without freeing
            or tensor.storage_offset() != 0
            or tensor.untyped_storage().nbytes()
            != tensor.numel() * tensor.element_size()
        ):
            return tensor
        saved = self._by_id.get(id(tensor))
        if saved is None or saved.source() is not tensor:
            saved = _Saved(
                tensor.shape, tensor.dtype, tensor.device, weakref.ref(tensor), tensor
            )
            self._by_id[id(tensor)] = saved
            self._pending.append((saved, chunk))
        ref = _Ref(saved, chunk)
        self._forward_refs.append(ref)
        return ref

    def _route_pending(self) -> None:
        # Decided when the forward ends, so a storage pinned during it stays on the device.
        # A save that stays is left to autograd; only moved ones are tracked for read back.
        for saved, chunk in self._pending:
            tensor = saved.tensor
            if tensor is None or tensor.untyped_storage().data_ptr() in self._pinned:
                continue
            decision = self._policy(tensor, chunk)
            if decision in _SAVE:
                continue
            if decision not in _OFFLOAD:
                raise ValueError(
                    f"a saved tensor is saved or offloaded, not given {decision}"
                )
            name = self._route(chunk)
            if self._stream is not None:
                self._stream.wait_stream(torch.cuda.current_stream(self._device))
            payload = self._backends[name].put(tensor, self._stream)
            if payload is None:
                self.stats[f"{name}_full"] += 1
                continue
            saved.payload, saved.backend = payload, name
            done = None
            if self._stream is not None:
                done = torch.cuda.Event()
                done.record(self._stream)
            self._outgoing.append((done, tensor))
            saved.tensor = None
            self.stats[f"{name}_bytes"] += tensor.numel() * tensor.element_size()
        for ref in self._forward_refs:
            if ref.saved.backend is not None:
                ref.saved.readers += 1
                self._chunks[ref.chunk].append(ref)

    def _unpack(self, packed: Any) -> torch.Tensor:
        if not isinstance(packed, _Ref):
            return packed
        self._drain()
        chunk = packed.chunk
        if chunk not in self._started:
            self._started.add(chunk)
            self._start_chunk(chunk)
        saved = packed.saved
        if saved.tensor is None:
            self.stats["late_fetches"] += 1
            self._fetch(saved)
        if saved.ready is not None:
            torch.cuda.current_stream(self._device).wait_event(saved.ready)
        assert saved.tensor is not None
        return saved.tensor

    def _start_chunk(self, chunk: ChunkKey) -> None:
        # Backward reaches this layer: the one after it is done, the ones before it come next.
        stage, mb, layer = chunk
        layers = self._layers.get(stage, [])
        if layer in layers:
            pos = layers.index(layer)
            if pos + 1 < len(layers):
                self._release((stage, mb, layers[pos + 1]))
            for ahead in layers[max(pos - self._lookahead, 0) : pos]:
                self.prefetch((stage, mb, ahead))

    def prefetch(self, chunk: ChunkKey) -> None:
        """Start reading ``chunk``'s saves back onto the device."""
        self._drain()
        for ref in self._chunks.get(chunk, []):
            if ref.saved.tensor is None:
                self._fetch(ref.saved)

    def prefetch_first(self, stage: int, mb: int) -> None:
        """Start reading back the saves of the layers ``stage``'s backward of ``mb`` needs first."""
        for layer in self._layers.get(stage, [])[-self._lookahead :]:
            self.prefetch((stage, mb, layer))

    def _fetch(self, saved: _Saved) -> None:
        assert saved.backend is not None
        out = torch.empty(saved.shape, dtype=saved.dtype, device=saved.device)
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream(self._device))
        self._backends[saved.backend].get(saved.payload, out, self._stream)
        if self._stream is not None:
            saved.ready = torch.cuda.Event()
            saved.ready.record(self._stream)
        saved.tensor = out

    def _release(self, chunk: ChunkKey) -> None:
        for ref in self._chunks.pop(chunk, []):
            saved = ref.saved
            saved.readers -= 1
            if saved.readers == 0:
                saved.tensor = None
                if saved.backend is not None:
                    self._backends[saved.backend].free(saved.payload)
        self._started.discard(chunk)

    def finish(self, stage: int, mb: int) -> None:
        """Drop what ``stage``'s backward of ``mb`` read back."""
        self._drain()
        for chunk in [c for c in self._chunks if c[0] == stage and c[1] == mb]:
            self._release(chunk)
        self._started -= {c for c in self._started if c[0] == stage and c[1] == mb}

    def _drain(self) -> None:
        # A source copied out is freed on the compute stream once its copy has finished.
        self._outgoing = [
            (e, t) for e, t in self._outgoing if e is not None and not e.query()
        ]
