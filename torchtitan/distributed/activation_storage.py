# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One storage for the tensors autograd saves: a backend per tensor, chosen by a policy."""

from __future__ import annotations

import contextlib
import ctypes
import glob
import json
import os
import socket
import time
import weakref
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist
import torch_remat as remat

from torchtitan.tools.utils import round_up

# (pipeline stage, micro-batch, layer): the unit that is saved together and read back together.
ChunkKey = tuple[int, int, int]


class StorageBackend(Protocol):
    """Where a saved tensor waits for backward."""

    def put(self, tensor: torch.Tensor, stream: torch.cuda.Stream | None) -> Any:
        ...

    def get(
        self, payload: Any, out: torch.Tensor, stream: torch.cuda.Stream | None
    ) -> None:
        ...

    def free(self, payload: Any) -> None:
        ...


class HostBackend:
    """Pinned host memory, copied on the storage stream."""

    def put(
        self, tensor: torch.Tensor, stream: torch.cuda.Stream | None
    ) -> torch.Tensor:
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
        pass


# The alignment a peer's RDMA write wants.
_ALIGNMENT = 512


class _PoolAllocator:
    """First-fit free list over one span of a pool, coalescing on free."""

    def __init__(self, base: int, capacity: int) -> None:
        self._free: list[tuple[int, int]] = [(base, capacity)]
        self._live: set[int] = set()

    def alloc(self, nbytes: int) -> int | None:
        nbytes = round_up(nbytes, _ALIGNMENT)
        for i, (off, size) in enumerate(self._free):
            if size >= nbytes:
                if size == nbytes:
                    self._free.pop(i)
                else:
                    self._free[i] = (off + nbytes, size - nbytes)
                self._live.add(off)
                return off
        return None

    def free(self, offset: int, nbytes: int) -> None:
        if offset not in self._live:
            raise RuntimeError(f"pool offset {offset} is freed twice")
        self._live.discard(offset)
        self._free.append((offset, round_up(nbytes, _ALIGNMENT)))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for off, size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((off, size))
        self._free = merged


def _load_transfer_engine():
    """Import mooncake's TransferEngine, preloading the CUDA 12 runtime its wheel links."""
    try:
        from mooncake.engine import TransferEngine  # pyrefly: ignore [missing-import]
    except ImportError:
        try:
            import nvidia

            for root in nvidia.__path__:
                for lib in glob.glob(
                    os.path.join(root, "cuda_runtime", "lib", "libcudart.so.12*")
                ):
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    break
            from mooncake.engine import (  # pyrefly: ignore [missing-import]
                TransferEngine,
            )
        except (ImportError, OSError) as err:
            raise ImportError(
                "a remote activation backend needs the mooncake-transfer-engine package "
                "(and its cu12 runtime, nvidia-cuda-runtime-cu12); import failed with: "
                f"{err}."
            ) from err
    return TransferEngine


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"mooncake {what} failed: rc={rc}")


class RemoteBackend:
    """A pool on another rank of ``group``, reached through mooncake's transfer engine.

    Built on every rank of the group. ``dests`` maps each source rank to the rank holding its
    pool and ``spans`` gives each source's bytes there; a destination's pool is its sources'
    spans laid end to end. Transfers run in order on the storage stream through a registered
    staging buffer; a full span or staging buffer keeps the tensor on the device.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        dests: dict[int, int],
        spans: dict[int, int],
        staging_bytes: int,
        device: torch.device,
    ) -> None:
        engine_cls = _load_transfer_engine()
        rank = dist.get_rank(group)
        if set(dests) & set(dests.values()):
            raise ValueError(f"a rank both parks and holds a pool: {dests}")
        spans = {src: round_up(n, _ALIGNMENT) for src, n in spans.items()}
        self._engine = engine_cls()
        host = os.environ.get("MC_LOCAL_HOSTNAME") or socket.gethostname()
        with socket.socket() as probe:
            probe.bind(("", 0))
            port = probe.getsockname()[1]
        _check(
            self._engine.initialize(f"{host}:{port}", "P2PHANDSHAKE", "tcp", ""),
            "initialize",
        )
        # The engine picks its own RPC port; peers address a segment by host:that_port.
        session = f"{host}:{self._engine.get_rpc_port()}"
        # The TCP transport serves host memory only; RDMA serves device memory directly.
        on_device = device.type == "cuda" and any(
            len(hcas) > 0
            for entry in json.loads(self._engine.get_local_topology()).values()
            for hcas in entry
        )

        def buffer(nbytes: int) -> torch.Tensor:
            if on_device:
                return torch.empty(nbytes, dtype=torch.uint8, device=device)
            return torch.empty(
                nbytes, dtype=torch.uint8, pin_memory=device.type == "cuda"
            )

        def sources_of(dest: int) -> list[int]:
            return sorted(src for src, dst in dests.items() if dst == dest)

        pool_base = 0
        self._pool: torch.Tensor | None = None
        if rank in dests.values():
            pool_bytes = sum(spans[src] for src in sources_of(rank))
            self._pool = buffer(pool_bytes)
            pool_base = self._pool.data_ptr()
            _check(
                self._engine.register_memory(pool_base, pool_bytes), "register_memory"
            )
        book: list[tuple[str, int] | None] = [None] * dist.get_world_size(group)
        dist.all_gather_object(book, (session, pool_base), group=group)
        self.parks = rank in dests
        self._staging: torch.Tensor | None = None
        if self.parks:
            dest = dests[rank]
            entry = book[dest]
            assert entry is not None
            self._dest_session, self._pool_base = entry
            before = sources_of(dest)[: sources_of(dest).index(rank)]
            base = sum(spans[src] for src in before)
            self._alloc = _PoolAllocator(base, spans[rank])
            self._staging = buffer(staging_bytes)
            _check(
                self._engine.register_memory(self._staging.data_ptr(), staging_bytes),
                "register_memory",
            )
            self._slots = _PoolAllocator(0, staging_bytes)
            self._busy: list[tuple[torch.cuda.Event | None, int, int]] = []
            # The first transfer opens the peer's segment, which can fail while its handshake
            # server is still coming up; open it here, retrying, instead of inside a forward.
            for _ in range(40):
                try:
                    opened = (
                        self._engine.transfer_sync_write(
                            self._dest_session,
                            self._staging.data_ptr(),
                            self._pool_base + base,
                            _ALIGNMENT,
                        )
                        == 0
                    )
                except RuntimeError:
                    opened = False
                if opened:
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError(
                    f"rank {rank} could not open the pool of rank {dest} at "
                    f"{self._dest_session}"
                )
        dist.barrier(group=group)

    def _slot(self, nbytes: int) -> int | None:
        waiting = []
        for event, offset, size in self._busy:
            if event is None or event.query():
                self._slots.free(offset, size)
            else:
                waiting.append((event, offset, size))
        self._busy = waiting
        slot = self._slots.alloc(nbytes)
        while slot is None and self._busy:
            event, offset, size = self._busy.pop(0)
            if event is not None:
                event.synchronize()
            self._slots.free(offset, size)
            slot = self._slots.alloc(nbytes)
        return slot

    def _retire(self, slot: int, nbytes: int, stream: torch.cuda.Stream | None) -> None:
        event = None
        if stream is not None:
            event = torch.cuda.Event()
            event.record(stream)
        self._busy.append((event, slot, nbytes))

    def _move(
        self,
        write: bool,
        slot: int,
        offset: int,
        nbytes: int,
        stream: torch.cuda.Stream | None,
    ) -> None:
        assert self._staging is not None
        local = self._staging.data_ptr() + slot
        remote = self._pool_base + offset
        if stream is not None:
            call = (
                self._engine.transfer_write_on_cuda
                if write
                else self._engine.transfer_read_on_cuda
            )
            call(self._dest_session, local, remote, nbytes, stream.cuda_stream)
        else:
            call = (
                self._engine.transfer_sync_write
                if write
                else self._engine.transfer_sync_read
            )
            _check(call(self._dest_session, local, remote, nbytes), "transfer")

    def put(self, tensor: torch.Tensor, stream: torch.cuda.Stream | None) -> Any:
        nbytes = tensor.numel() * tensor.element_size()
        offset = self._alloc.alloc(nbytes)
        if offset is None:
            return None
        slot = self._slot(nbytes)
        if slot is None:
            self._alloc.free(offset, nbytes)
            return None
        assert self._staging is not None
        with torch.cuda.stream(
            stream
        ) if stream is not None else contextlib.nullcontext():
            self._staging[slot : slot + nbytes].copy_(
                tensor.reshape(-1).view(torch.uint8), non_blocking=True
            )
        self._move(True, slot, offset, nbytes, stream)
        self._retire(slot, nbytes, stream)
        return offset, nbytes

    def get(
        self, payload: Any, out: torch.Tensor, stream: torch.cuda.Stream | None
    ) -> None:
        offset, nbytes = payload
        slot = self._slot(nbytes)
        if slot is None:
            raise RuntimeError(
                f"staging buffer too small for a {nbytes}-byte activation"
            )
        self._move(False, slot, offset, nbytes, stream)
        assert self._staging is not None
        with torch.cuda.stream(
            stream
        ) if stream is not None else contextlib.nullcontext():
            out.reshape(-1).view(torch.uint8).copy_(
                self._staging[slot : slot + nbytes], non_blocking=True
            )
        self._retire(slot, nbytes, stream)

    def free(self, payload: Any) -> None:
        self._alloc.free(*payload)


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
    """Routes saved tensors to backends in forward and reads them back one layer ahead in backward.

    Every device allocation happens on the compute stream; a tensor copied out stays referenced
    until its copy finishes instead of being handed to the storage stream's allocator. A pinned
    storage stays on the device whatever the policy says.
    """

    def __init__(
        self,
        device: torch.device,
        backends: dict[str, StorageBackend],
        policy: Callable[[torch.Tensor, ChunkKey], str | None],
        *,
        min_tensor_bytes: int = 1 << 20,
    ) -> None:
        self._device = device
        self._backends = backends
        self._policy = policy
        self._min_bytes = min_tensor_bytes
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
        self.counting = False
        self.chunk_bytes: Counter[ChunkKey] = Counter()

    def set_backends(self, backends: dict[str, StorageBackend]) -> None:
        self._backends = backends

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
        if not (self._backends or self.counting):
            return tensor
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
            if self.counting:
                self.chunk_bytes[chunk] += tensor.numel() * tensor.element_size()
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
            name = self._policy(tensor, chunk)
            if name is None:
                continue
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
        # Backward reaches this layer: the one after it is done, the one before it comes next.
        stage, mb, layer = chunk
        layers = self._layers.get(stage, [])
        if layer in layers:
            pos = layers.index(layer)
            if pos + 1 < len(layers):
                self._release((stage, mb, layers[pos + 1]))
            if pos > 0:
                self.prefetch((stage, mb, layers[pos - 1]))

    def prefetch(self, chunk: ChunkKey) -> None:
        """Start reading ``chunk``'s saves back onto the device."""
        self._drain()
        for ref in self._chunks.get(chunk, []):
            if ref.saved.tensor is None:
                self._fetch(ref.saved)

    def prefetch_first(self, stage: int, mb: int) -> None:
        """Start reading back the saves of the layer ``stage``'s backward of ``mb`` needs first."""
        layers = self._layers.get(stage)
        if layers:
            self.prefetch((stage, mb, layers[-1]))

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
