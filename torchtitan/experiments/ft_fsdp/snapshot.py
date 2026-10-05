# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-memory snapshots of sharded training state.

Each rank keeps ``num_local`` pinned CPU slots holding its parameter and
optimizer shards plus pickled CPU state. After every snapshot step the copy
runs on a side stream from a worker thread, so the training loop only pays
for recording an event. The next optimizer step waits on the copy's event
before mutating the tensors.

Committed slots are replicated with ``torch.distributed._transport`` to the
same local rank on the next host (rank ``r`` -> rank ``(r + G) % W``), which
holds two replica slots. A write invalidates the replica header, writes the
body, then writes a valid header, so a reader never sees a torn snapshot.
"""

import logging
import pickle
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_MAGIC = 0x46544653
_HEADER_WORDS = 4  # magic, step, meta_len, nbytes
_ALIGN = 64


@dataclass
class _Slot:
    body: torch.Tensor
    step: int = -1
    meta_len: int = 0
    reserved: bool = False


@dataclass
class _Job:
    step: int
    slot: int
    meta: bytes
    ready: torch.cuda.Event
    epoch: int
    launched: threading.Event = field(default_factory=threading.Event)
    copy_done: torch.cuda.Event | None = None


@dataclass
class _LinkOut:
    peer: str
    transport: object
    body_mems: list
    header_src: torch.Tensor
    header_mem: object
    remote_headers: list
    remote_bodies: list


@dataclass
class _LinkIn:
    peer: str
    transport: object


def _local(t: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return t._local_tensor if isinstance(t, DTensor) else t


def collect_state_tensors(
    model_parts: list[torch.nn.Module], optimizers: list[torch.optim.Optimizer]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (device tensors, CPU tensors) of parameters and optimizer state
    in a canonical order shared by all ranks of the same slot."""
    device_tensors: list[torch.Tensor] = []
    cpu_tensors: list[torch.Tensor] = []

    def add(t: torch.Tensor) -> None:
        t = _local(t)
        (cpu_tensors if t.device.type == "cpu" else device_tensors).append(t)

    for part in model_parts:
        for p in part.parameters():
            add(p.detach())
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state[p]
                for key in sorted(state):
                    value = state[key]
                    if not isinstance(value, torch.Tensor):
                        raise ValueError(
                            f"unsupported non-tensor optimizer state {key!r}"
                        )
                    add(value)
    return device_tensors, cpu_tensors


class _StaleLinkError(Exception):
    """The link being written was replaced during recovery."""


class TransportPool:
    """NIXL transports created ahead of use.

    Creating a NIXL agent takes about 15 s on a host with many NICs, so a
    process starts creating them before it is assigned a slot and each link
    takes a ready one.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"transport pool size must be >= 1, got {size}")
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="ftfsdp-nixl")
        self._ready: queue.SimpleQueue[Future] = queue.SimpleQueue()
        for _ in range(size):
            self._refill()

    def _refill(self) -> None:
        self._ready.put(self._executor.submit(_new_nixl_transport))

    def get(self):
        fut = self._ready.get()
        self._refill()
        return fut.result()


def _new_nixl_transport():
    from torch.distributed._transport import new_transport

    return new_transport("nixl", "cpu")


def _bootstrap(tr, store: dist.Store, *, rank: int, peer_rank: int, timeout: float):
    """Exchange agent metadata like ``new_transport_rank``, on an existing
    transport. ``store`` must be unique to the link."""
    try:
        store.set(str(rank), tr.bind(timeout=timeout))
        store.wait([str(peer_rank)], _td(timeout))
        tr.connect(store.get(str(peer_rank)), timeout=timeout)
    except BaseException:
        Snapshotter._close_transport(tr)
        raise
    return tr


class Snapshotter:
    def __init__(
        self,
        tensors: list[torch.Tensor],
        *,
        transports: TransportPool,
        device: torch.device,
        ident: str,
        store: dist.Store,
        num_local: int,
        meta_capacity: int,
        interval: int,
        procs_per_host: int,
        timeout: float,
        comm_failed: Callable[[], bool],
    ) -> None:
        if num_local < 3:
            raise ValueError(f"num_local_snapshots must be >= 3, got {num_local}")
        self.tensors = tensors
        self.transports = transports
        self.device = device
        self.ident = ident
        self.store = store
        self.meta_capacity = meta_capacity
        self.interval = interval
        self.procs_per_host = procs_per_host
        self.timeout = timeout
        self.comm_failed = comm_failed

        offsets = []
        offset = meta_capacity
        for t in tensors:
            offset = (offset + _ALIGN - 1) // _ALIGN * _ALIGN
            offsets.append(offset)
            offset += t.numel() * t.element_size()
        self.nbytes = offset
        self._offsets = offsets

        start = time.perf_counter()
        self.slots = [
            _Slot(torch.empty(self.nbytes, dtype=torch.uint8, pin_memory=True))
            for _ in range(num_local)
        ]
        self._views = [self._make_views(s.body) for s in self.slots]
        self.replica_headers = [
            torch.zeros(_HEADER_WORDS, dtype=torch.int64, pin_memory=True)
            for _ in range(2)
        ]
        # Sized for the predecessor, whose FSDP shards may be smaller.
        self.replica_nbytes = self.nbytes
        self.replica_bodies = self._alloc_replicas(self.nbytes)
        logger.info(
            f"snapshot: {len(tensors)} tensors, {self.nbytes / 2**20:.1f} MiB per "
            f"slot, {num_local} local + 2 replica slots allocated in "
            f"{time.perf_counter() - start:.2f}s"
        )

        self.link_out: _LinkOut | None = None
        self.link_in: _LinkIn | None = None

        self.aborted = threading.Event()
        self._epoch = 0
        self._cv = threading.Condition()
        self._queue: list[_Job] = []
        self._busy = False
        self._replicating = False
        self._pending: _Job | None = None
        self._closed = False
        self.stats = {"committed": 0, "replicated": 0, "dropped": 0}
        self.last_copy_s = 0.0
        self.last_replicate_s = 0.0
        self._stream = torch.cuda.Stream(device=device)
        self._worker = threading.Thread(
            target=self._run, name="ftfsdp-snapshot", daemon=True
        )
        self._worker.start()

    @staticmethod
    def _alloc_replicas(nbytes: int) -> list[torch.Tensor]:
        return [
            torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)
        ]

    def _make_views(self, body: torch.Tensor) -> list[torch.Tensor]:
        views = []
        for t, off in zip(self.tensors, self._offsets):
            n = t.numel() * t.element_size()
            views.append(body[off : off + n].view(t.dtype).view(t.shape))
        return views

    # Training loop side.

    def register_optimizer_hooks(self, optimizers: list[torch.optim.Optimizer]):
        for opt in optimizers:
            opt.register_step_pre_hook(lambda *_: self.wait_copy_launched())

    def wait_copy_launched(self) -> None:
        """Order the current stream after the in-flight snapshot copy so the
        optimizer cannot overwrite tensors that are still being read."""
        job = self._pending
        if job is None:
            return
        if not job.launched.wait(self.timeout):
            raise TimeoutError(f"snapshot copy for step {job.step} not launched")
        self._pending = None
        if job.copy_done is not None:
            torch.cuda.current_stream().wait_event(job.copy_done)

    def capture(self, step: int, meta: bytes) -> None:
        """Snapshot the current state as ``step``. Must be called on the
        training thread after the optimizer step."""
        if len(meta) > self.meta_capacity:
            raise ValueError(
                f"snapshot metadata is {len(meta)} bytes, larger than "
                f"meta_capacity_bytes={self.meta_capacity}"
            )
        self.wait_copy_launched()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        with self._cv:
            free = [i for i, s in enumerate(self.slots) if not s.reserved]
            slot = min(free, key=lambda i: self.slots[i].step)
            self.slots[slot].reserved = True
            self.slots[slot].step = -1
            job = _Job(step=step, slot=slot, meta=meta, ready=ready, epoch=self._epoch)
            self._queue.append(job)
            self._cv.notify_all()
        self._pending = job

    def abort(self) -> None:
        self.aborted.set()

    def pause(self) -> None:
        """Drop queued snapshots and wait for the worker to go idle."""
        self.aborted.set()
        self._pending = None
        with self._cv:
            self._epoch += 1
            for job in self._queue:
                self.slots[job.slot].reserved = False
                job.launched.set()
                self.stats["dropped"] += 1
            self._queue.clear()
            # A write to a dead successor can block until the transport
            # timeout. Do not wait for it: update_links replaces every link.
            if not self._cv.wait_for(
                lambda: not self._busy or self._replicating, self.timeout
            ):
                raise TimeoutError("snapshot worker did not go idle")

    def resume(self) -> None:
        self.aborted.clear()

    def flush(self) -> None:
        """Wait for queued snapshots to commit and replicate."""
        with self._cv:
            if not self._cv.wait_for(
                lambda: not self._queue and not self._busy, self.timeout
            ):
                raise TimeoutError("snapshot worker did not drain")
        self._pending = None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        for link in (self.link_out, self.link_in):
            if link is not None:
                self._close_transport(link.transport)
        self.link_out = self.link_in = None

    # Worker side.

    def _run(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._queue or self._closed)
                if self._closed:
                    return
                job = self._queue.pop(0)
                self._busy = True
            try:
                self._process(job)
            except Exception:
                logger.exception(f"snapshot for step {job.step} failed")
                job.launched.set()
            finally:
                with self._cv:
                    self.slots[job.slot].reserved = False
                    self._busy = False
                    self._cv.notify_all()

    def _process(self, job: _Job) -> None:
        start = time.perf_counter()
        slot = self.slots[job.slot]
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(job.ready)
            torch._foreach_copy_(self._views[job.slot], self.tensors, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self._stream)
        job.copy_done = done
        job.launched.set()
        n = len(job.meta)
        slot.body[:n].copy_(torch.frombuffer(bytearray(job.meta), dtype=torch.uint8))
        done.synchronize()
        if self.aborted.is_set() or job.epoch != self._epoch or self.comm_failed():
            self.stats["dropped"] += 1
            return
        with self._cv:
            slot.step = job.step
            slot.meta_len = n
        self.stats["committed"] += 1
        self.last_copy_s = time.perf_counter() - start
        with self._cv:
            link = self.link_out
            if link is None:
                return
            self._replicating = True
        start = time.perf_counter()
        try:
            self._replicate(link, job.slot)
        except _StaleLinkError:
            pass
        finally:
            with self._cv:
                self._replicating = False
                stale = link is not self.link_out
        if stale:
            # Replaced by update_links during recovery; nobody else owns it.
            # An abandoned write may hold it until the transport times out.
            threading.Thread(
                target=self._close_transport, args=(link.transport,), daemon=True
            ).start()
            return
        self.last_replicate_s = time.perf_counter() - start
        self.stats["replicated"] += 1

    def _replica_index(self, step: int) -> int:
        return (step // self.interval) % 2

    def _write(self, link: _LinkOut, local, remote) -> None:
        """Write and wait, giving up once recovery replaces ``link``. A write
        to a dead successor may not fail before the transport timeout, and the
        next snapshot copy must not wait for it."""
        deadline = time.monotonic() + self.timeout
        work = link.transport.write(local, remote, async_op=True, timeout=self.timeout)
        while not work.is_completed():
            if link is not self.link_out:
                raise _StaleLinkError()
            if time.monotonic() > deadline:
                raise TimeoutError("snapshot replication timed out")
            time.sleep(0.001)
        work.wait()

    def _write_header(self, link: _LinkOut, index: int, words: list[int]) -> None:
        link.header_src.copy_(torch.tensor(words, dtype=torch.int64))
        self._write(link, link.header_mem.to_view(), link.remote_headers[index])

    def _replicate(self, link: _LinkOut, slot_index: int) -> None:
        slot = self.slots[slot_index]
        index = self._replica_index(slot.step)
        self._write_header(link, index, [0, 0, 0, 0])
        self._write(
            link, link.body_mems[slot_index].to_view(), link.remote_bodies[index]
        )
        self._write_header(link, index, [_MAGIC, slot.step, slot.meta_len, self.nbytes])

    # Recovery side. Callers must pause() first.

    def committed_steps(self) -> list[int]:
        return sorted(s.step for s in self.slots if s.step >= 0)

    def replica_steps(self) -> dict[int, tuple[int, int]]:
        """Valid replicas held for the predecessor: step -> (index, meta_len)."""
        out = {}
        for i, h in enumerate(self.replica_headers):
            magic, step, meta_len, nbytes = h.tolist()
            if magic == _MAGIC and nbytes == self.replica_nbytes:
                out[step] = (i, meta_len)
        return out

    def fetch_replica(self, step: int, index: int, meta_len: int) -> None:
        """Read the successor's replica of this rank into a local slot."""
        link = self.link_out
        assert link is not None
        slot_index = min(range(len(self.slots)), key=lambda i: self.slots[i].step)
        slot = self.slots[slot_index]
        slot.step = -1
        mem = link.body_mems[slot_index]
        link.transport.read(
            mem.to_mutable_view(), link.remote_bodies[index], timeout=self.timeout
        )
        slot.step = step
        slot.meta_len = meta_len

    def restore(self, step: int) -> bytes:
        """Copy the slot holding ``step`` into the training tensors and return
        its metadata. Newer local slots are invalidated."""
        matches = [i for i, s in enumerate(self.slots) if s.step == step]
        if not matches:
            raise RuntimeError(
                f"no local snapshot for step {step}: {self.committed_steps()}"
            )
        slot = self.slots[matches[0]]
        torch._foreach_copy_(self.tensors, self._views[matches[0]], non_blocking=True)
        torch.cuda.synchronize()
        for s in self.slots:
            if s.step > step:
                s.step = -1
        return bytes(slot.body[: slot.meta_len].numpy())

    def replicate_now(self, step: int) -> None:
        """Synchronously replicate ``step`` and invalidate the other replica."""
        link = self.link_out
        if link is None:
            return
        matches = [i for i, s in enumerate(self.slots) if s.step == step]
        assert matches, f"step {step} not committed"
        self._replicate(link, matches[0])
        other = 1 - self._replica_index(step)
        self._write_header(link, other, [0, 0, 0, 0])

    # Links.

    def update_links(
        self, *, rank: int, ident_of_rank: list[str], gen: int, timeout: float
    ) -> None:
        """Connect to the successor (writes) and predecessor (receives).

        A link whose peer is unchanged is kept: both ends see the same pair
        of idents, so they agree without talking, and memory registration
        (the slow part) is skipped. A changed link is rebuilt on a fresh
        transport because an in-flight write to a dead successor may still
        own the old one. Callers must pause() first. Replica contents are
        kept."""
        world = len(ident_of_rank)
        g = self.procs_per_host
        succ = pred = None
        if world > g:
            succ = ident_of_rank[(rank + g) % world]
            pred = ident_of_rank[(rank - g) % world]
        with self._cv:
            old_out = self.link_out
            # A write to a live successor finishes quickly; wait for it so
            # it cannot race the replicas written after recovery. link_out
            # stays set meanwhile so the worker does not close it as stale.
            keep_out = (
                old_out is not None
                and old_out.peer == succ
                and self._cv.wait_for(lambda: not self._replicating, self.timeout)
            )
            self.link_out = None
            busy_out = self._replicating
        if old_out is not None and not keep_out and not busy_out:
            self._close_transport(old_out.transport)
        keep_in = self.link_in is not None and self.link_in.peer == pred
        if self.link_in is not None and not keep_in:
            self._close_transport(self.link_in.transport)
            self.link_in = None
        if world <= g:
            return
        start = time.perf_counter()
        with ThreadPoolExecutor(2) as pool:
            fut_out = (
                None if keep_out else pool.submit(self._connect_out, succ, gen, timeout)
            )
            fut_in = (
                None if keep_in else pool.submit(self._connect_in, pred, gen, timeout)
            )
            link_in = self.link_in if fut_in is None else fut_in.result()
            link_out = old_out if fut_out is None else fut_out.result()
        with self._cv:
            self.link_in = link_in
            self.link_out = link_out
        logger.info(
            f"snapshot links: -> {succ} (kept={keep_out}), <- {pred} "
            f"(kept={keep_in}) in {time.perf_counter() - start:.2f}s"
        )

    def _connect_in(self, pred: str, gen: int, timeout: float) -> _LinkIn:
        # Each link uses its own client: a blocking wait on a shared TCPStore
        # client stalls the other link's sets and deadlocks the ring. Keep a
        # reference: PrefixStore does not keep a Python store subclass alive.
        client = self.store.clone()
        store = dist.PrefixStore(f"ftfsdp/link/{gen}/{pred}->{self.ident}/", client)
        start = time.perf_counter()
        tr = _bootstrap(
            self.transports.get(), store, rank=1, peer_rank=0, timeout=timeout
        )
        bootstrap_s = time.perf_counter() - start
        store.wait(["nbytes"], _td(timeout))
        peer_s = time.perf_counter() - start - bootstrap_s
        nbytes = int(store.get("nbytes"))
        if nbytes != self.replica_nbytes:
            for h in self.replica_headers:
                h.zero_()
            self.replica_bodies = self._alloc_replicas(nbytes)
            self.replica_nbytes = nbytes
        headers = [tr.register_memory(t) for t in self.replica_headers]
        bodies = [tr.register_memory(t) for t in self.replica_bodies]
        register_s = time.perf_counter() - start - bootstrap_s - peer_s
        logger.info(
            f"snapshot link <- {pred}: bootstrap {bootstrap_s:.2f}s, wait peer "
            f"{peer_s:.2f}s, alloc+register {register_s:.2f}s"
        )
        store.set(
            "replica",
            pickle.dumps(
                (
                    [m.to_remote_buffer().serialize() for m in headers],
                    [m.to_remote_buffer().serialize() for m in bodies],
                )
            ),
        )
        return _LinkIn(peer=pred, transport=tr)

    def _connect_out(self, succ: str, gen: int, timeout: float) -> _LinkOut:
        from torch.distributed._transport.nixl._memory import NIXLRemoteBuffer

        client = self.store.clone()
        store = dist.PrefixStore(f"ftfsdp/link/{gen}/{self.ident}->{succ}/", client)
        start = time.perf_counter()
        tr = _bootstrap(
            self.transports.get(), store, rank=0, peer_rank=1, timeout=timeout
        )
        bootstrap_s = time.perf_counter() - start
        body_mems = [tr.register_memory(s.body) for s in self.slots]
        header_src = torch.zeros(_HEADER_WORDS, dtype=torch.int64, pin_memory=True)
        header_mem = tr.register_memory(header_src)
        register_s = time.perf_counter() - start - bootstrap_s
        store.set("nbytes", str(self.nbytes))
        store.wait(["replica"], _td(timeout))
        headers, bodies = pickle.loads(store.get("replica"))
        logger.info(
            f"snapshot link -> {succ}: bootstrap {bootstrap_s:.2f}s, register "
            f"{register_s:.2f}s, wait peer "
            f"{time.perf_counter() - start - bootstrap_s - register_s:.2f}s"
        )
        return _LinkOut(
            peer=succ,
            transport=tr,
            body_mems=body_mems,
            header_src=header_src,
            header_mem=header_mem,
            remote_headers=[NIXLRemoteBuffer.deserialize(b) for b in headers],
            remote_bodies=[NIXLRemoteBuffer.deserialize(b) for b in bodies],
        )

    @staticmethod
    def _close_transport(tr) -> None:
        try:
            tr.close(timeout=5.0)
        except Exception as e:
            logger.warning(f"closing snapshot transport failed: {e}")


def _td(seconds: float):
    from datetime import timedelta

    return timedelta(seconds=seconds)
