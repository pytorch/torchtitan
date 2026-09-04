# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Balancing activation memory across PP ranks (report sec on Mooncake offload).

Under interleaved 1F1B the resident activation load is uneven across PP ranks,
so heavy ranks park the tensors autograd saves for backward in a pool that
lives on a light rank's GPU, through the Mooncake Transfer Engine. The engine
picks its transport from the hardware: RDMA where an HCA exists, TCP where one
does not, so the same wiring runs on a workstation and on a cluster.

Mechanism: ``saved_tensors_hooks`` around the stage's forward. ``pack`` copies
a saved tensor into the remote pool and frees the local storage; ``unpack``
reads it back when backward needs it. Copies are exact, so the loss is bitwise
the unbalanced run.

``mooncake-transfer-engine`` is an optional dependency with the same standing
fla has; its wheel links CUDA 12's runtime, which ships as the
``nvidia-cuda-runtime-cu12`` wheel and is preloaded here so no environment
variable is needed.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


def _load_transfer_engine():
    """Import mooncake's TransferEngine, preloading the cu12 runtime it links."""
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
            raise ValueError(
                "pp_balance_source_ranks is set, which needs the "
                "mooncake-transfer-engine package (and its cu12 runtime, "
                "nvidia-cuda-runtime-cu12); import failed with: "
                f"{err}."
            ) from err
    return TransferEngine


@dataclass
class _Handle:
    """One parked tensor: where it sits in the pool and how to rebuild it."""

    offset: int
    nbytes: int
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


class _PoolAllocator:
    """First-fit free-list over the remote pool, coalescing on free.

    Parked tensors outlive each other in 1F1B order, so a bump pointer leaks;
    this stays O(handles) and handles are few (saved tensors of in-flight
    micro-batches).
    """

    def __init__(self, capacity: int) -> None:
        self._free: list[tuple[int, int]] = [(0, capacity)]

    def alloc(self, nbytes: int) -> int | None:
        # 512-byte alignment keeps peer RDMA writes well-formed.
        nbytes = (nbytes + 511) & ~511
        for i, (off, size) in enumerate(self._free):
            if size >= nbytes:
                if size == nbytes:
                    self._free.pop(i)
                else:
                    self._free[i] = (off + nbytes, size - nbytes)
                return off
        return None

    def free(self, offset: int, nbytes: int) -> None:
        nbytes = (nbytes + 511) & ~511
        self._free.append((offset, nbytes))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for off, size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((off, size))
        self._free = merged


class PPBalanceEngine:
    """Per-rank handle on the balance fabric.

    Every PP rank constructs one (the collectives that exchange the address
    book are group-wide). The dest rank allocates and registers the pool; the
    source ranks register a staging buffer they funnel transfers through, so
    per-tensor register/unregister churn stays off the hot path.
    """

    def __init__(
        self,
        pp_group,
        *,
        dest_rank: int,
        pool_bytes: int,
        staging_bytes: int,
        min_tensor_bytes: int,
    ) -> None:
        engine_cls = _load_transfer_engine()
        self._group = pp_group
        self._rank = dist.get_rank(pp_group)
        self._dest_rank = dest_rank
        self._min_tensor_bytes = min_tensor_bytes
        self._device = torch.device("cuda", torch.cuda.current_device())

        self._engine = engine_cls()
        # The engine picks its own RPC port regardless of the one requested;
        # a segment is addressed by host:that_port, so the address book shares
        # get_rpc_port(), not the string initialize() was given.
        rc = self._engine.initialize(
            f"127.0.0.1:{17000 + dist.get_rank()}", "P2PHANDSHAKE", "tcp", ""
        )
        if rc != 0:
            raise RuntimeError(f"mooncake TransferEngine initialize failed: rc={rc}")
        session = f"127.0.0.1:{self._engine.get_rpc_port()}"

        # The TCP transport serves host memory only; RDMA serves GPU memory
        # directly. Both sides of a transfer resolve this from their own
        # topology, which is identical across ranks of one homogeneous job.
        self._buffers_on_gpu = not os.environ.get("MC_FORCE_TCP") and any(
            len(hcas) > 0
            for entry in json.loads(self._engine.get_local_topology()).values()
            for hcas in entry
        )

        def _buffer(nbytes: int) -> torch.Tensor:
            if self._buffers_on_gpu:
                return torch.empty(nbytes, dtype=torch.uint8, device=self._device)
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

        pool_base = 0
        if self._rank == dest_rank:
            self._pool = _buffer(pool_bytes)
            pool_base = self._pool.data_ptr()
            _check(self._engine.register_memory(pool_base, pool_bytes))
        else:
            self._staging = _buffer(staging_bytes)
            _check(
                self._engine.register_memory(self._staging.data_ptr(), staging_bytes)
            )
            self._staging_bytes = staging_bytes

        book: list[tuple[str, int] | None] = [None] * dist.get_world_size(pp_group)
        dist.all_gather_object(book, (session, pool_base), group=pp_group)
        dest_entry = book[dest_rank]
        assert dest_entry is not None
        self._dest_session, self._pool_base = dest_entry
        self._alloc = _PoolAllocator(pool_bytes)
        self._outstanding = 0
        self._stats = {
            "park": 0,
            "park_mib": 0.0,
            "fetch": 0,
            "skip_small": 0,
            "skip_pool": 0,
        }
        import atexit

        atexit.register(
            lambda: print(f"[PPBAL] rank {self._rank} stats={self._stats}", flush=True)
        )
        logger.info(
            "PP balance: rank %d role=%s dest=%s pool=%d MiB",
            self._rank,
            "dest" if self._rank == dest_rank else "source",
            self._dest_session,
            pool_bytes >> 20,
        )

    # ---- source-side API -------------------------------------------------

    def park(self, t: torch.Tensor) -> _Handle | None:
        """Copy ``t`` into the remote pool; None means "keep it local"."""
        nbytes = t.numel() * t.element_size()
        # Contiguous only: a normalized layout keeps the values but changes the
        # strides the backward kernels tile by.
        if (
            not t.is_cuda
            or not t.is_contiguous()
            or nbytes < self._min_tensor_bytes
            or nbytes > self._staging_bytes
        ):
            self._stats["skip_small"] += 1
            return None
        offset = self._alloc.alloc(nbytes)
        if offset is None:
            self._stats["skip_pool"] += 1
            return None
        flat = t.detach().reshape(-1).view(torch.uint8)
        self._staging[:nbytes].copy_(flat)
        torch.cuda.current_stream().synchronize()
        rc = self._engine.transfer_sync_write(
            self._dest_session,
            self._staging.data_ptr(),
            self._pool_base + offset,
            nbytes,
        )
        if rc != 0:
            logger.warning(
                "PP balance: write to %s failed (rc=%d, %d bytes); keeping the "
                "tensor local.",
                self._dest_session,
                rc,
                nbytes,
            )
            self._alloc.free(offset, nbytes)
            return None
        self._outstanding += 1
        self._stats["park"] += 1
        self._stats["park_mib"] += nbytes / (1 << 20)
        return _Handle(offset, nbytes, t.shape, t.dtype, t.device)

    def fetch(self, h: _Handle) -> torch.Tensor:
        """Read a parked tensor back; frees its pool range."""
        rc = self._engine.transfer_sync_read(
            self._dest_session,
            self._staging.data_ptr(),
            self._pool_base + h.offset,
            h.nbytes,
        )
        if rc != 0:
            raise RuntimeError(f"mooncake read failed: rc={rc}")
        out = torch.empty(h.shape, dtype=h.dtype, device=h.device)
        out.reshape(-1).view(torch.uint8).copy_(self._staging[: h.nbytes])
        torch.cuda.current_stream().synchronize()
        self._alloc.free(h.offset, h.nbytes)
        self._outstanding -= 1
        self._stats["fetch"] += 1
        return out

    def hooks(self):
        """The saved-tensors hooks that route through this engine."""

        def pack(t: torch.Tensor):
            h = self.park(t)
            return t if h is None else h

        def unpack(obj):
            return self.fetch(obj) if isinstance(obj, _Handle) else obj

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _check(rc: int) -> None:
    if rc != 0:
        raise RuntimeError(f"mooncake register_memory failed: rc={rc}")


@dataclass(frozen=True)
class PPBalanceKnobs:
    """Which PP ranks park, where, and how much: ``pipeline_kimi_k3``'s ``pp_balance``."""

    pp_balance_source_ranks: tuple[int, ...] = ()
    pp_balance_dest_rank: int = -1
    pp_balance_pool_gib: float = 2.0
    pp_balance_staging_mib: int = 256
    pp_balance_min_tensor_mib: int = 1


def install_pp_balance(
    pp_schedule, pp_group, knobs: PPBalanceKnobs
) -> PPBalanceEngine | None:
    """Wire the balance hooks onto this rank's stages, per the knob record.

    Every rank in the group constructs the engine (the address-book exchange
    is a collective); only ranks named in ``pp_balance_source_ranks`` wrap
    their stages' forwards.
    """
    source_ranks = list(knobs.pp_balance_source_ranks)
    if not source_ranks:
        return None
    dest = knobs.pp_balance_dest_rank
    if dest in source_ranks:
        raise ValueError(
            f"pp_balance_dest_rank={dest} is also a source rank; the pool "
            "cannot live on a rank that parks into it."
        )
    engine = PPBalanceEngine(
        pp_group,
        dest_rank=dest,
        pool_bytes=int(knobs.pp_balance_pool_gib * (1 << 30)),
        staging_bytes=int(knobs.pp_balance_staging_mib * (1 << 20)),
        min_tensor_bytes=int(knobs.pp_balance_min_tensor_mib * (1 << 20)),
    )
    if engine._rank not in source_ranks:
        return engine

    wrapped = 0
    for stage in _stages_of(pp_schedule):
        orig = stage.forward_one_chunk

        def hooked(*args, _orig=orig, _eng=engine, **kwargs):
            with _eng.hooks():
                return _orig(*args, **kwargs)

        stage.forward_one_chunk = hooked
        wrapped += 1
    logger.info(
        "PP balance: rank %d parks saved tensors of %d stage(s) on rank %d.",
        engine._rank,
        wrapped,
        dest,
    )
    return engine


def _stages_of(schedule):
    stages = getattr(schedule, "_stages", None)
    if stages is None:
        stage = getattr(schedule, "_stage", None)
        stages = [stage] if stage is not None else []
    return stages
