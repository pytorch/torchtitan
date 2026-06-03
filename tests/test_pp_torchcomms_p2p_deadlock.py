# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reproducer for the pipeline-parallel send/recv deadlock with the torchcomms
communication backend (``--comm.mode torchcomms``).

Root cause
----------
Every PP schedule in ``torch.distributed.pipelining`` fuses a send and a recv
into a single *mixed* P2P batch (e.g. 1F1B fires ``fwd_sends + bwd_recvs`` via
``_batch_p2p`` -> ``torch.distributed.batch_isend_irecv``). ``batch_isend_irecv``
only coalesces the ops into one NCCL group (``ncclGroupStart/End``) when the
backend reports ``supports_coalescing == True``. The torchcomms
``_BackendWrapper`` reports ``supports_coalescing == False``, so
``batch_isend_irecv`` falls back to issuing each op *sequentially in list order*
on the single PP communicator: the send is enqueued first, the recv second.

When two neighboring PP ranks each enqueue their send before their recv on the
same in-order communicator stream, rank i's send waits for rank i+1 to post the
matching recv -- but that recv is queued *behind* rank i+1's own send, which is
in turn waiting for rank i. The result is a circular wait: a deadlock.

Fix
---
torchcomms' ``BackendWrapper`` implements P2P coalescing (``supportsCoalescing()``
-> true; ``startCoalescing``/``endCoalescing`` accumulate the ops into a
``BatchSendRecv`` and ``issue()`` them as one fused, deadlock-free batch). With a
torchcomms build that includes it, ``batch_isend_irecv`` takes the coalescing
path and the exchange below completes.

This test exercises the exact ``batch_isend_irecv`` path the schedules use, with
a symmetric (send-first) mixed batch between rank pairs:

* ``nccl``                  -> completes (coalesced into one NCCL group)
* ``torchcomms``            -> completes once the coalescing fix is built in;
                              deadlocks against a stale/pre-fix build
* ``torchcomms`` native     -> completes (``BatchSendRecv``, the primitive the
                              coalescing path routes to)

Cross-batch ordering hazard (coalescing is NOT sufficient)
----------------------------------------------------------
Coalescing only makes a *single* mixed batch safe. A single PP communicator is a
FIFO and a schedule issues many batches; when ranks reach their per-neighbour
exchanges in different relative orders (pipeline skew, looped / V schedules, skip
connections) the FIFO forms a dependency cycle and deadlocks *even though every
batch is coalesced*. This is backend-independent -- it deadlocks on nccl and
torchcomms alike (see the ring tests below).

The fix -- per-direction communicators (``torchtitan.distributed.pp_p2p``, behind
the ``TORCHTITAN_PP_PER_DIRECTION_P2P`` feature flag) -- puts forward (``down``)
and backward (``up``) P2P on separate communicators/streams, so neither can block
the other in a shared FIFO. With the flag on, the ring completes.

Run directly (pair cases need >= 2 GPUs; ring cases need >= 3):

    python tests/test_pp_torchcomms_p2p_deadlock.py --backend nccl
    python tests/test_pp_torchcomms_p2p_deadlock.py --backend torchcomms

Or via pytest (spawns subprocesses):

    pytest tests/test_pp_torchcomms_p2p_deadlock.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Make `torchtitan` importable when this file is run as a script (then sys.path[0]
# is the tests/ dir, not the repo root). Runs at import time, so spawn-launched
# child processes that re-import this module pick it up too.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _init_pg(
    backend_mode: str, rank: int, world_size: int, master_port: int = 29593
) -> torch.device:
    """Init a process group, optionally routing P2P through torchcomms."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    device_id = None
    if backend_mode == "torchcomms":
        import torchcomms  # noqa: F401
        import torch.distributed.config as dist_config

        dist_config.use_torchcomms = True
        # torchcomms requires a concrete device_id at init time.
        device_id = device

    import torch.distributed as dist

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device_id,
    )
    return device


def _run_mixed_batch_p2p(rank: int, world_size: int, *, native_batch: bool) -> None:
    """Mimic the schedule's mixed send+recv batch between neighbor ranks.

    Each rank pairs with its neighbor and builds a P2POp list with the *send
    first* and the recv second -- the same ordering ``_batch_p2p`` produces for
    ``fwd_sends + bwd_recvs``. With proper coalescing this is deadlock-free;
    without it, the symmetric send-first ordering deadlocks.

    When ``native_batch`` is set, the same exchange is issued through
    torchcomms' own ``BatchSendRecv`` primitive (``comm.batch_op_create()``),
    which fuses the send+recv into a single deadlock-free issue -- this is the
    primitive the backend coalescing path should route to (see the fix notes).
    """
    import torch.distributed as dist

    # Pair (0,1), (2,3), ... so every rank both sends and recvs.
    peer = rank ^ 1
    if peer >= world_size:
        return

    send_t = torch.full((1024,), float(rank), device="cuda")
    recv_t = torch.empty((1024,), device="cuda")

    if native_batch:
        comm = dist.group.WORLD._get_backend(send_t.device).get_comm()
        batch = comm.batch_op_create()
        batch.send(send_t, peer)
        batch.recv(recv_t, peer)
        batch.issue(async_op=True).wait()
    else:
        ops = [
            dist.P2POp(dist.isend, send_t, peer),  # send first ...
            dist.P2POp(dist.irecv, recv_t, peer),  # ... recv second (deadlock hazard)
        ]
        works = dist.batch_isend_irecv(ops)
        for w in works:
            w.wait()
    torch.cuda.synchronize()

    expected = float(peer)
    assert torch.allclose(recv_t, torch.full_like(recv_t, expected)), (
        f"rank {rank} got {recv_t[0].item()} expected {expected}"
    )


def _worker(
    rank: int, world_size: int, backend_mode: str, native_batch: bool = False
) -> None:
    import torch.distributed as dist

    _init_pg(backend_mode, rank, world_size)
    if rank == 0:
        how = "torchcomms BatchSendRecv" if native_batch else "batch_isend_irecv"
        print(f"[{backend_mode}] running mixed {how} ...", flush=True)
    _run_mixed_batch_p2p(rank, world_size, native_batch=native_batch)
    dist.barrier()
    if rank == 0:
        print(f"[{backend_mode}] COMPLETED (no deadlock)", flush=True)
    dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Cross-batch ordering hazard (coalescing does NOT cover this) + the fix.      #
# --------------------------------------------------------------------------- #
#
# Coalescing only makes a *single* mixed batch safe. A single PP communicator is
# a FIFO, and a schedule issues many batches; when ranks reach their per-neighbour
# exchanges in different relative orders, the FIFO forms a dependency cycle and
# deadlocks even though every batch is coalesced.
#
# We isolate that with an N-rank ring where each rank does a coalesced exchange
# with its next neighbour and then with its prev neighbour, each as a *separate*
# blocking batch_isend_irecv. For N>=3 the firsts form a cycle (rank r blocks on
# exch(next=r+1), whose match is r+1's *second* op, which is queued behind r+1's
# own first op exch(r+2), ...). This is backend-independent: it deadlocks on nccl
# and torchcomms alike, because the hazard is the single shared communicator.


def _run_ring_single_comm(rank: int, world_size: int) -> None:
    """Per-neighbour coalesced exchanges, in uniform [next, prev] order, on ONE
    comm. Each exchange is individually coalesced (safe), but the cross-batch
    FIFO cycle deadlocks for world_size >= 3."""
    import torch.distributed as dist

    grp = dist.group.WORLD
    nxt = (rank + 1) % world_size
    prv = (rank - 1) % world_size
    for peer in (nxt, prv):
        send_t = torch.full((256,), float(rank), device="cuda")
        recv_t = torch.empty((256,), device="cuda")
        ops = [
            dist.P2POp(dist.isend, send_t, peer, group=grp),
            dist.P2POp(dist.irecv, recv_t, peer, group=grp),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()  # block before the next exchange -> models a schedule action
    torch.cuda.synchronize()


def _run_ring_per_direction(rank: int, world_size: int) -> None:
    """The fix: split the same transfers across two per-direction communicators
    (down: r->r+1, up: r->r-1). Forward and backward P2P then live on separate
    comms/streams, so neither can block the other in a FIFO -> no cycle."""
    from torchtitan.distributed.pp_p2p import (
        bidirectional_exchange,
        build_pp_direction_groups,
    )

    nxt = (rank + 1) % world_size
    prv = (rank - 1) % world_size
    groups = build_pp_direction_groups(list(range(world_size)))

    send_next = torch.full((256,), float(rank), device="cuda")  # activation -> next
    send_prev = torch.full((256,), float(rank) + 0.5, device="cuda")  # grad -> prev
    recv_prev = torch.empty((256,), device="cuda")  # activation <- prev
    recv_next = torch.empty((256,), device="cuda")  # grad <- next

    bidirectional_exchange(
        groups,
        next_peer=nxt,
        prev_peer=prv,
        send_next=send_next,
        recv_prev=recv_prev,
        send_prev=send_prev,
        recv_next=recv_next,
    )
    torch.cuda.synchronize()

    # recv_prev carries prev's activation (prev's send_next == float(prv));
    # recv_next carries next's gradient   (next's send_prev == float(nxt)+0.5).
    assert torch.allclose(recv_prev, torch.full_like(recv_prev, float(prv))), (
        f"rank {rank} recv_prev={recv_prev[0].item()} expected {float(prv)}"
    )
    assert torch.allclose(recv_next, torch.full_like(recv_next, float(nxt) + 0.5)), (
        f"rank {rank} recv_next={recv_next[0].item()} expected {float(nxt) + 0.5}"
    )


def _ring_worker(
    rank: int, world_size: int, backend_mode: str, per_direction: bool
) -> None:
    import torch.distributed as dist

    from torchtitan.distributed.pp_p2p import per_direction_p2p_enabled

    _init_pg(backend_mode, rank, world_size, master_port=29603)
    use_fix = per_direction or per_direction_p2p_enabled()
    if rank == 0:
        which = "per-direction comms (fix)" if use_fix else "single comm"
        print(f"[{backend_mode}] ring cross-batch via {which} ...", flush=True)
    if use_fix:
        _run_ring_per_direction(rank, world_size)
    else:
        _run_ring_single_comm(rank, world_size)
    dist.barrier()
    if rank == 0:
        print(f"[{backend_mode}] ring COMPLETED (no deadlock)", flush=True)
    dist.destroy_process_group()


def _main_single_process() -> int:
    """Entry point for a single torchrun-launched process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nccl", "torchcomms"], required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _worker(rank, world_size, args.backend)
    return 0


def _watch(procs, timeout_s: float) -> str:
    """Join `procs` with a wall-clock timeout; classify the outcome."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(not p.is_alive() for p in procs):
            break
        time.sleep(0.5)

    alive = [p for p in procs if p.is_alive()]
    if alive:
        result = "deadlock"
    elif all(p.exitcode == 0 for p in procs):
        result = "completed"
    else:
        result = "crashed"
    for p in alive:
        p.terminate()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    return result


def _spawn_and_watch(
    backend_mode: str,
    world_size: int,
    timeout_s: float,
    native_batch: bool = False,
) -> str:
    """Spawn `world_size` pair-exchange workers; return the outcome."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(r, world_size, backend_mode, native_batch))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    return _watch(procs, timeout_s)


def _spawn_and_watch_ring(
    backend_mode: str,
    world_size: int,
    timeout_s: float,
    per_direction: bool,
) -> str:
    """Spawn `world_size` ring workers (cross-batch ordering); return outcome."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_ring_worker, args=(r, world_size, backend_mode, per_direction)
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    return _watch(procs, timeout_s)


# ----------------------------- pytest ----------------------------------------


def _require_n_gpus(n: int):
    import pytest

    if not torch.cuda.is_available() or torch.cuda.device_count() < n:
        pytest.skip(f"needs >= {n} GPUs")


def _require_2_gpus():
    _require_n_gpus(2)


def _require_torchcomms():
    try:
        import torchcomms  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("torchcomms not installed")


def test_nccl_mixed_p2p_completes():
    _require_2_gpus()
    result = _spawn_and_watch("nccl", world_size=2, timeout_s=60)
    assert result == "completed", "nccl mixed batch_isend_irecv should not deadlock"


def test_torchcomms_mixed_p2p_completes():
    """Regression guard for the PP send/recv deadlock with the torchcomms backend.

    The fix is torchcomms' ``BackendWrapper`` coalescing (``supportsCoalescing()``
    -> true, ``start/endCoalescing`` mapped to ``BatchSendRecv``), which makes
    ``batch_isend_irecv`` fuse the mixed send+recv batch into one issue instead
    of enqueuing send-before-recv on the single comm.

    Before that fix (or against a stale torchcomms build) this same exchange
    deadlocks -- exactly what the 1F1B schedule's ``fwd_sends + bwd_recvs`` hits.
    """
    _require_2_gpus()
    try:
        import torchcomms  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("torchcomms not installed")
    result = _spawn_and_watch("torchcomms", world_size=2, timeout_s=45)
    assert result == "completed", (
        "torchcomms mixed batch_isend_irecv deadlocked -- the BackendWrapper "
        "coalescing fix is missing or the installed torchcomms build is stale."
    )


def test_torchcomms_native_batch_completes():
    """The fix path: torchcomms' native BatchSendRecv handles the same mixed
    send+recv exchange without deadlock. Routing the backend coalescing path to
    this primitive is what resolves the bug.
    """
    _require_2_gpus()
    try:
        import torchcomms  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("torchcomms not installed")
    result = _spawn_and_watch(
        "torchcomms", world_size=2, timeout_s=45, native_batch=True
    )
    assert result == "completed", (
        "torchcomms native BatchSendRecv should issue mixed P2P deadlock-free"
    )


# --- cross-batch ordering hazard: coalescing is NOT sufficient ---------------


def test_ring_single_comm_deadlocks_nccl():
    """The cross-batch FIFO cycle is backend-independent: even nccl (which
    coalesces each exchange) deadlocks, because the hazard is the single shared
    communicator, not the backend."""
    _require_n_gpus(3)
    result = _spawn_and_watch_ring("nccl", 3, timeout_s=40, per_direction=False)
    assert result == "deadlock", (
        "expected the single-comm ring to deadlock (cross-batch ordering cycle)"
    )


def test_ring_single_comm_deadlocks_torchcomms():
    """Same cross-batch cycle with torchcomms -- coalescing (now built in) makes
    each exchange safe but does not prevent the inter-batch deadlock."""
    _require_n_gpus(3)
    _require_torchcomms()
    result = _spawn_and_watch_ring("torchcomms", 3, timeout_s=40, per_direction=False)
    assert result == "deadlock", (
        "expected the single-comm ring to deadlock even with coalescing"
    )


def test_ring_per_direction_comms_completes_torchcomms():
    """The fix (feature flag ON): per-direction communicators put forward and
    backward P2P on separate comms, removing the cross-batch cycle."""
    _require_n_gpus(3)
    _require_torchcomms()
    result = _spawn_and_watch_ring("torchcomms", 3, timeout_s=40, per_direction=True)
    assert result == "completed", (
        "per-direction comms should resolve the cross-batch ordering deadlock"
    )


def test_ring_per_direction_comms_completes_nccl():
    """The fix also resolves the deadlock on the nccl backend."""
    _require_n_gpus(3)
    result = _spawn_and_watch_ring("nccl", 3, timeout_s=40, per_direction=True)
    assert result == "completed", (
        "per-direction comms should resolve the cross-batch ordering deadlock"
    )


if __name__ == "__main__":
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "--backend" in sys.argv:
        # Launched by torchrun: act as a single rank.
        sys.exit(_main_single_process())

    # Standalone driver: run both backends and report.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["nccl", "torchcomms", "both"],
        default="both",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    backends = ["nccl", "torchcomms"] if args.backend == "both" else [args.backend]
    rc = 0
    for b in backends:
        res = _spawn_and_watch(b, args.world_size, args.timeout)
        print(f"==> backend={b:>11} batch_isend_irecv: {res.upper()}", flush=True)
        if b == "nccl" and res != "completed":
            rc = 1

    # Demonstrate the fix path: torchcomms native BatchSendRecv is deadlock-free.
    if "torchcomms" in backends:
        res = _spawn_and_watch(
            "torchcomms", args.world_size, args.timeout, native_batch=True
        )
        print(f"==> backend= torchcomms BatchSendRecv: {res.upper()}", flush=True)

    # Cross-batch ordering hazard (coalescing does NOT cover this) + the fix.
    if torch.cuda.device_count() >= 3:
        for b in backends:
            res = _spawn_and_watch_ring(b, 3, args.timeout, per_direction=False)
            print(f"==> backend={b:>11} ring single-comm: {res.upper()}", flush=True)
            res = _spawn_and_watch_ring(b, 3, args.timeout, per_direction=True)
            print(
                f"==> backend={b:>11} ring per-direction (FIX): {res.upper()}",
                flush=True,
            )
    else:
        print("(skipping ring cross-batch cases: need >= 3 GPUs)", flush=True)
    sys.exit(rc)
