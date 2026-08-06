#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone RL weight-sync scalability benchmark.

Estimates how long a torchstore weight-sync GET takes for large models when the GET
is staged through host memory -- the shape the RL generator uses under
``--generator.manual-cpu-stage-weight-sync``, i.e. wherever registering vLLM's live GPU
params as the RDMA destination is not an option.

This script is deliberately SELF-CONTAINED: it depends only on ``torch``, ``numpy``,
``torchstore``, ``monarch`` and the stdlib. It imports NOTHING from
``torchtitan.experiments.rl`` -- no generator, trainer, RL config, or model code --
so it probes the transport in isolation.

The mental model is "truck vs detour": the CPU hop is the cheap part. At ~500K
tensors the per-op overhead, not bandwidth, is what dominates the GET -- which is
why the hop costs little relative to the transfer it protects, and why a
single-size measurement cannot tell you anything (see the two-term fit below).

What it measures (the "truck"): the cross-node staged GET wall time at a few
synthetic (num_tensors, total_bytes) points, then fits the two-term model

    time(N, bytes) = N * per_op_overhead + bytes / bandwidth

and extrapolates to a real safetensors manifest (e.g. Kimi-K3: 497,220 tensors,
1.56 TB). One point cannot separate the per-op term from the bandwidth term -- that
is why the tiny-model 1.2-1.7 GB/s number seen in real runs is useless on its own,
and why this sweeps >=2 well-separated points.

Modes
-----
``--mode manifest``  Parse safetensors headers only (no tensor-data load) and print
                     the exact manifest. Runs anywhere, no cluster / GPU.

``--mode transport`` SPMD harness (torch.distributed + torchstore.spmd). Rank 0 on
                     node A PUTs a synthetic state dict; rank 0 of node B GETs it
                     (remote volume -> RDMA over the fabric), timing the staged GET
                     (host dst) + H2D copy, and attempting the direct-GPU-dst GET
                     (expected to error cross-node on GB300 -- caught and reported).
                     ``--data-transport`` selects which torchstore transport carries
                     that PUT/GET (default ``auto`` = torchstore's own availability
                     cascade), mirroring the RL controller's ``weight_sync_transport``
                     -- this is the fast, vLLM-free way to bisect a transport failure.
                     It is a different axis from ``--transport``, which selects
                     monarch's SPMD control-plane transport.
                     Launch under srun across 2 nodes; see the sbatch launcher
                     ``bench_weight_sync.sbatch`` next to this file.

Example (manifest only, no cluster):
    python -m torchtitan.experiments.rl.scripts.bench_weight_sync \\
        --mode manifest \\
        --model-dir <dir of *.safetensors shards>

Example (transport, under srun -N2):
    srun -N2 --ntasks-per-node=1 python -m \\
        torchtitan.experiments.rl.scripts.bench_weight_sync \\
        --mode transport --transport tcp --data-transport torchcomms \\
        --model-dir <dir of *.safetensors shards> \\
        --world-size-extrapolate 64

    Drop ``--data-transport`` (or pass ``auto``) to let torchstore resolve the data
    plane per transfer, which is the shape the RL loop runs in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import struct
import subprocess
import time
from dataclasses import dataclass, field

import torch


logger = logging.getLogger("bench_weight_sync")

# safetensors dtype string -> (torch dtype or None, element size in bytes). We size
# tensors from the header's data_offsets (exact on-disk bytes), so element size is
# only used to synthesize matching tensors; dtypes torch lacks map to None and are
# synthesized as a raw uint8 buffer of the right byte count.
_ST_DTYPE: dict[str, tuple[torch.dtype | None, int]] = {
    "F64": (torch.float64, 8),
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "BOOL": (torch.bool, 1),
    # FP8 packed MoE weights (Kimi-K3). torch names vary by version; fall back to a
    # uint8 byte buffer if unavailable so the manifest and byte counts stay exact.
    "F8_E4M3": (getattr(torch, "float8_e4m3fn", None), 1),
    "F8_E5M2": (getattr(torch, "float8_e5m2", None), 1),
}


# ---------------------------------------------------------------------------
# Manifest: parse safetensors shard headers only (no tensor-data load)
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    """Exact tensor manifest for a model, from safetensors headers only."""

    num_tensors: int = 0
    total_bytes: int = 0
    tensor_bytes: list[int] = field(default_factory=list)
    dtype_count: dict[str, int] = field(default_factory=dict)
    dtype_bytes: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "Manifest") -> None:
        self.num_tensors += other.num_tensors
        self.total_bytes += other.total_bytes
        self.tensor_bytes.extend(other.tensor_bytes)
        for d, c in other.dtype_count.items():
            self.dtype_count[d] = self.dtype_count.get(d, 0) + c
        for d, b in other.dtype_bytes.items():
            self.dtype_bytes[d] = self.dtype_bytes.get(d, 0) + b


def parse_safetensors_header(path: str) -> Manifest:
    """Parse a single .safetensors shard's header (8-byte LE length + JSON).

    Reads only the header, never the tensor data. Byte sizes come from each
    entry's ``data_offsets`` (authoritative on-disk size).
    """
    m = Manifest()
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        nbytes = end - start
        dtype = entry["dtype"]
        m.num_tensors += 1
        m.total_bytes += nbytes
        m.tensor_bytes.append(nbytes)
        m.dtype_count[dtype] = m.dtype_count.get(dtype, 0) + 1
        m.dtype_bytes[dtype] = m.dtype_bytes.get(dtype, 0) + nbytes
    return m


def parse_model_dir(model_dir: str) -> Manifest:
    """Aggregate the manifest across every .safetensors shard in a directory."""
    shards = sorted(
        os.path.join(model_dir, n)
        for n in os.listdir(model_dir)
        if n.endswith(".safetensors")
    )
    if not shards:
        raise ValueError(f"no .safetensors shards found in {model_dir}")
    total = Manifest()
    for i, shard in enumerate(shards):
        total.merge(parse_safetensors_header(shard))
        logger.debug("parsed shard %d/%d: %s", i + 1, len(shards), shard)
    return total


def _fmt_bytes(n: float) -> str:
    # Decimal (1000-based) units to match torchstore's GB/s (nbytes / 1e9).
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000.0 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} TB"


def print_manifest(m: Manifest, model_dir: str) -> None:
    avg = m.total_bytes / m.num_tensors if m.num_tensors else 0
    tb = sorted(m.tensor_bytes)
    p50 = tb[len(tb) // 2] if tb else 0
    p99 = tb[int(len(tb) * 0.99)] if tb else 0
    print(f"\n=== Manifest: {model_dir} ===")
    print(f"  num_tensors : {m.num_tensors:,}")
    print(f"  total_bytes : {m.total_bytes:,} ({_fmt_bytes(m.total_bytes)})")
    print(
        f"  per-tensor  : avg {_fmt_bytes(avg)}, p50 {_fmt_bytes(p50)}, "
        f"p99 {_fmt_bytes(p99)}, min {_fmt_bytes(tb[0])}, max {_fmt_bytes(tb[-1])}"
    )
    print("  dtype histogram (count / bytes):")
    for d in sorted(m.dtype_count, key=lambda k: -m.dtype_bytes[k]):
        print(f"    {d:10s} {m.dtype_count[d]:>12,}  {_fmt_bytes(m.dtype_bytes[d])}")


# ---------------------------------------------------------------------------
# Synthetic state dicts for the transport sweep
# ---------------------------------------------------------------------------


@dataclass
class SizePoint:
    """One sweep point: ``num_tensors`` tensors summing to ~``total_bytes``."""

    num_tensors: int
    total_bytes: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def per_tensor_numel(self) -> int:
        elem = torch.empty(0, dtype=self.dtype).element_size()
        return max(1, self.total_bytes // (self.num_tensors * elem))


def default_points() -> list[SizePoint]:
    """Ascending tensor-count sweep at ~fixed bytes (isolates the per-op term as the
    slope vs N) with one higher-bytes anchor at a fixed N (isolates bandwidth). Kept
    ascending so the low-N points complete and record before any high-N point hits
    the torchstore RDMA work-request ceiling (a single get_batch posting ~100k WRs
    overflows the QP -> status=5 flush cascade); the first failing N locates that
    ceiling instead of aborting the run."""
    gb = 10**9
    return [
        SizePoint(num_tensors=128, total_bytes=4 * gb),
        SizePoint(num_tensors=2_048, total_bytes=4 * gb),
        SizePoint(num_tensors=8_192, total_bytes=4 * gb),
        SizePoint(num_tensors=8_192, total_bytes=16 * gb),  # bandwidth anchor
        SizePoint(num_tensors=32_768, total_bytes=4 * gb),
        SizePoint(num_tensors=65_536, total_bytes=4 * gb),  # probes the WR ceiling
    ]


def smoke_points() -> list[SizePoint]:
    """Tiny points for a single-node plumbing smoke test. Kept small enough that
    the same-host SHM staging path fits a modest /dev/shm (e.g. 64 MB sandboxes);
    the real cross-node run uses RDMA and is not /dev/shm bound."""
    mb = 10**6
    return [
        SizePoint(num_tensors=16, total_bytes=4 * mb),
        SizePoint(num_tensors=2_000, total_bytes=4 * mb),
        SizePoint(num_tensors=128, total_bytes=8 * mb),
    ]


def build_state_dict(point: SizePoint, device: str) -> dict[str, torch.Tensor]:
    """Allocate a synthetic state dict for a size point on ``device``.

    All tensors share the per-point element count so total bytes match the point;
    values are left uninitialized (empty) -- the transport moves bytes regardless.
    """
    numel = point.per_tensor_numel
    return {
        f"t{i}": torch.empty(numel, dtype=point.dtype, device=device)
        for i in range(point.num_tensors)
    }


# ---------------------------------------------------------------------------
# SPMD env glue (SLURM -> torchrun-style RANK/WORLD_SIZE/...)
# ---------------------------------------------------------------------------


def populate_env_from_slurm() -> None:
    """Fill torchrun-style env vars from SLURM_* when RANK is not already set.

    Lets the bench launch with a bare ``srun python -m ...`` (no torchrun). Master
    is the first node in the allocation; port is fixed and offset for the two
    TCPStores (barrier PG on MASTER_PORT, torchstore rendezvous on MASTER_PORT+1).
    """
    if "RANK" in os.environ:
        return
    if "SLURM_PROCID" not in os.environ:
        return  # not under SLURM; caller must set RANK/WORLD_SIZE/... itself
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["LOCAL_WORLD_SIZE"] = os.environ.get(
        "SLURM_NTASKS_PER_NODE", os.environ["SLURM_NTASKS"]
    )
    if "MASTER_ADDR" not in os.environ:
        nodelist = os.environ["SLURM_NODELIST"]
        first = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], text=True
        ).splitlines()[0]
        os.environ["MASTER_ADDR"] = first
    os.environ.setdefault("MASTER_PORT", "29610")


# ---------------------------------------------------------------------------
# Two-term fit + extrapolation
# ---------------------------------------------------------------------------


def _gbps(nbytes: float, seconds: float) -> float:
    return (nbytes / 1e9) / seconds if seconds > 0 else 0.0


def _point_bytes(point: SizePoint) -> int:
    elem = torch.empty(0, dtype=point.dtype).element_size()
    return point.per_tensor_numel * elem * point.num_tensors


def _short(e: Exception) -> str:
    """First line of an exception message, truncated -- some transport errors dump
    tens of thousands of per-op failure lines."""
    msg = str(e).strip().splitlines()
    return msg[0][:100] if msg else type(e).__name__


def fit_two_term(samples: list[tuple[int, int, float]]) -> tuple[float, float]:
    """Fit ``time = per_op * N + bytes / bandwidth`` to (N, bytes, seconds) samples.

    Returns ``(per_op_seconds, bandwidth_bytes_per_s)``. Needs >=2 samples with
    well-separated N/bytes ratios; otherwise the two terms are not identifiable.
    """
    import numpy as np

    if len(samples) < 2:
        raise ValueError("two-term fit needs >=2 samples")
    a = np.array([[n, b] for (n, b, _) in samples], dtype=float)
    y = np.array([t for (_, _, t) in samples], dtype=float)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    per_op, inv_bw = float(coef[0]), float(coef[1])
    bandwidth = 1.0 / inv_bw if inv_bw > 0 else float("inf")
    return per_op, bandwidth


def predict(per_op: float, bandwidth: float, num_tensors: int, nbytes: int) -> float:
    return per_op * num_tensors + nbytes / bandwidth


# ---------------------------------------------------------------------------
# Transport bench (SPMD)
# ---------------------------------------------------------------------------

# CLI value -> torchstore TransportType member name for the data-plane PUT/GET.
# Deliberately duplicated from _WEIGHT_SYNC_TRANSPORTS in rl/controller.py instead of
# imported: this file imports nothing from torchtitan.experiments.rl (see the module
# docstring), which is exactly what makes it usable to bisect torchstore with vLLM out
# of the picture. Member names rather than members so importing this module does not
# require torchstore (--mode manifest runs without it); resolved in _transport_main.
# "Unset" ("auto") leaves torchstore's per-transfer cascade in place; every other value
# pins all transfers, same-host included. SharedMemory is not offered: pinning it skips
# is_local_to_volume with no availability check behind it, so cross-host transfers would
# be routed through host-local shared memory.
_DATA_TRANSPORTS: dict[str, str] = {
    "auto": "Unset",
    "gloo": "Gloo",
    "monarch_rdma": "MonarchRDMA",
    "monarch_rpc": "MonarchRPC",
    "torchcomms": "TorchComms",
}


async def _transport_main(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    import torchstore as ts
    from torchstore.spmd import initialize as spmd_initialize, SPMDEnv
    from torchstore.strategy import LocalRankStrategy
    from torchstore.transport import TransportType

    env = SPMDEnv.from_env()
    rank, world, lws = env.rank, env.world_size, env.local_world_size
    num_hosts = world // lws

    # Barrier / gather process group on MASTER_PORT (CPU gloo; works on GB300 with
    # the SHM/RDMA transports disabled -- see the sbatch launcher's env).
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{env.master_addr}:{env.master_port}",
        rank=rank,
        world_size=world,
    )

    # torchstore rendezvous on a separate port so its TCPStore does not clash with
    # the barrier PG's store above.
    ts_env = SPMDEnv(
        rank=rank,
        local_rank=env.local_rank,
        world_size=world,
        local_world_size=lws,
        master_addr=env.master_addr,
        master_port=env.master_port + 1,
    )
    # default_transport_type rides on every StorageVolumeRef: "Unset" ("auto") keeps
    # torchstore's per-transfer resolution, a pin also covers the same-host PUT.
    data_transport = TransportType[_DATA_TRANSPORTS[args.data_transport]]
    if rank == 0:
        logger.info(
            "data plane: --data-transport %s -> TransportType.%s; "
            "control plane: --transport %s",
            args.data_transport,
            data_transport.name,
            args.transport,
        )
    await spmd_initialize(
        strategy=LocalRankStrategy(default_transport_type=data_transport),
        env=ts_env,
        transport=args.transport,
    )

    putter = 0
    # Rank 0 of the second host: forces the GET source volume to be remote so the
    # transport is real cross-node RDMA. Single-host runs fall back to a same-host
    # getter (SHM) -- a floor, not the cross-rail number; warn below.
    getter = lws if num_hosts >= 2 else (1 if world >= 2 else 0)
    if rank == getter and num_hosts < 2:
        logger.warning(
            "single host (num_hosts=%d): GET is same-host (SHM), NOT the cross-node "
            "RDMA transport. Launch with -N2+ for the real number.",
            num_hosts,
        )

    points = smoke_points() if args.smoke else default_points()
    samples: list[tuple[int, int, float]] = []  # (N, bytes, transport_seconds)
    records: list[dict] = []  # printed rows for the getter

    # Each point's PUT/GET is caught independently so one failure (e.g. the RDMA
    # work-request ceiling at high N) records a row instead of crashing the rank and
    # stranding its peer at the barrier. Both ranks hit both barriers every point.
    for i, point in enumerate(points):
        key = f"bench_sd_{i}"
        nbytes = _point_bytes(point)

        if rank == putter:
            try:
                src = build_state_dict(point, device=args.src_device)
                await ts.put_state_dict(src, key, direct_rdma=False)
            except Exception as e:  # noqa: BLE001 -- report, keep lockstep
                logger.warning("point %d PUT failed: %s", i, _short(e))
            finally:
                src = None
                if args.src_device.startswith("cuda"):
                    torch.cuda.empty_cache()
        dist.barrier()

        if rank == getter:
            rec = {
                "N": point.num_tensors,
                "bytes": nbytes,
                "transport_s": None,
                "h2d_s": None,
                "staged": "?",
            }
            # Staged GET: transport lands in host memory (the endpoint that sidesteps
            # the GPU registration failure); on success, H2D-copy into GPU params. This is the
            # fit data. The direct-GPU-dst GET is NOT run inline: it can crash the
            # shared StorageVolume actor (poisoning every later GET), so it is a
            # separate opt-in probe (--probe-direct-gpu) after the sweep.
            host_dst = build_state_dict(point, device="cpu")
            try:
                t0 = time.perf_counter()
                await ts.get_state_dict(
                    key, user_state_dict=host_dst, strict=False, direct_rdma=False
                )
                rec["transport_s"] = time.perf_counter() - t0
                rec["staged"] = "ok"
            except Exception as e:  # noqa: BLE001
                rec["staged"] = f"ERR {type(e).__name__}: {_short(e)}"

            if rec["staged"] == "ok":
                gpu_dst = build_state_dict(point, device="cuda")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for name, t in gpu_dst.items():
                        t.copy_(host_dst[name])
                torch.cuda.synchronize()
                rec["h2d_s"] = time.perf_counter() - t0
                del gpu_dst
                samples.append((point.num_tensors, nbytes, rec["transport_s"]))

            del host_dst
            torch.cuda.empty_cache()
            records.append(rec)
            logger.info(
                "point %d: N=%d bytes=%s staged=%s",
                i,
                rec["N"],
                _fmt_bytes(nbytes),
                rec["staged"],
            )
        dist.barrier()

    if rank == getter:
        _report(args, records, samples)

    # Optional final probe of the pre-staging direct-GPU-dst path. Run last, on a
    # dedicated key, because a GPU-dst RDMA GET can crash the StorageVolume actor;
    # keeping it after the staged sweep protects the fit data. Off by default.
    if args.probe_direct_gpu:
        await _probe_direct_gpu(ts, dist, rank, putter, getter)

    try:
        await ts.shutdown()
    except Exception as e:  # noqa: BLE001 -- best-effort teardown
        logger.warning("torchstore shutdown raised: %s", _short(e))
    dist.destroy_process_group()


async def _probe_direct_gpu(ts, dist, rank, putter, getter) -> None:
    """One-shot check of the pre-staging path: transport (direct_rdma=False) writing
    into a CUDA destination. Expected to fail cross-node on GB300 (GPU memory-region
    registration / volume-actor crash) -- reported, not fatal. Must be the last thing
    the run does, since it can leave the StorageVolume actor dead."""
    point = SizePoint(num_tensors=128, total_bytes=2 * 10**9)
    key = "bench_direct_probe"
    nbytes = _point_bytes(point)
    if rank == putter:
        try:
            src = build_state_dict(point, device="cpu")
            await ts.put_state_dict(src, key, direct_rdma=False)
            src = None
        except Exception as e:  # noqa: BLE001
            logger.warning("direct-probe PUT failed: %s", _short(e))
    dist.barrier()
    if rank == getter:
        print(
            f"\n=== direct-GPU-dst GET probe (N={point.num_tensors}, "
            f"{_fmt_bytes(nbytes)}, pre-staging path) ==="
        )
        gpu_dst = build_state_dict(point, device="cuda")
        try:
            t0 = time.perf_counter()
            await ts.get_state_dict(
                key, user_state_dict=gpu_dst, strict=False, direct_rdma=False
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(f"  ok: {dt:.3f}s ({_gbps(nbytes, dt):.2f} GB/s)")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {type(e).__name__}: {_short(e)}")
        del gpu_dst
        torch.cuda.empty_cache()
    dist.barrier()


def _report(args, records, samples) -> None:
    print("\n=== Step 1: cross-node staged GET (the 'truck') ===")
    print(
        f"  transport={args.transport} data_transport={args.data_transport}  "
        "getter measures remote-volume GET\n"
    )
    print(
        f"  {'num_tensors':>12} {'bytes':>12} {'transport':>11} {'GB/s':>8} "
        f"{'H2D copy':>10} {'H2D GB/s':>9}  staged GET"
    )
    for r in records:
        tr, h2d = r["transport_s"], r["h2d_s"]
        tr_s = f"{tr:>10.3f}s" if tr is not None else f"{'--':>11}"
        gbps = f"{_gbps(r['bytes'], tr):>8.2f}" if tr else f"{'--':>8}"
        h2d_s = f"{h2d:>9.3f}s" if h2d is not None else f"{'--':>10}"
        h2d_gbps = f"{_gbps(r['bytes'], h2d):>9.2f}" if h2d else f"{'--':>9}"
        print(
            f"  {r['N']:>12,} {_fmt_bytes(r['bytes']):>12} {tr_s} {gbps} "
            f"{h2d_s} {h2d_gbps}  {r['staged']}"
        )

    # A missing mapping means the PUT failed upstream, not the GET -- don't count
    # it toward the GET-side RDMA work-request ceiling.
    put_failed = [r for r in records if "Mapping is missing" in r["staged"]]
    get_failed = [
        r
        for r in records
        if r["staged"] != "ok" and "Mapping is missing" not in r["staged"]
    ]
    if get_failed:
        first = min(r["N"] for r in get_failed)
        smaller_ok = [r for r in records if r["staged"] == "ok" and r["N"] < first]
        if smaller_ok:
            # A work-request ceiling is by definition N-dependent, so it only earns
            # that name when a smaller batch went through on the same transport.
            print(
                f"\n  RDMA staged-GET ceiling: first GET failure at N={first:,} "
                f"tensors, with N={max(r['N'] for r in smaller_ok):,} still ok "
                "(single get_batch posts one work-request per tensor -> QP overflow)."
            )
        else:
            # Nothing succeeded, so nothing here is N-dependent and calling it a WR
            # ceiling would misattribute a blanket transport failure. The remedy is
            # transport-specific, so name the transport under test rather than
            # whichever one happened to be common when this was written.
            hints = {
                "monarch_rdma": "whether MONARCH_RDMA_IBVERBS_TARGET pins every "
                "endpoint into one plane (unpinned -> IBV_WC_RETRY_EXC_ERR, "
                "status=12/vendor_err=129 on every transfer)",
                "torchcomms": "whether NCCL_IB_GID_INDEX names a RoCE v2 GID with a "
                "ROUTABLE address; a link-local or RoCE v1 GID serves same-host "
                "transfers and then fails every cross-host one",
            }
            hint = hints.get(
                args.data_transport,
                "the transport itself -- every size failed, so this is not a "
                "batching or staging effect",
            )
            print(
                f"\n  every GET failed, smallest point (N={first:,}) included -- a "
                "blanket transport failure, NOT a work-request ceiling. For "
                f"--data-transport {args.data_transport}, check {hint}."
            )
    if put_failed:
        print(
            f"  note: {len(put_failed)} point(s) had no data (PUT failed upstream); "
            "not a GET ceiling."
        )

    if len(samples) < 2:
        print(
            f"\n  only {len(samples)} successful point(s) -- need >=2 for the fit; "
            "skipping fit/extrapolation."
        )
        return

    per_op, bw = fit_two_term(samples)
    print(
        "\n  two-term fit  time(N,bytes) = N*per_op + bytes/bandwidth "
        f"(over {len(samples)} successful points):"
    )
    print(f"    per_op_overhead : {per_op * 1e6:.2f} us/tensor")
    print(f"    bandwidth       : {bw / 1e9:.2f} GB/s")

    if args.model_dir:
        m = parse_model_dir(args.model_dir)
        w = max(1, args.world_size_extrapolate)
        n_r, b_r = m.num_tensors // w, m.total_bytes // w
        t_transport = predict(per_op, bw, n_r, b_r)
        t_perop = per_op * n_r
        t_bw = b_r / bw
        print(
            f"\n=== Extrapolation to {os.path.basename(args.model_dir.rstrip('/'))} "
            f"(per rank, world_size={w}) ==="
        )
        print(f"  per-rank shard : {n_r:,} tensors, {_fmt_bytes(b_r)}")
        print(f"  predicted staged transport GET : {t_transport:.1f}s")
        print(
            f"    per-op term   : {t_perop:.1f}s  ({100 * t_perop / t_transport:.0f}%)"
        )
        print(f"    bandwidth term: {t_bw:.1f}s  ({100 * t_bw / t_transport:.0f}%)")
        print(
            "  NOTE: ranks pull in parallel so wall-clock ~= per-rank time; this "
            "ignores fabric bandwidth contention across concurrent ranks."
        )


def run_transport(args: argparse.Namespace) -> None:
    populate_env_from_slurm()
    asyncio.run(_transport_main(args))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone RL weight-sync scalability benchmark."
    )
    p.add_argument(
        "--mode",
        choices=["manifest", "transport"],
        default="manifest",
        help="manifest: parse safetensors headers only (no cluster). "
        "transport: SPMD cross-node staged-GET timing (launch under srun).",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Directory of *.safetensors shards to build the exact manifest / "
        "extrapolation target (e.g. the Kimi-K3 checkpoint dir).",
    )
    p.add_argument(
        "--transport",
        default="tcp",
        choices=["ipc", "tcp", "metatls", "metatls-hostname"],
        help="torchstore SPMD worker transport -- monarch's control plane, NOT the "
        "transport the weights move over (that is --data-transport). 'ipc' is "
        "single-host only. 'tcp' "
        "maps to monarch's TcpWithHostname (works on this GB300/CoreWeave cluster); "
        "the 'metatls*' variants need Meta TLS infra that is absent here (the attach "
        "config push hangs -> MESH_ATTACH_CONFIG_TIMEOUT).",
    )
    p.add_argument(
        "--data-transport",
        default="auto",
        choices=sorted(_DATA_TRANSPORTS),
        help="torchstore data-plane transport for the weight PUT/GET -- the axis "
        "under test, distinct from --transport above (monarch's control plane). "
        "'auto' leaves torchstore's per-transfer availability cascade in place "
        "(SharedMemory same-host, then TorchComms, MonarchRDMA, Gloo cross-host); any "
        "other value pins EVERY transfer to that transport, same-host ones included. "
        "'torchcomms' is the only pin torchstore availability-checks (it raises unless "
        "USE_TORCHCOMMS and USE_TORCHCOMMS_RDMA are 1); the other pins ignore their "
        "TORCHSTORE_*_ENABLED gates, which only shape 'auto'. Same values as the RL "
        "controller's --weight-sync-transport.",
    )
    p.add_argument(
        "--src-device",
        default="cuda",
        help="Device for the PUT source state dict (mimics the trainer's GPU "
        "weights). 'cpu' to avoid GPU allocation on the putter.",
    )
    p.add_argument(
        "--world-size-extrapolate",
        type=int,
        default=64,
        help="Assumed generator world size for the per-rank extrapolation "
        "(shard = manifest / world_size).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny size points for a single-node plumbing smoke test.",
    )
    p.add_argument(
        "--probe-direct-gpu",
        action="store_true",
        help="After the staged sweep, run one direct-GPU-dst GET (pre-staging path). "
        "Off by default: it can crash the shared StorageVolume actor, so it runs last.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.mode == "manifest":
        if not args.model_dir:
            raise SystemExit("--mode manifest requires --model-dir")
        print_manifest(parse_model_dir(args.model_dir), args.model_dir)
    else:
        run_transport(args)


if __name__ == "__main__":
    main()
