#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extract per-step metrics from a PyTorch profiler Chrome trace.

Emits a JSON blob that a higher-level agent can diff against a second run's
output. Opinionated about PyTorch profiler traces (the output of
``torch.profiler.profile(...).export_chrome_trace``), which is what both
torchtitan and Megatron-LM produce when profiling is enabled.

Usage:
    python analyze_trace.py <trace.json|trace.json.gz> [--step N] [--output out.json]

If ``--step`` is omitted, the last ProfilerStep in the file is used, which is
the "active" step in a standard wait+warmup+active schedule.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# Kernel-name pattern tests run case-insensitively against the full kernel
# name. Ordered: first match wins. Keep these aligned with kernels that
# actually appear in torchtitan and Megatron traces on H100/B200.
_COMM_PATTERNS = (
    # Any NCCL kernel — covers ncclKernel_*, ncclDevKernel_*, and the newer
    # symmetric-memory variant ncclSymkDevKernel_* emitted by recent PyTorch
    # builds. Match before the reduction patterns so "ReduceScatter" in the
    # kernel name doesn't get misfiled as a reduction.
    "nccl",
    "c10d::",
    # Symmetric-memory collectives from torch.distributed._symmetric_memory
    # and NVSHMEM-backed kernels, which don't always carry the "nccl" prefix.
    "symm_mem",
    "nvshmem",
    # Peer-to-peer device copies: when Megatron is launched with
    # --use-nccl-ub (NCCL user-buffer registration + symmetric memory),
    # AllGather skips the traditional ring/tree kernel and issues direct
    # cudaMemcpyAsync peer copies between ranks. Those show up in the
    # Chrome trace as `Memcpy PtoP (Device -> Device)` under the
    # gpu_memcpy category. They are comm, not local memcpy — bucketing
    # them as memcpy under-reports comm.total_us and inflates apparent
    # overlap %. _categorize_kernel checks this pattern before the
    # gpu_memcpy early-return.
    "ptop",
)
_GEMM_PATTERNS = (
    "gemm",
    "cutlass",
    "cublas",
    "sgemm",
    "hgemm",
    "bf16gemm",
    "ampere_",
    "turing_",
    "hopper_",
    "blackwell_",
    "splitk_",
    "wgmma",
    "cudnn::gemm",
)
_ATTENTION_PATTERNS = (
    "flash",
    "fmha",
    "attention",
    "sdpa",
    "mha_",
)
_REDUCTION_PATTERNS = (
    "reduce",
    "softmax",
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "rms_norm",
    "cross_entropy",
)
_ELEMENTWISE_PATTERNS = (
    "elementwise",
    "vectorized_",
    "unrolled_",
    "activation",
    "gelu",
    "silu",
    "relu",
    "dropout",
    "copy_",
    "fill",
    "index_",
    "gather",
    "scatter",
)

CATEGORY_ORDER = (
    "comm",
    "gemm",
    "attention",
    "reduction",
    "elementwise",
    "memcpy",
    "other_kernel",
)


def _load_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return json.load(f)


def _categorize_kernel(name: str, cat: str) -> str:
    low = name.lower()
    # Check comm patterns before the gpu_memcpy early-return so peer-to-peer
    # copies (NCCL-UB / symmetric-memory AllGather path) get counted as comm.
    for pat in _COMM_PATTERNS:
        if pat.lower() in low:
            return "comm"
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memcpy"
    for pat in _GEMM_PATTERNS:
        if pat in low:
            return "gemm"
    for pat in _ATTENTION_PATTERNS:
        if pat in low:
            return "attention"
    for pat in _REDUCTION_PATTERNS:
        if pat in low:
            return "reduction"
    for pat in _ELEMENTWISE_PATTERNS:
        if pat in low:
            return "elementwise"
    return "other_kernel"


def _find_step_event(
    events: list[dict[str, Any]], step: int | None
) -> dict[str, Any]:
    step_events = []
    for e in events:
        name = e.get("name", "")
        if not isinstance(name, str) or not name.startswith("ProfilerStep#"):
            continue
        if e.get("ph") != "X" or "dur" not in e:
            continue
        step_events.append(e)

    if not step_events:
        raise SystemExit(
            "No ProfilerStep#N events found. Is this actually a "
            "torch.profiler Chrome trace with an active step?"
        )

    step_events.sort(key=lambda e: e["ts"])
    if step is None:
        return step_events[-1]

    want = f"ProfilerStep#{step}"
    for e in step_events:
        if e["name"] == want:
            return e
    available = [e["name"].removeprefix("ProfilerStep#") for e in step_events]
    raise SystemExit(f"Step {step} not found. Available steps: {available}")


def _events_in_window(
    events: list[dict[str, Any]], t0: float, t1: float
) -> list[dict[str, Any]]:
    out = []
    for e in events:
        if e.get("ph") != "X":
            continue
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        # Keep events that *start* inside the window. This matches how the
        # torch profiler step is conventionally interpreted.
        if ts < t0 or ts >= t1:
            continue
        out.append(e)
    return out


def _merge_intervals(
    intervals: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _intersect_total(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Sum of overlap length between two lists of non-overlapping intervals."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _is_kernel(e: dict[str, Any]) -> bool:
    cat = e.get("cat", "")
    return cat in ("kernel", "gpu_memcpy", "gpu_memset")


def _build_thread_name_map(
    events: list[dict[str, Any]],
) -> dict[tuple[int, int], str]:
    # Metadata events name the CUDA streams / CPU threads.
    names: dict[tuple[int, int], str] = {}
    for e in events:
        if e.get("ph") != "M":
            continue
        if e.get("name") != "thread_name":
            continue
        pid = e.get("pid")
        tid = e.get("tid")
        label = (e.get("args") or {}).get("name")
        if pid is None or tid is None or not label:
            continue
        names[(pid, tid)] = label
    return names


def analyze(trace_path: Path, step: int | None) -> dict[str, Any]:
    trace = _load_trace(trace_path)
    events = trace.get("traceEvents") or []

    step_event = _find_step_event(events, step)
    step_num = int(step_event["name"].removeprefix("ProfilerStep#"))
    t0 = float(step_event["ts"])
    t1 = t0 + float(step_event["dur"])

    window = _events_in_window(events, t0, t1)
    thread_names = _build_thread_name_map(events)

    category_time_us: dict[str, float] = defaultdict(float)
    category_count: dict[str, int] = defaultdict(int)

    kernel_time_us: dict[str, float] = defaultdict(float)
    kernel_count: dict[str, int] = Counter()
    kernel_cat: dict[str, str] = {}

    stream_time_us: dict[str, float] = defaultdict(float)

    comm_intervals: list[tuple[float, float]] = []
    noncomm_kernel_intervals: list[tuple[float, float]] = []
    all_kernel_intervals: list[tuple[float, float]] = []

    gpu_event_ts_min = float("inf")
    gpu_event_ts_max = float("-inf")

    for e in window:
        cat = e.get("cat", "")
        name = e.get("name", "") or ""
        dur = float(e.get("dur", 0.0))
        ts = float(e.get("ts", 0.0))

        if _is_kernel(e):
            category = _categorize_kernel(name, cat)
            category_time_us[category] += dur
            category_count[category] += 1
            kernel_time_us[name] += dur
            kernel_count[name] += 1
            kernel_cat[name] = category
            stream_label = thread_names.get(
                (e.get("pid"), e.get("tid")), f"stream:{e.get('tid')}"
            )
            stream_time_us[stream_label] += dur
            all_kernel_intervals.append((ts, ts + dur))
            if category == "comm":
                comm_intervals.append((ts, ts + dur))
            else:
                noncomm_kernel_intervals.append((ts, ts + dur))
            if ts < gpu_event_ts_min:
                gpu_event_ts_min = ts
            if ts + dur > gpu_event_ts_max:
                gpu_event_ts_max = ts + dur

    comm_merged = _merge_intervals(comm_intervals)
    noncomm_merged = _merge_intervals(noncomm_kernel_intervals)
    all_merged = _merge_intervals(all_kernel_intervals)

    total_comm_time = sum(e - s for s, e in comm_merged)
    overlapped_comm_time = _intersect_total(comm_merged, noncomm_merged)
    gpu_busy_time = sum(e - s for s, e in all_merged)
    # compute_busy = union of non-comm kernel intervals across all streams.
    # Relates to the others by: gpu_busy = compute_busy + comm - (comm ∩ compute)
    compute_busy_time = sum(e - s for s, e in noncomm_merged)
    step_wall_us = t1 - t0
    # gpu_span: first-to-last GPU kernel activity in the step. Use this as the
    # denominator for GPU-side %s — the ProfilerStep CPU wall can include a
    # dispatch-then-block tail that isn't really GPU idle.
    gpu_span_us = (
        gpu_event_ts_max - gpu_event_ts_min
        if gpu_event_ts_max > gpu_event_ts_min
        else 0.0
    )
    gpu_idle_us = max(0.0, gpu_span_us - gpu_busy_time) if gpu_span_us else 0.0

    # GPU-side percentages denominate by gpu_span_us (first → last kernel),
    # not the CPU ProfilerStep wall. Keeps these numbers honest when CPU and
    # GPU timelines are misaligned (e.g. aot_fx_trace dispatch-then-block).
    gpu_denom = gpu_span_us if gpu_span_us else step_wall_us

    top_kernels = []
    for name, t in sorted(
        kernel_time_us.items(), key=lambda kv: kv[1], reverse=True
    )[:10]:
        cnt = kernel_count[name]
        top_kernels.append(
            {
                "name": name,
                "category": kernel_cat[name],
                "total_us": round(t, 2),
                "count": cnt,
                "avg_us": round(t / cnt, 3) if cnt else 0.0,
                "pct_of_step": round(100.0 * t / gpu_denom, 2)
                if gpu_denom
                else 0.0,
            }
        )

    categories = []
    for c in CATEGORY_ORDER:
        categories.append(
            {
                "category": c,
                "total_us": round(category_time_us.get(c, 0.0), 2),
                "count": category_count.get(c, 0),
                "pct_of_step": round(
                    100.0 * category_time_us.get(c, 0.0) / gpu_denom, 2
                )
                if gpu_denom
                else 0.0,
            }
        )

    streams = sorted(
        (
            {
                "stream": label,
                "total_us": round(t, 2),
                "pct_of_step": round(100.0 * t / gpu_denom, 2)
                if gpu_denom
                else 0.0,
            }
            for label, t in stream_time_us.items()
        ),
        key=lambda d: d["total_us"],
        reverse=True,
    )

    return {
        "trace_path": str(trace_path),
        "step": step_num,
        "step_wall_us": round(step_wall_us, 2),
        "step_wall_ms": round(step_wall_us / 1000.0, 3),
        "gpu_span_us": round(gpu_span_us, 2),
        "gpu_span_ms": round(gpu_span_us / 1000.0, 3),
        "gpu": {
            "busy_us": round(gpu_busy_time, 2),
            "idle_us": round(gpu_idle_us, 2),
            # Utilization denominated by GPU-side span (first → last kernel),
            # not by the CPU-side ProfilerStep window. This avoids misreading
            # a CPU dispatch-then-block tail as GPU idle time.
            "utilization_pct": round(100.0 * gpu_busy_time / gpu_span_us, 2)
            if gpu_span_us
            else 0.0,
            # Union of non-comm kernel intervals — the "compute" side of the
            # overlap picture. gpu_busy = compute_busy + comm - (comm ∩ compute).
            "compute_busy_us": round(compute_busy_time, 2),
        },
        "comm": {
            "total_us": round(total_comm_time, 2),
            "overlapped_us": round(overlapped_comm_time, 2),
            "overlap_pct": round(
                100.0 * overlapped_comm_time / total_comm_time, 2
            )
            if total_comm_time
            else 0.0,
            "exposed_us": round(total_comm_time - overlapped_comm_time, 2),
            # Exposed comm is a GPU-timeline property, so denominate by GPU span.
            "exposed_pct_of_step": round(
                100.0 * (total_comm_time - overlapped_comm_time) / gpu_span_us,
                2,
            )
            if gpu_span_us
            else 0.0,
        },
        "kernel_categories": categories,
        "streams": streams,
        "top_kernels": top_kernels,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "trace", type=Path, help="Path to Chrome trace JSON (.json or .json.gz)"
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="ProfilerStep# to analyze. Defaults to the last step in the trace.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write metrics JSON here instead of stdout.",
    )
    args = p.parse_args()

    if not args.trace.exists():
        print(f"Trace not found: {args.trace}", file=sys.stderr)
        sys.exit(1)

    metrics = analyze(args.trace, args.step)
    payload = json.dumps(metrics, indent=2)
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
