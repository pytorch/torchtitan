#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Analyze training logs from deterministic vs non-deterministic benchmark runs.

Parses TorchTitan training output to extract per-step metrics (tokens/sec,
TFLOPs, MFU, loss, memory) and produces a side-by-side comparison with
slowdown percentages. Optionally compares PyTorch profiler traces to identify
the top kernel-level slowdowns.

Can be run standalone:
    python scripts/determinism_analyze.py \
        --nondet-log outputs/nondet_training.log \
        --det-log outputs/det_training.log \
        --warmup-steps 3 \
        --output benchmark_summary.txt

Or with trace comparison (requires both runs to have profiling enabled):
    python scripts/determinism_analyze.py \
        --nondet-log outputs/nondet_training.log \
        --det-log outputs/det_training.log \
        --nondet-trace outputs/profile_traces/nondet/iteration_10/rank0_trace.json \
        --det-trace outputs/profile_traces/det/iteration_10/rank0_trace.json \
        --output benchmark_summary.txt
"""

import argparse
import json
import re
import sys


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (color codes) from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_training_log(log_path: str) -> list[dict]:
    """Parse a TorchTitan training log and extract per-step metrics.

    Handles ANSI color codes in the output. Looks for lines like:
        step:  5  loss: 11.12340  grad_norm: 1.2345  memory: 45.67GiB(56.78%)  tps: 12,345  tflops: 67.89  mfu: 12.34%

    Returns a list of dicts, one per step, with keys:
        step, loss, memory_gib, memory_pct, tps, tflops, mfu
    """
    steps = []
    # Match the metrics line from MetricsProcessor output.
    # Note: tps uses comma formatting (e.g., "12,345"), tflops may also have
    # commas (e.g., "1,234.56"), and mfu can be "N/A" when quantization is active.
    pattern = re.compile(
        r"step:\s*(\d+)\s+"
        r"loss:\s*([\d.]+)\s+"
        r".*?"  # skip grad_norm and other fields
        r"memory:\s*([\d.]+)GiB\(([\d.]+)%\)\s+"
        r".*?"  # skip optional fields
        r"tps:\s*([\d,]+)\s+"
        r".*?"  # skip optional fields like tps_per_gpu
        r"tflops:\s*([\d,.]+)\s+"
        r"mfu:\s*([\d.]+%|N/A)"
    )

    with open(log_path) as f:
        for line in f:
            # Strip ANSI color codes before matching
            clean_line = strip_ansi(line)
            m = pattern.search(clean_line)
            if m:
                mfu_str = m.group(7)
                mfu_val = float(mfu_str.rstrip("%")) if mfu_str != "N/A" else 0.0
                steps.append(
                    {
                        "step": int(m.group(1)),
                        "loss": float(m.group(2)),
                        "memory_gib": float(m.group(3)),
                        "memory_pct": float(m.group(4)),
                        "tps": int(m.group(5).replace(",", "")),
                        "tflops": float(m.group(6).replace(",", "")),
                        "mfu": mfu_val,
                    }
                )

    return steps


def compute_averages(
    steps: list[dict], warmup_steps: int
) -> dict[str, float]:
    """Compute average metrics, skipping the first `warmup_steps` steps."""
    # Filter out warmup steps
    steady = [s for s in steps if s["step"] > warmup_steps]
    if not steady:
        return {
            "avg_tps": 0,
            "avg_tflops": 0,
            "avg_mfu": 0,
            "avg_loss": 0,
            "peak_memory_gib": 0,
            "peak_memory_pct": 0,
            "num_steps": 0,
        }

    return {
        "avg_tps": sum(s["tps"] for s in steady) / len(steady),
        "avg_tflops": sum(s["tflops"] for s in steady) / len(steady),
        "avg_mfu": sum(s["mfu"] for s in steady) / len(steady),
        "avg_loss": sum(s["loss"] for s in steady) / len(steady),
        "peak_memory_gib": max(s["memory_gib"] for s in steady),
        "peak_memory_pct": max(s["memory_pct"] for s in steady),
        "num_steps": len(steady),
    }


def parse_chrome_trace(trace_path: str) -> dict[str, dict]:
    """Parse a Chrome trace JSON and aggregate kernel durations.

    Returns a dict mapping kernel name -> {total_us, count, avg_us}.
    Only includes CUDA kernel events (cat == "kernel").
    """
    with open(trace_path) as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])

    kernels: dict[str, dict] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        # Filter for GPU kernel events
        if ev.get("cat") != "kernel":
            continue
        name = ev.get("name", "unknown")
        dur = ev.get("dur", 0)  # duration in microseconds

        if name not in kernels:
            kernels[name] = {"total_us": 0, "count": 0}
        kernels[name]["total_us"] += dur
        kernels[name]["count"] += 1

    # Compute averages
    for k, v in kernels.items():
        v["avg_us"] = v["total_us"] / v["count"] if v["count"] > 0 else 0

    return kernels


def compare_traces(
    nondet_kernels: dict[str, dict],
    det_kernels: dict[str, dict],
    top_n: int = 30,
) -> list[dict]:
    """Compare kernel durations between non-deterministic and deterministic traces.

    Returns top_n kernels sorted by absolute slowdown (det_total - nondet_total).
    """
    all_kernel_names = set(nondet_kernels.keys()) | set(det_kernels.keys())
    comparisons = []

    for name in all_kernel_names:
        nd = nondet_kernels.get(name, {"total_us": 0, "count": 0, "avg_us": 0})
        d = det_kernels.get(name, {"total_us": 0, "count": 0, "avg_us": 0})

        slowdown_us = d["total_us"] - nd["total_us"]
        slowdown_pct = (
            (slowdown_us / nd["total_us"] * 100) if nd["total_us"] > 0 else float("inf")
        )

        comparisons.append(
            {
                "name": name,
                "nondet_total_us": nd["total_us"],
                "det_total_us": d["total_us"],
                "nondet_count": nd["count"],
                "det_count": d["count"],
                "nondet_avg_us": nd["avg_us"],
                "det_avg_us": d["avg_us"],
                "slowdown_us": slowdown_us,
                "slowdown_pct": slowdown_pct,
            }
        )

    # Sort by absolute slowdown (biggest regression first)
    comparisons.sort(key=lambda x: x["slowdown_us"], reverse=True)
    return comparisons[:top_n]


def format_summary(
    nondet_avgs: dict[str, float],
    det_avgs: dict[str, float],
    warmup_steps: int,
    nondet_log: str,
    det_log: str,
    trace_comparison: list[dict] | None = None,
) -> str:
    """Format a human-readable summary comparing the two runs."""
    lines = []
    lines.append("=" * 72)
    lines.append("  DETERMINISTIC vs NON-DETERMINISTIC BENCHMARK RESULTS")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Non-deterministic log: {nondet_log}")
    lines.append(f"  Deterministic log:     {det_log}")
    lines.append(f"  Warmup steps skipped:  {warmup_steps}")
    lines.append(
        f"  Steady-state steps:    {int(nondet_avgs['num_steps'])} (nondet), "
        f"{int(det_avgs['num_steps'])} (det)"
    )
    lines.append("")

    # Throughput comparison
    lines.append("-" * 72)
    lines.append("  THROUGHPUT & EFFICIENCY")
    lines.append("-" * 72)
    lines.append(
        f"  {'Metric':<25} {'Non-Det':>12} {'Det':>12} {'Slowdown':>12}"
    )
    lines.append(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    def pct_change(nd: float, d: float) -> str:
        if nd == 0:
            return "N/A"
        pct = (d - nd) / nd * 100
        return f"{pct:+.1f}%"

    metrics = [
        ("Tokens/sec (TPS)", "avg_tps", ".0f"),
        ("TFLOPs", "avg_tflops", ".1f"),
        ("MFU (%)", "avg_mfu", ".2f"),
        ("Loss", "avg_loss", ".4f"),
        ("Peak Memory (GiB)", "peak_memory_gib", ".2f"),
    ]

    for label, key, fmt in metrics:
        nd_val = nondet_avgs[key]
        d_val = det_avgs[key]
        change = pct_change(nd_val, d_val)
        lines.append(
            f"  {label:<25} {nd_val:>12{fmt}} {d_val:>12{fmt}} {change:>12}"
        )

    lines.append("")
    lines.append(
        f"  Overall MFU degradation: "
        f"{pct_change(nondet_avgs['avg_mfu'], det_avgs['avg_mfu'])}"
    )
    lines.append("")

    # Trace comparison
    if trace_comparison:
        lines.append("-" * 72)
        lines.append("  TOP KERNEL SLOWDOWNS (from profiler traces)")
        lines.append("-" * 72)
        lines.append(
            f"  {'Kernel':<50} {'NonDet(us)':>12} {'Det(us)':>12} {'Δ(us)':>12} {'Δ(%)':>8}"
        )
        lines.append(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

        for k in trace_comparison:
            name = k["name"]
            if len(name) > 48:
                name = name[:45] + "..."
            pct = (
                f"{k['slowdown_pct']:+.0f}%"
                if k["slowdown_pct"] != float("inf")
                else "NEW"
            )
            lines.append(
                f"  {name:<50} {k['nondet_total_us']:>12.0f} "
                f"{k['det_total_us']:>12.0f} {k['slowdown_us']:>12.0f} {pct:>8}"
            )

        lines.append("")
        lines.append("  Note: Δ(us) = det - nondet. Positive = slower in deterministic mode.")
        lines.append(
            "  Look for large Δ values — these are the kernels to optimize for faster "
            "deterministic training."
        )
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze deterministic vs non-deterministic benchmark results"
    )
    parser.add_argument(
        "--nondet-log", required=True, help="Path to non-deterministic training log"
    )
    parser.add_argument(
        "--det-log", required=True, help="Path to deterministic training log"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Number of warmup steps to skip (default: 3)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the summary (also printed to stdout)",
    )
    parser.add_argument(
        "--nondet-trace",
        default=None,
        help="Path to non-deterministic Chrome trace JSON (optional)",
    )
    parser.add_argument(
        "--det-trace",
        default=None,
        help="Path to deterministic Chrome trace JSON (optional)",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=30,
        help="Number of top kernel slowdowns to show (default: 30)",
    )
    args = parser.parse_args()

    # Parse logs
    nondet_steps = parse_training_log(args.nondet_log)
    det_steps = parse_training_log(args.det_log)

    if not nondet_steps:
        print(f"ERROR: No training steps found in {args.nondet_log}", file=sys.stderr)
        print(
            "Make sure the log contains lines with 'step: N  loss: ...  tps: ... mfu: ...'",
            file=sys.stderr,
        )
        sys.exit(1)

    if not det_steps:
        print(f"ERROR: No training steps found in {args.det_log}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Parsed {len(nondet_steps)} steps from non-deterministic log, "
        f"{len(det_steps)} steps from deterministic log."
    )

    # Compute averages
    nondet_avgs = compute_averages(nondet_steps, args.warmup_steps)
    det_avgs = compute_averages(det_steps, args.warmup_steps)

    # Optional trace comparison
    trace_comparison = None
    if args.nondet_trace and args.det_trace:
        print("Parsing profiler traces for kernel comparison...")
        nondet_kernels = parse_chrome_trace(args.nondet_trace)
        det_kernels = parse_chrome_trace(args.det_trace)
        trace_comparison = compare_traces(
            nondet_kernels, det_kernels, top_n=args.top_kernels
        )
        print(
            f"Found {len(nondet_kernels)} non-det kernels, "
            f"{len(det_kernels)} det kernels."
        )

    # Format and output summary
    summary = format_summary(
        nondet_avgs,
        det_avgs,
        args.warmup_steps,
        args.nondet_log,
        args.det_log,
        trace_comparison,
    )

    print()
    print(summary)

    if args.output:
        with open(args.output, "w") as f:
            f.write(summary + "\n")
        print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    main()
