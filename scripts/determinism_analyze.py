#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Analyze and compare deterministic vs non-deterministic TorchTitan training logs.

Parses training metric lines from TorchTitan output, computes per-step and
average metrics (skipping warmup), and formats a side-by-side comparison table
with percent-change columns.

Optionally parses Chrome trace JSON files (e.g., rank0_trace.json) to diff
kernel-level durations between the two runs, sorted by absolute slowdown.

Usage:
    python3 scripts/determinism_analyze.py \
        --nondet-log nondet_training.log \
        --det-log det_training.log \
        --warmup-steps 3 \
        --output comparison_report.txt

    # With kernel trace diff:
    python3 scripts/determinism_analyze.py \
        --nondet-log nondet_training.log \
        --det-log det_training.log \
        --nondet-trace nondet_traces/rank0_trace.json \
        --det-trace det_traces/rank0_trace.json \
        --top-kernels 30
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI escape code stripper
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI color/style escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Metric data classes
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    step: int
    loss: float
    grad_norm: float = 0.0
    memory_gib: float = 0.0
    memory_pct: float = 0.0
    tps: int = 0
    tflops: float = 0.0
    mfu: float | None = None  # None when reported as "N/A"


@dataclass
class RunSummary:
    label: str
    steps: list[StepMetrics] = field(default_factory=list)

    # Computed averages (post-warmup)
    avg_loss: float = 0.0
    avg_tps: float = 0.0
    avg_tflops: float = 0.0
    avg_mfu: float | None = None
    avg_memory_gib: float = 0.0
    avg_memory_pct: float = 0.0
    avg_grad_norm: float = 0.0


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

# Regex for the TorchTitan metrics line (after ANSI stripping).
# Example (after stripping):
#   step:  1  loss: 10.87234  grad_norm:  0.1234  memory:  3.21GiB(12.34%)  tps: 1,234  tflops: 12.34  mfu: 0.12%
_METRICS_RE = re.compile(
    r"step:\s*(\d+)\s+"
    r"loss:\s*([\d.]+)\s+"
    r".*?"
    r"grad_norm:\s*([\d.]+)\s+"
    r".*?"
    r"memory:\s*([\d.]+)GiB\(([\d.]+)%\)\s+"
    r".*?"
    r"tps:\s*([\d,]+)\s+"
    r".*?"
    r"tflops:\s*([\d,.]+)\s+"
    r"mfu:\s*([\d.]+%|N/A)"
)


def parse_log(path: str) -> list[StepMetrics]:
    """Parse a TorchTitan training log and return per-step metrics."""
    results: list[StepMetrics] = []
    seen_steps: set[int] = set()

    with open(path) as f:
        for raw_line in f:
            line = strip_ansi(raw_line)
            m = _METRICS_RE.search(line)
            if not m:
                continue

            step = int(m.group(1))
            if step in seen_steps:
                continue
            seen_steps.add(step)

            mfu_raw = m.group(8)
            mfu_val = None if mfu_raw == "N/A" else float(mfu_raw.rstrip("%"))

            results.append(
                StepMetrics(
                    step=step,
                    loss=float(m.group(2)),
                    grad_norm=float(m.group(3)),
                    memory_gib=float(m.group(4)),
                    memory_pct=float(m.group(5)),
                    tps=int(m.group(6).replace(",", "")),
                    tflops=float(m.group(7).replace(",", "")),
                    mfu=mfu_val,
                )
            )

    results.sort(key=lambda s: s.step)
    return results


def compute_summary(label: str, steps: list[StepMetrics], warmup: int) -> RunSummary:
    """Compute average metrics, skipping the first *warmup* steps."""
    summary = RunSummary(label=label, steps=steps)
    post = [s for s in steps if s.step > warmup]
    if not post:
        return summary

    n = len(post)
    summary.avg_loss = sum(s.loss for s in post) / n
    summary.avg_tps = sum(s.tps for s in post) / n
    summary.avg_tflops = sum(s.tflops for s in post) / n
    summary.avg_memory_gib = sum(s.memory_gib for s in post) / n
    summary.avg_memory_pct = sum(s.memory_pct for s in post) / n
    summary.avg_grad_norm = sum(s.grad_norm for s in post) / n

    mfu_vals = [s.mfu for s in post if s.mfu is not None]
    if mfu_vals:
        summary.avg_mfu = sum(mfu_vals) / len(mfu_vals)

    return summary


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def pct_change(baseline: float, test: float) -> str:
    """Format percent change from baseline to test."""
    if baseline == 0:
        return "N/A"
    pct = (test - baseline) / baseline * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def format_comparison(nondet: RunSummary, det: RunSummary, warmup: int) -> str:
    """Build a human-readable comparison report."""
    lines: list[str] = []
    sep = "=" * 80
    lines.append(sep)
    lines.append("  Deterministic vs Non-Deterministic Training Comparison")
    lines.append(sep)
    lines.append(f"  Warmup steps skipped: {warmup}")
    lines.append(
        f"  Steps compared: {len([s for s in nondet.steps if s.step > warmup])} (nondet)"
        f" / {len([s for s in det.steps if s.step > warmup])} (det)"
    )
    lines.append(sep)
    lines.append("")

    # --- Averages table ---
    header = f"{'Metric':<20} {'Non-Det':>14} {'Det':>14} {'Change':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    rows = [
        ("Avg Loss", f"{nondet.avg_loss:.5f}", f"{det.avg_loss:.5f}",
         pct_change(nondet.avg_loss, det.avg_loss)),
        ("Avg TPS", f"{nondet.avg_tps:,.0f}", f"{det.avg_tps:,.0f}",
         pct_change(nondet.avg_tps, det.avg_tps)),
        ("Avg TFLOPS", f"{nondet.avg_tflops:,.2f}", f"{det.avg_tflops:,.2f}",
         pct_change(nondet.avg_tflops, det.avg_tflops)),
        ("Avg MFU",
         f"{nondet.avg_mfu:.2f}%" if nondet.avg_mfu is not None else "N/A",
         f"{det.avg_mfu:.2f}%" if det.avg_mfu is not None else "N/A",
         pct_change(nondet.avg_mfu, det.avg_mfu)
         if nondet.avg_mfu is not None and det.avg_mfu is not None else "N/A"),
        ("Avg Memory (GiB)", f"{nondet.avg_memory_gib:.2f}",
         f"{det.avg_memory_gib:.2f}",
         pct_change(nondet.avg_memory_gib, det.avg_memory_gib)),
        ("Avg Memory (%)", f"{nondet.avg_memory_pct:.2f}%",
         f"{det.avg_memory_pct:.2f}%",
         pct_change(nondet.avg_memory_pct, det.avg_memory_pct)),
        ("Avg Grad Norm", f"{nondet.avg_grad_norm:.4f}",
         f"{det.avg_grad_norm:.4f}",
         pct_change(nondet.avg_grad_norm, det.avg_grad_norm)),
    ]

    for metric, nd_val, d_val, chg in rows:
        lines.append(f"{metric:<20} {nd_val:>14} {d_val:>14} {chg:>12}")

    lines.append("")

    # --- Per-step table ---
    lines.append(sep)
    lines.append("  Per-Step Comparison")
    lines.append(sep)

    step_header = (
        f"{'Step':>4}  "
        f"{'ND Loss':>10} {'D Loss':>10}  "
        f"{'ND TPS':>10} {'D TPS':>10}  "
        f"{'ND TFLOPS':>10} {'D TFLOPS':>10}  "
        f"{'ND Mem':>8} {'D Mem':>8}"
    )
    lines.append(step_header)
    lines.append("-" * len(step_header))

    nd_by_step = {s.step: s for s in nondet.steps}
    d_by_step = {s.step: s for s in det.steps}
    all_steps = sorted(set(nd_by_step.keys()) | set(d_by_step.keys()))

    for step in all_steps:
        nd = nd_by_step.get(step)
        d = d_by_step.get(step)
        warmup_marker = " *" if step <= warmup else ""
        lines.append(
            f"{step:>4}{warmup_marker:<2}"
            f"{nd.loss if nd else 0:>10.5f} {d.loss if d else 0:>10.5f}  "
            f"{nd.tps if nd else 0:>10,} {d.tps if d else 0:>10,}  "
            f"{nd.tflops if nd else 0:>10.2f} {d.tflops if d else 0:>10.2f}  "
            f"{nd.memory_gib if nd else 0:>7.2f}G {d.memory_gib if d else 0:>7.2f}G"
        )

    lines.append("")
    lines.append("  (* = warmup step, excluded from averages)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chrome trace (kernel) analysis
# ---------------------------------------------------------------------------


@dataclass
class KernelStats:
    name: str
    total_dur_us: float = 0.0
    count: int = 0

    @property
    def avg_dur_us(self) -> float:
        return self.total_dur_us / self.count if self.count else 0.0


def parse_trace_kernels(path: str) -> dict[str, KernelStats]:
    """Parse a Chrome trace JSON and aggregate kernel durations."""
    with open(path) as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])

    kernels: dict[str, KernelStats] = defaultdict(lambda: KernelStats(name=""))
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("cat") != "kernel":
            continue
        if ev.get("ph") not in ("X", "B", "E"):
            continue

        name = ev.get("name", "<unknown>")
        dur = ev.get("dur", 0)  # microseconds

        ks = kernels[name]
        ks.name = name
        ks.total_dur_us += dur
        ks.count += 1

    return dict(kernels)


def format_kernel_diff(
    nondet_kernels: dict[str, KernelStats],
    det_kernels: dict[str, KernelStats],
    top_n: int,
) -> str:
    """Build a kernel-level duration diff table sorted by absolute slowdown."""
    lines: list[str] = []
    sep = "=" * 110
    lines.append(sep)
    lines.append("  Kernel Duration Diff (Deterministic vs Non-Deterministic)")
    lines.append(sep)
    lines.append("")

    all_names = set(nondet_kernels.keys()) | set(det_kernels.keys())

    diffs: list[tuple[str, float, float, float, int, int]] = []
    for name in all_names:
        nd = nondet_kernels.get(name)
        d = det_kernels.get(name)
        nd_avg = nd.avg_dur_us if nd else 0.0
        d_avg = d.avg_dur_us if d else 0.0
        abs_diff = d_avg - nd_avg  # positive = det is slower
        nd_count = nd.count if nd else 0
        d_count = d.count if d else 0
        diffs.append((name, nd_avg, d_avg, abs_diff, nd_count, d_count))

    # Sort by absolute slowdown (largest positive diff first)
    diffs.sort(key=lambda x: abs(x[3]), reverse=True)

    if top_n > 0:
        diffs = diffs[:top_n]

    header = (
        f"{'Kernel':<60} "
        f"{'ND Avg(us)':>11} {'D Avg(us)':>11} "
        f"{'Diff(us)':>11} {'Change':>9} "
        f"{'ND #':>5} {'D #':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name, nd_avg, d_avg, diff, nd_cnt, d_cnt in diffs:
        chg = pct_change(nd_avg, d_avg) if nd_avg > 0 else "N/A"
        display_name = name if len(name) <= 58 else name[:55] + "..."
        sign = "+" if diff >= 0 else ""
        lines.append(
            f"{display_name:<60} "
            f"{nd_avg:>11.1f} {d_avg:>11.1f} "
            f"{sign}{diff:>10.1f} {chg:>9} "
            f"{nd_cnt:>5} {d_cnt:>5}"
        )

    lines.append("")
    lines.append(f"  Total unique kernels: {len(all_names)}")
    lines.append(f"  Showing top {min(top_n, len(all_names))} by absolute duration diff")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare deterministic vs non-deterministic TorchTitan training runs."
    )
    parser.add_argument(
        "--nondet-log", required=True, help="Path to non-deterministic training log."
    )
    parser.add_argument(
        "--det-log", required=True, help="Path to deterministic training log."
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Number of initial steps to exclude from averages (default: 3).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the comparison report (also printed to stdout).",
    )
    parser.add_argument(
        "--nondet-trace",
        default=None,
        help="Path to non-deterministic Chrome trace JSON (rank0_trace.json).",
    )
    parser.add_argument(
        "--det-trace",
        default=None,
        help="Path to deterministic Chrome trace JSON (rank0_trace.json).",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=20,
        help="Number of top kernels to show in the diff table (default: 20).",
    )
    args = parser.parse_args()

    # --- Parse training logs ---
    nondet_steps = parse_log(args.nondet_log)
    det_steps = parse_log(args.det_log)

    if not nondet_steps:
        print(f"ERROR: No metrics found in non-deterministic log: {args.nondet_log}", file=sys.stderr)
        sys.exit(1)
    if not det_steps:
        print(f"ERROR: No metrics found in deterministic log: {args.det_log}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(nondet_steps)} steps from non-deterministic log.")
    print(f"Parsed {len(det_steps)} steps from deterministic log.")

    # --- Compute summaries ---
    nondet_summary = compute_summary("Non-Deterministic", nondet_steps, args.warmup_steps)
    det_summary = compute_summary("Deterministic", det_steps, args.warmup_steps)

    # --- Format comparison report ---
    report = format_comparison(nondet_summary, det_summary, args.warmup_steps)

    # --- Optional kernel diff ---
    if args.nondet_trace and args.det_trace:
        print(f"Parsing non-deterministic trace: {args.nondet_trace}")
        print(f"Parsing deterministic trace: {args.det_trace}")
        nondet_kernels = parse_trace_kernels(args.nondet_trace)
        det_kernels = parse_trace_kernels(args.det_trace)
        print(f"Found {len(nondet_kernels)} unique kernels in non-det trace.")
        print(f"Found {len(det_kernels)} unique kernels in det trace.")
        report += "\n" + format_kernel_diff(nondet_kernels, det_kernels, args.top_kernels)

    # --- Output ---
    print()
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n")
        print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
