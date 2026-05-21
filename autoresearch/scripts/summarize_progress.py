#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate a progress chart from autoresearch/results.tsv.

Renders a side-by-side TPS and MFU view: per-experiment scatter (color-coded
by status) and the cumulative-best curve over ``keep`` entries. Saves to
``autoresearch/progress.png``.

Usage::

    python autoresearch/scripts/summarize_progress.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TSV = REPO_ROOT / "autoresearch" / "results.tsv"
OUTPUT_PNG = REPO_ROOT / "autoresearch" / "progress.png"

_STATUS_COLOR = {"keep": "tab:green", "discard": "tab:orange", "crash": "tab:red"}


def _plot_metric(ax, indices, values, statuses, label, color):
    """Scatter per-experiment values; overlay cumulative-best (over keep)."""
    best_so_far: list[float] = []
    current_best = 0.0
    for v, s in zip(values, statuses):
        if s == "keep" and v > current_best:
            current_best = v
        best_so_far.append(current_best)

    seen = set()
    for i, (v, s) in enumerate(zip(values, statuses)):
        if v == 0.0 and s == "crash":
            continue
        legend_label = s if s not in seen else None
        seen.add(s)
        ax.scatter(
            i + 1,
            v,
            color=_STATUS_COLOR.get(s, "gray"),
            alpha=0.8,
            s=50,
            edgecolors="black",
            linewidths=0.5,
            label=legend_label,
        )
    ax.plot(
        indices,
        best_so_far,
        color=color,
        label=f"Cumulative best (keep) {label}",
    )
    ax.set_xlabel("Experiment #")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")


def main() -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not available")
        return 1

    if not RESULTS_TSV.exists():
        print(f"ERROR: {RESULTS_TSV} not found")
        return 1

    rows = list(csv.DictReader(RESULTS_TSV.open(), delimiter="\t"))
    if not rows:
        print(f"ERROR: {RESULTS_TSV} has no data rows")
        return 1

    indices = list(range(1, len(rows) + 1))
    tps = [float(r["tps"]) for r in rows]
    mfu = [float(r.get("mfu_pct", 0)) for r in rows]
    statuses = [r["status"] for r in rows]

    fig, (ax_tps, ax_mfu) = plt.subplots(1, 2, figsize=(14, 5))
    _plot_metric(ax_tps, indices, tps, statuses, "TPS", "tab:blue")
    _plot_metric(ax_mfu, indices, mfu, statuses, "MFU %", "tab:purple")
    fig.suptitle(f"Autoresearch progress ({len(rows)} experiments)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=120)
    print(f"Saved chart to {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
