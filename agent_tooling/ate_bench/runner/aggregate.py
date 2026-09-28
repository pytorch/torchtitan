"""Aggregate ATE-Bench Q&A run metrics into a Table-5-style report.

Reads the output directory produced by run_qa.py (one subdir per task, each with
``runN.metrics.json`` files) and reports the *median* across attempts per task,
matching the paper's protocol (median of 3 independent attempts; lower is more
efficient). Prints Agent Turns / Per-Turn Context / Output Tokens (the three Q&A
metrics in paper Table 5) plus session duration, and writes summary.json.

Usage:
    python agent_tooling/ate_bench/runner/aggregate.py <results_dir> [<results_dir2> ...]

Passing two or more results dirs (e.g. one per framework) prints them side by side
so you can reproduce the cross-framework comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


METRIC_KEYS = [
    ("agent_turns", "Turns"),
    ("per_turn_context", "Per-Turn Ctx"),
    ("output_tokens", "Output Tok"),
    ("session_duration_s", "Duration(s)"),
    ("active_gpu_time_s", "GPU Time(s)"),
]


def _is_correct(row: dict) -> bool | None:
    """Pull a correctness verdict from operate (passed) or feature (correct) rows."""
    c = row.get("correctness")
    if not isinstance(c, dict):
        return None
    if "correct" in c:
        return bool(c["correct"])
    if "passed" in c:
        return bool(c["passed"])
    return None


def load_task_medians(results_dir: Path) -> dict[str, dict]:
    """Return {task_id: {metric: median_value, 'n', 'n_err', 'n_correct'}}."""
    out: dict[str, dict] = {}
    for task_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        metric_files = sorted(task_dir.glob("run*.metrics.json"))
        if not metric_files:
            continue
        rows = [json.loads(f.read_text()) for f in metric_files]
        ok = [r for r in rows if not r.get("is_error") and not r.get("error")]
        n_err = len(rows) - len(ok)
        verdicts = [_is_correct(r) for r in rows]
        graded = [v for v in verdicts if v is not None]
        agg: dict = {
            "n": len(ok),
            "n_err": n_err,
            "n_correct": sum(1 for v in graded if v),
            "n_graded": len(graded),
        }
        for key, _ in METRIC_KEYS:
            vals = [r[key] for r in ok if r.get(key) is not None]
            agg[key] = median(vals) if vals else None
        out[task_dir.name] = agg
    return out


def _fmt(key: str, val) -> str:
    if val is None:
        return "-"
    if key in ("per_turn_context", "output_tokens"):
        return f"{val / 1000:.1f}K"
    if key in ("session_duration_s", "active_gpu_time_s"):
        return f"{val:.0f}"
    return f"{val:.0f}"


def print_single(label: str, medians: dict[str, dict]) -> None:
    print(f"\n# {label}  (median of attempts; lower is more efficient)\n")
    header = ["Task", "n(ok/err)", "correct"] + [h for _, h in METRIC_KEYS]
    widths = [max(len(header[0]), 6), 10, 8] + [max(len(h), 12) for _, h in METRIC_KEYS]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for task_id, agg in medians.items():
        correct = f"{agg['n_correct']}/{agg['n_graded']}" if agg.get("n_graded") else "-"
        row = [task_id, f"{agg['n']}/{agg['n_err']}", correct]
        row += [_fmt(k, agg.get(k)) for k, _ in METRIC_KEYS]
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def print_compare(by_label: dict[str, dict[str, dict]]) -> None:
    """Side-by-side per-metric comparison across frameworks/labels."""
    labels = list(by_label)
    all_tasks = sorted({t for m in by_label.values() for t in m})
    for key, hname in METRIC_KEYS:
        print(f"\n# {hname}  (median; lower is more efficient)\n")
        header = ["Task"] + labels
        widths = [6] + [max(len(lbl), 10) for lbl in labels]
        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        print("  ".join("-" * w for w in widths))
        for task_id in all_tasks:
            row = [task_id]
            for lbl in labels:
                agg = by_label[lbl].get(task_id, {})
                row.append(_fmt(key, agg.get(key)))
            print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", type=Path, help="results dir(s) from run_qa.py")
    args = ap.parse_args(argv)

    by_label: dict[str, dict[str, dict]] = {}
    for d in args.results:
        d = d.expanduser().resolve()
        if not d.is_dir():
            ap.error(f"{d} is not a directory")
        label = d.name
        medians = load_task_medians(d)
        by_label[label] = medians
        summary_path = d / "summary.json"
        summary_path.write_text(json.dumps(medians, indent=2), encoding="utf-8")

    if len(by_label) == 1:
        (label,) = by_label
        print_single(label, by_label[label])
    else:
        print_compare(by_label)

    print()
    for d in args.results:
        print(f"Wrote {d.expanduser().resolve() / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
