# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Aggregate per-rollout grades into a per-task pass-rate and emit a curated JSONL.

Stage 2 of the pass-rate data-curation ("data washing") pipeline -- the pure-CPU
half. ``curate_passrate.py`` (stage 1) runs the policy over the task pool and
writes one small JSON per (instance, sample) carrying at least
``{"instance_id", "solved"}``. This script reads that directory, computes::

    pass_rate(task) = solved_samples / graded_samples

then keeps only tasks inside a learnability BAND (default 0.2 < pass_rate < 0.7,
exclusive), discarding all-fail (0.0 -- too hard or a broken env, no learning
signal) and all-solve (1.0 -- too easy, zero within-group variance), and writes a
curated JSONL that is a drop-in for ``SWER2EDataset`` (the original row plus
``metadata.pass_rate`` / ``resolved`` / ``num_samples``). This mirrors the msl/rl
recipe (rejection-sampling pass@k -> instance_summary -> exclusive-band filter);
the band IS the curriculum that unstarves binary-reward GRPO on sparse R2E.

The reader is format-agnostic: it accepts any directory of JSON files that each
carry ``instance_id`` + ``solved`` (the ``curate_passrate.py`` output AND the
training rollout dumps written by ``rollouter._maybe_dump_trace``), so the same
aggregator can post-process a wash job or an existing training run's dumps.

Example::

    python -m torchtitan.experiments.rl.examples.swe_r2e.aggregate_passrate \
        --results-dir /mnt/<bucket>/.../curate_out/results \
        --source-jsonl /mnt/<bucket>/.../r2e_subset_4p5k.jsonl \
        --out /mnt/<bucket>/.../r2e_band_20_70.jsonl \
        --pass-min 0.2 --pass-max 0.7 --min-samples 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass, field

# Grades we count toward the pass-rate denominator. An ERROR/timeout attempt is an
# infra failure (sandbox/agent crash, not a fair "the model could not solve it"),
# so by default it does not count as a graded sample; --include-errors overrides.
_GRADED_STATUSES = {"completed"}


@dataclass(slots=True)
class TaskStats:
    """Per-instance roll-up of one task's sampled rollouts."""

    instance_id: str
    graded: int = 0
    """Samples that ran and were graded (the pass-rate denominator)."""
    resolved: int = 0
    """Graded samples that fully solved the task (solved == True)."""
    errors: int = 0
    """Attempts dropped from the denominator (status not in _GRADED_STATUSES)."""

    @property
    def pass_rate(self) -> float:
        return self.resolved / self.graded if self.graded else 0.0


def _iter_result_files(results_dir: str):
    """Yield every ``*.json`` under ``results_dir`` (recursively)."""
    for root, _dirs, files in os.walk(results_dir):
        for name in files:
            if name.endswith(".json"):
                yield os.path.join(root, name)


def collect_stats(results_dir: str, *, include_errors: bool) -> dict[str, TaskStats]:
    """Read per-(instance, sample) grade JSONs into per-instance ``TaskStats``.

    Each file must carry ``instance_id`` and ``solved``; ``status`` is optional
    (treated as graded when absent). Files that fail to parse or lack an
    ``instance_id`` are skipped (best-effort, so a partial wash still aggregates).
    """
    stats: dict[str, TaskStats] = {}
    n_files = n_bad = 0
    for path in _iter_result_files(results_dir):
        n_files += 1
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            n_bad += 1
            continue
        iid = rec.get("instance_id")
        if not iid:
            n_bad += 1
            continue
        ts = stats.get(iid)
        if ts is None:
            ts = stats[iid] = TaskStats(instance_id=iid)
        status = str(rec.get("status", "completed")).lower()
        graded = include_errors or status in _GRADED_STATUSES
        if not graded:
            ts.errors += 1
            continue
        ts.graded += 1
        if bool(rec.get("solved")):
            ts.resolved += 1
    print(f"read {n_files} files ({n_bad} skipped) -> {len(stats)} unique instances")
    return stats


# Histogram bins as inclusive-low/exclusive-high fractions, with 0.0 and 1.0 split
# out (those are the all-fail / all-solve buckets the band filter always drops).
_BINS = [
    (0.0, 0.0, "==0% (all-fail / broken)"),
    (0.0, 0.1, "(0,10%]"),
    (0.1, 0.2, "(10,20%]"),
    (0.2, 0.3, "(20,30%]"),
    (0.3, 0.4, "(30,40%]"),
    (0.4, 0.5, "(40,50%]"),
    (0.5, 0.6, "(50,60%]"),
    (0.6, 0.7, "(60,70%]"),
    (0.7, 0.8, "(70,80%]"),
    (0.8, 0.9, "(80,90%]"),
    (0.9, 1.0, "(90,100%)"),
    (1.0, 1.0, "==100% (all-solve / too-easy)"),
]


def _bin_label(pr: float) -> str:
    if pr <= 0.0:
        return _BINS[0][2]
    if pr >= 1.0:
        return _BINS[-1][2]
    for lo, hi, label in _BINS[1:-1]:
        if lo < pr <= hi:
            return label
    return _BINS[-2][2]


def print_histogram(stats: dict[str, TaskStats], *, min_samples: int) -> None:
    """Print the pass-rate distribution, counting only tasks with enough samples."""
    counts: Counter[str] = Counter()
    n_thin = 0
    for ts in stats.values():
        if ts.graded < min_samples:
            n_thin += 1
            continue
        counts[_bin_label(ts.pass_rate)] += 1
    total = sum(counts.values())
    print(f"\npass-rate histogram (>= {min_samples} graded samples; {total} tasks):")
    for _lo, _hi, label in _BINS:
        c = counts.get(label, 0)
        bar = "#" * min(60, c)
        print(f"  {label:32s} {c:5d} {bar}")
    if n_thin:
        print(f"  ({n_thin} tasks had < {min_samples} graded samples -> excluded)")


def _row_instance_id(row: dict) -> str | None:
    md = row.get("metadata") or {}
    iid = md.get("instance_id")
    if iid:
        return iid
    label = row.get("label")
    return label if isinstance(label, str) else None


@dataclass(slots=True)
class BandResult:
    """Outcome of applying the band filter to the source pool."""

    kept: list[dict] = field(default_factory=list)
    n_too_hard: int = 0  # pass_rate <= pass_min (incl. all-fail)
    n_too_easy: int = 0  # pass_rate >= pass_max (incl. all-solve)
    n_thin: int = 0  # graded < min_samples
    n_unsampled: int = 0  # in source but never rolled out


def apply_band(
    source_rows: list[dict],
    stats: dict[str, TaskStats],
    *,
    pass_min: float,
    pass_max: float,
    min_samples: int,
) -> BandResult:
    """Keep source rows whose pass_rate is strictly inside (pass_min, pass_max).

    Bounds are EXCLUSIVE (msl/rl convention): all-fail (<=pass_min) and all-solve
    (>=pass_max) are dropped. Each kept row gets ``metadata.pass_rate`` /
    ``resolved`` / ``num_samples`` stamped for provenance and step-2 inspection.
    """
    res = BandResult()
    for row in source_rows:
        iid = _row_instance_id(row)
        ts = stats.get(iid) if iid else None
        if ts is None:
            res.n_unsampled += 1
            continue
        if ts.graded < min_samples:
            res.n_thin += 1
            continue
        pr = ts.pass_rate
        if pr <= pass_min:
            res.n_too_hard += 1
            continue
        if pr >= pass_max:
            res.n_too_easy += 1
            continue
        md = dict(row.get("metadata") or {})
        md["pass_rate"] = round(pr, 4)
        md["resolved"] = ts.resolved
        md["num_samples"] = ts.graded
        kept = dict(row)
        kept["metadata"] = md
        res.kept.append(kept)
    return res


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_instance_summary(stats: dict[str, TaskStats], path: str) -> None:
    """Write the msl-style instance_summary.csv (instance_id, resolved, num_samples,
    pass_rate) for the full sampled pool, so the band can be re-derived offline."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "resolved", "num_samples", "errors", "pass_rate"])
        for ts in sorted(stats.values(), key=lambda t: t.pass_rate):
            w.writerow(
                [
                    ts.instance_id,
                    ts.resolved,
                    ts.graded,
                    ts.errors,
                    f"{ts.pass_rate:.4f}",
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        required=True,
        help="Directory of per-(instance,sample) grade JSONs (recursively walked).",
    )
    ap.add_argument(
        "--source-jsonl",
        required=True,
        help="The original task pool JSONL the wash ran over (joined by instance_id).",
    )
    ap.add_argument("--out", required=True, help="Curated band JSONL to write.")
    ap.add_argument(
        "--pass-min", type=float, default=0.2, help="Exclusive lower band bound."
    )
    ap.add_argument(
        "--pass-max", type=float, default=0.7, help="Exclusive upper band bound."
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=8,
        help="Require this many graded samples before trusting a task's pass-rate.",
    )
    ap.add_argument(
        "--include-errors",
        action="store_true",
        help="Count ERROR/timeout attempts in the pass-rate denominator (default: drop them).",
    )
    ap.add_argument(
        "--summary-csv",
        default="",
        help="Optional path for the full instance_summary.csv (default: <out>.summary.csv).",
    )
    args = ap.parse_args()

    if not 0.0 <= args.pass_min < args.pass_max <= 1.0:
        raise ValueError(
            f"need 0 <= pass_min ({args.pass_min}) < pass_max ({args.pass_max}) <= 1"
        )

    stats = collect_stats(args.results_dir, include_errors=args.include_errors)
    if not stats:
        raise ValueError(f"no per-sample results found under {args.results_dir!r}")
    print_histogram(stats, min_samples=args.min_samples)

    with open(args.source_jsonl) as f:
        source_rows = [json.loads(line) for line in f if line.strip()]

    res = apply_band(
        source_rows,
        stats,
        pass_min=args.pass_min,
        pass_max=args.pass_max,
        min_samples=args.min_samples,
    )

    write_jsonl(res.kept, args.out)
    summary_csv = args.summary_csv or f"{args.out}.summary.csv"
    write_instance_summary(stats, summary_csv)

    print(
        f"\nsource tasks: {len(source_rows)} | sampled: {len(stats)} | "
        f"band ({args.pass_min}, {args.pass_max}) kept: {len(res.kept)}"
    )
    print(
        f"  dropped: too_hard(<= {args.pass_min})={res.n_too_hard} "
        f"too_easy(>= {args.pass_max})={res.n_too_easy} "
        f"thin(< {args.min_samples} samples)={res.n_thin} "
        f"never_sampled={res.n_unsampled}"
    )
    print(f"curated band -> {args.out}")
    print(f"instance summary -> {summary_csv}")


if __name__ == "__main__":
    main()
