#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare two torchtitan weight-hash dumps step-by-step.

Usage:
    python scripts/hash_compare.py <run_a_dir> <run_b_dir>

Each argument is either a ``weight_hashes/`` directory or the parent
``dump_folder`` (the script will append ``weight_hashes/`` if needed).
The script walks all ``rank*.jsonl`` files present in both dirs and
reports the first diverging step + offending parameter FQNs.

Exit codes:
    0 — all hashes match across all ranks
    1 — at least one hash mismatch
    2 — input error (missing files, malformed JSON)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _resolve(path: str) -> str:
    if os.path.basename(path.rstrip("/")) == "weight_hashes":
        return path
    candidate = os.path.join(path, "weight_hashes")
    if os.path.isdir(candidate):
        return candidate
    return path


def _load(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _compare_rank(rank: int, a_path: str, b_path: str) -> bool:
    a = _load(a_path)
    b = _load(b_path)
    if len(a) != len(b):
        print(
            f"  rank{rank}: record-count mismatch ({len(a)} vs {len(b)}); "
            "comparing the common prefix"
        )
    ok = True
    for ra, rb in zip(a, b):
        if ra["step"] != rb["step"]:
            print(f"  rank{rank}: step misalignment {ra['step']} vs {rb['step']}")
            return False
        if ra["global_hash"] == rb["global_hash"]:
            continue
        ok = False
        print(f"  rank{rank}: DIVERGENCE at step {ra['step']}")
        ap = ra.get("per_param") or {}
        bp = rb.get("per_param") or {}
        diff = [k for k in sorted(set(ap) | set(bp)) if ap.get(k) != bp.get(k)]
        if diff:
            print(f"    {len(diff)} param(s) differ. First 10:")
            for k in diff[:10]:
                print(f"      - {k}")
        # First divergence is enough; bail out for this rank.
        return False
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_a", help="First run dir (dump_folder or weight_hashes/)")
    p.add_argument("run_b", help="Second run dir (dump_folder or weight_hashes/)")
    args = p.parse_args()

    a_dir = _resolve(args.run_a)
    b_dir = _resolve(args.run_b)

    if not os.path.isdir(a_dir):
        print(f"error: {a_dir} is not a directory", file=sys.stderr)
        return 2
    if not os.path.isdir(b_dir):
        print(f"error: {b_dir} is not a directory", file=sys.stderr)
        return 2

    a_files = {f for f in os.listdir(a_dir) if f.startswith("rank") and f.endswith(".jsonl")}
    b_files = {f for f in os.listdir(b_dir) if f.startswith("rank") and f.endswith(".jsonl")}
    common = sorted(a_files & b_files, key=lambda s: int(s[4:-6]))
    only_a = a_files - b_files
    only_b = b_files - a_files
    if only_a or only_b:
        print(f"warning: rank file set differs (only in A: {sorted(only_a)}; "
              f"only in B: {sorted(only_b)})")

    if not common:
        print("error: no overlapping rank files to compare", file=sys.stderr)
        return 2

    print(f"Comparing {len(common)} rank(s) between:\n  A: {a_dir}\n  B: {b_dir}")
    all_ok = True
    for fname in common:
        rank = int(fname[4:-6])
        ok = _compare_rank(rank, os.path.join(a_dir, fname), os.path.join(b_dir, fname))
        if not ok:
            all_ok = False
    if all_ok:
        print("PASS: all per-rank hashes match across all steps")
        return 0
    print("FAIL: hash divergence detected")
    return 1


if __name__ == "__main__":
    sys.exit(main())
