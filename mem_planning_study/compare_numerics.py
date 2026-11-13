# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bitwise loss / grad_norm comparison across runs (rank-local memstat.jsonl).

Usage: compare_numerics.py <run_a> <run_b> [<run_c> ...]
Each run dir holds rank*/memstat.jsonl with float.hex() loss and grad_norm.
"""

import glob
import json
import os
import sys

ROOT = "/data/users/bahuang/local/torchtitan_2/outputs/memplan"


def load(run):
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, run, "rank*", "memstat.jsonl"))):
        rank = os.path.basename(os.path.dirname(path))
        out[rank] = {r["step"]: r for r in map(json.loads, open(path))}
    return out


def main():
    runs = sys.argv[1:]
    data = {r: load(r) for r in runs}
    ref = runs[0]
    ok = True
    for rank in sorted(data[ref]):
        steps = sorted(data[ref][rank])
        for step in steps:
            vals = []
            for r in runs:
                rec = data[r].get(rank, {}).get(step)
                vals.append(
                    None if rec is None else (rec.get("loss"), rec.get("grad_norm"))
                )
            same = all(v == vals[0] for v in vals)
            ok &= same
            loss = float.fromhex(vals[0][0]) if vals[0] and vals[0][0] else float("nan")
            gn = float.fromhex(vals[0][1]) if vals[0] and vals[0][1] else float("nan")
            extra = "" if same else "  MISMATCH " + " | ".join(str(v) for v in vals)
            print(
                f"{rank} step {step:3d} loss={loss:.10f} grad_norm={gn:.10f} {'==' if same else '!='}{extra}"
            )
    print("ALL BITWISE EQUAL" if ok else "DIFFERENCES FOUND")


if __name__ == "__main__":
    main()
