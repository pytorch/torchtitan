# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU time saved between consecutive profiler traces, counted from kernel call counts.

    python account.py <trace A> <trace B> [<trace C> ...] [--top N]

For each kernel (name and grid; memcpys and memsets by size) whose call count differs between two
traces, the saving is (calls in A - calls in B) x its time per call, taken from the trace where it is
called more often. Kernels with equal counts contribute nothing, so a uniform clock slowdown of every
kernel (step 2 of the README) moves the result by its percentage only. Kernels launched under a
collective op are excluded, as in ``../qwen3_30b_eager_study/busy.py``.
"""
import collections
import gzip
import json
import re
import sys

COMM = ("c10d::", "_c10d_functional::", "nccl", "record_param_comms")


def kernel_times(path):
    """Total time (us) and call count per kernel key of one trace."""
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        ev = json.load(f)["traceEvents"]
    comm_ops = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "cpu_op":
            if any(k in e["name"] for k in COMM):
                comm_ops[e["tid"]].append((e["ts"], e["ts"] + e["dur"]))
    launches = {
        e["args"].get("correlation"): e
        for e in ev
        if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "args" in e
    }
    total, calls = collections.Counter(), collections.Counter()
    for k in ev:
        if k.get("ph") != "X" or k.get("cat") not in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ):
            continue
        launch = launches.get(k.get("args", {}).get("correlation"))
        if launch and any(
            s <= launch["ts"] <= t for s, t in comm_ops.get(launch["tid"], ())
        ):
            continue
        name = k["name"]
        if "Memcpy" in name or "Memset" in name:
            name = f"{name} {k['args'].get('bytes')}"
        key = re.sub(r"0x[0-9a-f]+", "", name)[:140] + f" grid={k['args'].get('grid')}"
        total[key] += k["dur"]
        calls[key] += 1
    return total, calls


def saving(a, b, top=0):
    """GPU time (ms) that trace b saves against trace a."""
    (ta, ca), (tb, cb) = a, b
    rows = []
    for key in set(ca) | set(cb):
        if ca[key] == cb[key]:
            continue
        per_call = ta[key] / ca[key] if ca[key] >= cb[key] else tb[key] / cb[key]
        rows.append(((ca[key] - cb[key]) * per_call, ca[key], cb[key], key))
    rows.sort(key=lambda r: -abs(r[0]))
    for d, x, y, key in rows[:top]:
        print(f"   {d / 1e3:+8.2f} ms  {x:5d} -> {y:5d}  {key[:100]}")
    return sum(r[0] for r in rows) / 1e3


def main():
    args = sys.argv[1:]
    top = 0
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i : i + 2]
    traces = [kernel_times(p) for p in args]
    for i in range(1, len(traces)):
        d = saving(traces[i - 1], traces[i], top)
        print(f"{args[i - 1]} -> {args[i]}: saving {d:.1f} ms")


if __name__ == "__main__":
    main()
