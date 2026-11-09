# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU time of one profiled step from a torch profiler chrome trace.

    python busy.py <trace.json.gz> [--top N]

busy: union of GPU kernel/memcpy/memset intervals over all streams. Kernels launched under a
c10d / functional-collective CPU op are reported separately (the fake PG's emulated collectives).
"""
import collections
import gzip
import json
import sys


def load(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        return json.load(f)["traceEvents"]


def union(iv):
    tot, cs, ce = 0.0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ((ce - cs) if ce is not None else 0.0)


# Collective ops only: FSDP2's copy-in (fsdp::all_gather_copy_in) and GraphTrainer's bucket packing
# (bucketing::_pre_bucket_*) contain collective names but run real kernels, which stay counted.
COMM = ("c10d::", "_c10d_functional::", "nccl", "record_param_comms")


def main():
    ev = load(sys.argv[1])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 0
    gpu = [
        e
        for e in ev
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    # CPU op intervals per thread for collective ops; runtime launches carry the correlation id.
    cpu_ops = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    comm_ops = collections.defaultdict(list)
    for e in cpu_ops:
        if any(k in e["name"] for k in COMM):
            comm_ops[e["tid"]].append((e["ts"], e["ts"] + e["dur"]))
    launches = {
        e["args"].get("correlation"): e
        for e in ev
        if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "args" in e
    }

    def is_comm(k):
        l = launches.get(k.get("args", {}).get("correlation"))
        return bool(l) and any(s <= l["ts"] <= t for s, t in comm_ops.get(l["tid"], ()))

    iv_all = [(e["ts"], e["ts"] + e["dur"]) for e in gpu]
    comm = [e for e in gpu if is_comm(e)]
    iv_nocomm = [(e["ts"], e["ts"] + e["dur"]) for e in gpu if not is_comm(e)]
    print(
        f"kernels {len(gpu)}  busy {union(iv_all) / 1e3:.1f} ms  "
        f"busy_excl_comm {union(iv_nocomm) / 1e3:.1f} ms  "
        f"comm-launched kernels {len(comm)} ({sum(e['dur'] for e in comm) / 1e3:.1f} ms summed)  "
        f"span {(max(t for _, t in iv_all) - min(s for s, _ in iv_all)) / 1e3:.1f} ms"
    )
    if top:
        by = collections.Counter()
        cnt = collections.Counter()
        for e in gpu:
            by[e["name"][:110]] += e["dur"]
            cnt[e["name"][:110]] += 1
        for n, d in by.most_common(top):
            print(f"  {d / 1e3:9.2f} ms  {cnt[n]:6d}  {n}")


main()
