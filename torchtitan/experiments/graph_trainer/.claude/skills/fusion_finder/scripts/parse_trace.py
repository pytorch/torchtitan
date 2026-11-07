# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parse the Kineto trace of the profiled step and attribute GPU kernels.

Every GPU activity (kernel / memcpy / memset) is mapped to the CUDA runtime
call that launched it (``correlation``), and from there, by time containment
on the launching CPU thread, to the enclosing ``_RecordFunctionFast`` label
emitted by ``custom_codegen_pass`` ("node_name: op (file:line)"). Kernels
launched outside the FX graph (optimizer, grad clipping, ...) are attributed
to their outermost enclosing non-step CPU op instead.

Usage: ``python parse_trace.py <capture dir>``. Reads the newest
``iteration_*/rank0_trace.json.gz`` and writes ``analysis/{kernels,labels,long_cpu_ops}.pkl``
and ``analysis/trace_summary.json``.
"""

import argparse
import bisect
import glob
import gzip
import json
import os
import re
from collections import defaultdict

import pandas as pd

LABEL_RE = re.compile(r"^(?P<node>[A-Za-z_]\w*): (?P<op>\S+)(?: \((?P<src>.*)\))?$")
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="capture directory (enable_fusion_finder out_dir)")
    args = p.parse_args()
    traces = sorted(
        glob.glob(os.path.join(args.root, "iteration_*", "rank0_trace.json*")),
        key=lambda t: int(re.search(r"iteration_(\d+)", t).group(1)),
    )
    if not traces:
        raise FileNotFoundError(f"no iteration_*/rank0_trace.json* under {args.root}")
    trace_path = traces[-1]
    out = os.path.join(args.root, "analysis")
    os.makedirs(out, exist_ok=True)
    print("trace", trace_path)

    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as f:
        trace = json.load(f)
    events = trace["traceEvents"]
    print("events", len(events))

    gpu = []
    runtime = {}  # correlation -> (tid, ts, dur, name)
    labels_by_tid = defaultdict(list)  # tid -> [(ts, end, node, op, src)]
    cpu_ops_by_tid = defaultdict(
        list
    )  # top-level-ish cpu ops for non-graph attribution
    step_events = []
    label_rows = []
    cat_counts = defaultdict(int)
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        cat_counts[cat] += 1
        name = e.get("name", "")
        if cat in GPU_CATS:
            a = e.get("args", {})
            gpu.append(
                (
                    float(e["ts"]),
                    float(e.get("dur", 0)),
                    name,
                    cat,
                    a.get("stream"),
                    a.get("correlation"),
                    a.get("device"),
                )
            )
        elif cat in ("cuda_runtime", "cuda_driver"):
            a = e.get("args", {})
            c = a.get("correlation")
            if c is not None:
                runtime[c] = (e["tid"], float(e["ts"]), float(e.get("dur", 0)), name)
        elif cat in ("cpu_op", "user_annotation"):
            if name.startswith("ProfilerStep#"):
                step_events.append(
                    (float(e["ts"]), float(e["ts"]) + float(e["dur"]), name)
                )
                continue
            m = LABEL_RE.match(name)
            ts = float(e["ts"])
            end = ts + float(e.get("dur", 0))
            if m:
                labels_by_tid[e["tid"]].append(
                    (ts, end, m.group("node"), m.group("op"), m.group("src"))
                )
                a = e.get("args", {})
                label_rows.append(
                    dict(
                        ts=ts,
                        dur=end - ts,
                        tid=e["tid"],
                        node=m.group("node"),
                        input_dims=a.get("Input Dims"),
                        input_types=a.get("Input type"),
                        input_strides=a.get("Input Strides"),
                    )
                )
            else:
                cpu_ops_by_tid[e["tid"]].append((ts, end, name))
    print("cat counts", dict(cat_counts))
    print("steps", step_events)
    print("label tids", {k: len(v) for k, v in labels_by_tid.items()})

    for tid in labels_by_tid:
        labels_by_tid[tid].sort()
    label_starts = {tid: [x[0] for x in v] for tid, v in labels_by_tid.items()}
    for tid in cpu_ops_by_tid:
        cpu_ops_by_tid[tid].sort()
    cpu_starts = {tid: [x[0] for x in v] for tid, v in cpu_ops_by_tid.items()}

    def find_label(tid, ts):
        lst = labels_by_tid.get(tid)
        if not lst:
            return None
        i = bisect.bisect_right(label_starts[tid], ts) - 1
        # labels are not nested; but guard by checking a few predecessors
        for j in range(i, max(i - 3, -1), -1):
            s, en, node, op, src = lst[j]
            if s <= ts <= en:
                return lst[j]
        return None

    def find_outer_cpu_op(tid, ts):
        """Outermost cpu op containing ts (excluding ProfilerStep)."""
        lst = cpu_ops_by_tid.get(tid)
        if not lst:
            return None
        i = bisect.bisect_right(cpu_starts[tid], ts) - 1
        best = None
        # walk back a bounded number of events looking for containing ops;
        # keep the earliest-starting (outermost) one.
        for j in range(i, max(i - 4000, -1), -1):
            s, en, name = lst[j]
            if s <= ts <= en:
                best = lst[j]
        return best

    rows = []
    for ts, dur, name, cat, stream, corr, dev in gpu:
        rt = runtime.get(corr)
        node = op = src = None
        launch_ts = None
        outer = None
        tid = None
        if rt is not None:
            tid, launch_ts, _, _ = rt
            lab = find_label(tid, launch_ts)
            if lab is not None:
                node, op, src = lab[2], lab[3], lab[4]
            else:
                o = find_outer_cpu_op(tid, launch_ts)
                outer = o[2] if o else None
        rows.append(
            dict(
                ts=ts,
                dur=dur,
                kernel=name,
                cat=cat,
                stream=stream,
                corr=corr,
                launch_ts=launch_ts,
                tid=tid,
                node=node,
                label_op=op,
                label_src=src,
                outer_cpu_op=outer,
            )
        )
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df.to_pickle(os.path.join(out, "kernels.pkl"))
    pd.DataFrame(label_rows).to_pickle(os.path.join(out, "labels.pkl"))
    # Non-graph CPU ops of interest (optimizer etc.) for the summary.
    outer_rows = []
    for tid, lst in cpu_ops_by_tid.items():
        for s_, en, name in lst:
            if en - s_ > 1000:  # >1 ms CPU ops only
                outer_rows.append(dict(tid=tid, ts=s_, dur=en - s_, name=name))
    pd.DataFrame(outer_rows).to_pickle(os.path.join(out, "long_cpu_ops.pkl"))

    summary = {
        "trace": trace_path,
        "num_events": len(events),
        "cat_counts": dict(cat_counts),
        "steps": step_events,
        "num_gpu_activities": len(df),
        "num_labels": {str(k): len(v) for k, v in labels_by_tid.items()},
        "kernels_with_node": int(df["node"].notna().sum()),
        "kernels_with_outer_cpu_op": int(df["outer_cpu_op"].notna().sum()),
        "kernels_without_runtime": int(df["launch_ts"].isna().sum()),
    }
    with open(os.path.join(out, "trace_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
