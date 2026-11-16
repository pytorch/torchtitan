# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attribute the profiled step's GPU kernels to FX nodes and rank glue by roofline gap.

    python node_table.py <capture dir>

Reads ``graph.jsonl.gz`` and ``profile/iteration_<step>/rank0_trace.json[.gz]``
(written by a ``configure_capture`` run) and ``hbm_bandwidth.json``. Writes
``node_table.csv`` (one row per node, in graph order) and ``step_budget.json``,
and prints the step budget and the glue ops furthest above their roofline.

A kernel is attributed through its CUDA launch call (same ``correlation`` id) to
the innermost node label enclosing the launch on the launching thread. Kernels
launched outside every label (optimizer, gradient clipping) are outside the graph.
"""

import argparse
import bisect
import csv
import glob
import gzip
import json
import os
import re
from collections import defaultdict

import torch

_LABEL_RE = re.compile(r"^([A-Za-z_]\w*): ")
_GEMMS = {
    "aten.mm",
    "aten.addmm",
    "aten.bmm",
    "aten.baddbmm",
    "aten._grouped_mm",
    "aten._scaled_mm",
    "aten._scaled_mm_v2",
    "aten._scaled_grouped_mm",
    "aten._scaled_grouped_mm_v2",
}
_ATTENTION = (
    "flex_attention",
    "scaled_dot_product",
    "flash_attention",
    "flash_attn",
    "efficient_attention",
    "cudnn_attention",
    "varlen_attn",
)
_TOP = 25


def _input_nodes(x) -> list[str]:
    """Names of the nodes referenced by an encoded args/kwargs structure, in order."""
    if isinstance(x, list):
        return [n for v in x for n in _input_nodes(v)]
    if isinstance(x, dict):
        if set(x) == {"node"}:
            return [x["node"]]
        return [n for v in x.values() for n in _input_nodes(v)]
    return []


def _tensors(val) -> list[dict]:
    if isinstance(val, dict) and val.get("t") == "tensor":
        return [val]
    if isinstance(val, list):
        return [t for v in val for t in _tensors(v)]
    return []


def _nbytes(t: dict, symbols: dict, *, unique: bool) -> int | None:
    """Bytes of an encoded tensor; ``unique`` skips broadcast (stride 0) dims.

    None if a size is a symbol the trace did not resolve; 0 for host tensors.
    """
    if t["device"] != "cuda":
        return 0
    n = getattr(torch, t["dtype"]).itemsize
    for size, stride in zip(t["shape"], t["stride"]):
        size = symbols.get(size) if isinstance(size, str) else size
        if size is None:
            return None
        if not (unique and stride == 0):
            n *= size
    return n


def _category(target: str, kernel_names: set[str]) -> str:
    lower = target.lower()
    if target.rsplit(".", 1)[0] in _GEMMS:
        return "gemm"
    if any(a in lower for a in _ATTENTION) or any("flex" in k for k in kernel_names):
        return "attention"
    if target.startswith("dist_moe.") and not target.startswith("dist_moe.prepare_"):
        return "fused_moe"
    if (
        target.startswith(("_c10d_functional.", "c10d_functional.", "_dtensor."))
        or "wait_tensor" in target
        or (target.startswith("torchtitan.") and "token_exchange" in target)
    ):
        return "collective"
    return "glue" if kernel_names else "no_kernel"


def _busy_us(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if e > end:
            total += e - max(s, end)
            end = e
    return total


def _load_trace(capture_dir: str) -> list[dict]:
    paths = glob.glob(
        os.path.join(capture_dir, "profile", "iteration_*", "rank0_trace.json*")
    )
    if not paths:
        raise FileNotFoundError(
            f"no profile/iteration_*/rank0_trace.json* in {capture_dir}"
        )
    path = max(paths, key=lambda p: int(re.search(r"iteration_(\d+)", p).group(1)))
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)["traceEvents"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    args = parser.parse_args()
    root = args.capture_dir
    with gzip.open(os.path.join(root, "graph.jsonl.gz"), "rt") as f:
        nodes = {r["name"]: r for r in map(json.loads, f)}
    with open(os.path.join(root, "hbm_bandwidth.json")) as f:
        hbm_gbps = json.load(f)["hbm_gbps"]

    kernels = []  # (ts, dur, name, correlation)
    launches = {}  # correlation -> (tid, ts) of the CUDA runtime call
    labels = defaultdict(list)  # tid -> [(ts, end, node)]
    input_dims = {}  # node -> runtime input sizes of its first call
    for e in _load_trace(root):
        if e.get("ph") != "X":
            continue
        cat, eargs = e.get("cat"), e.get("args", {})
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append((e["ts"], e["dur"], e["name"], eargs.get("correlation")))
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in eargs:
            launches[eargs["correlation"]] = (e["tid"], e["ts"])
        elif cat in ("cpu_op", "user_annotation"):
            m = _LABEL_RE.match(e["name"])
            if m and m.group(1) in nodes:
                labels[e["tid"]].append((e["ts"], e["ts"] + e["dur"], m.group(1)))
                input_dims.setdefault(m.group(1), eargs.get("Input Dims"))

    # Labels nest only around subgraph calls; parent[i] is the label enclosing label i.
    starts, parents = {}, {}
    for tid, lst in labels.items():
        lst.sort()
        parent, stack = [], []
        for i, (s, _, _) in enumerate(lst):
            while stack and lst[stack[-1]][1] < s:
                stack.pop()
            parent.append(stack[-1] if stack else -1)
            stack.append(i)
        starts[tid], parents[tid] = [s for s, _, _ in lst], parent

    def node_of(correlation) -> str | None:
        if correlation not in launches:
            return None
        tid, ts = launches[correlation]
        if tid not in labels:
            return None
        i = bisect.bisect_right(starts[tid], ts) - 1
        while i >= 0 and labels[tid][i][1] < ts:
            i = parents[tid][i]
        return labels[tid][i][2] if i >= 0 else None

    kernel_us = defaultdict(float)
    num_kernels = defaultdict(int)
    kernel_names = defaultdict(set)
    outside_us = 0.0
    for _, dur, name, correlation in kernels:
        node = node_of(correlation)
        if node is None:
            outside_us += dur
            continue
        kernel_us[node] += dur
        num_kernels[node] += 1
        kernel_names[node].add(name[:80])

    # Unbacked sizes (e.g. MoE routed rows) from the sizes each label recorded for
    # its node arguments.
    symbols = {}
    for name, dims in input_dims.items():
        arg_nodes = _input_nodes(nodes[name]["args"])
        if not isinstance(dims, list) or len(dims) != len(arg_nodes):
            continue
        for arg, sizes in zip(arg_nodes, dims):
            val = nodes[arg]["val"]
            if (
                _tensors(val) == [val]
                and isinstance(sizes, list)
                and len(sizes) == len(val["shape"])
            ):
                for s, v in zip(val["shape"], sizes):
                    if isinstance(s, str):
                        symbols.setdefault(s, v)

    rows = []
    for name, n in nodes.items():
        if n["op"] not in ("call_function", "call_module"):
            continue
        category = _category(n["target"], kernel_names[name])
        nbytes = ideal_us = gap_us = None
        if category == "glue":
            reads = [
                _nbytes(t, symbols, unique=True)
                for a in dict.fromkeys(_input_nodes([n["args"], n["kwargs"]]))
                for t in _tensors(nodes[a]["val"])
            ]
            writes = [_nbytes(t, symbols, unique=False) for t in _tensors(n["val"])]
            if None not in reads + writes:
                nbytes = sum(reads + writes)
                ideal_us = nbytes / (hbm_gbps * 1e3)
                gap_us = max(kernel_us[name] - ideal_us, 0.0)
        rows.append(
            {
                "name": name,
                "phase": n["phase"],
                "module_fqn": n["module_fqn"],
                "target": n["target"],
                "category": category,
                "kernel_us": round(kernel_us[name], 2),
                "num_kernels": num_kernels[name],
                "bytes": nbytes,
                "ideal_us": None if ideal_us is None else round(ideal_us, 2),
                "gap_us": None if gap_us is None else round(gap_us, 2),
                "src": n["src"],
                "kernels": " | ".join(sorted(kernel_names[name])[:2]),
            }
        )
    with open(os.path.join(root, "node_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_category = defaultdict(float)
    for r in rows:
        by_category[r["category"]] += r["kernel_us"] / 1e3
    glue = [r for r in rows if r["category"] == "glue"]
    kernel_ms = sum(d for _, d, _, _ in kernels) / 1e3
    window_ms = (
        max(t + d for t, d, _, _ in kernels) - min(t for t, _, _, _ in kernels)
    ) / 1e3
    busy_ms = _busy_us([(t, t + d) for t, d, _, _ in kernels]) / 1e3
    budget = {
        "gpu_window_ms": round(window_ms, 3),
        "gpu_busy_ms": round(busy_ms, 3),
        "gpu_idle_ms": round(window_ms - busy_ms, 3),
        "kernel_ms": round(kernel_ms, 3),
        "outside_graph_ms": round(outside_us / 1e3, 3),
        "graph_ms_by_category": {
            c: round(v, 3) for c, v in sorted(by_category.items())
        },
        "glue_gap_ms": round(sum(r["gap_us"] or 0 for r in glue) / 1e3, 3),
        "glue_nodes_without_bytes": sum(r["bytes"] is None for r in glue),
        "hbm_gbps": hbm_gbps,
    }
    with open(os.path.join(root, "step_budget.json"), "w") as f:
        json.dump(budget, f, indent=1)
    print(json.dumps(budget, indent=1))

    # The same op in every layer is one group: these are the ops worth a kernel.
    groups = defaultdict(lambda: [0, 0.0, 0.0, ""])
    for r in glue:
        key = (
            r["phase"],
            re.sub(r"\blayers\.\d+", "layers.*", r["module_fqn"]),
            r["target"],
        )
        g = groups[key]
        g[0] += 1
        g[1] += r["kernel_us"] / 1e3
        g[2] += (r["gap_us"] or 0) / 1e3
        g[3] = g[3] or r["src"]
    print(
        f"\nglue furthest above roofline ({hbm_gbps} GB/s), grouped by phase, module and op:"
    )
    print(f"{'gap_ms':>8} {'meas_ms':>8} {'nodes':>5}  phase      module / op / source")
    for (phase, fqn, target), (count, meas, gap, src) in sorted(
        groups.items(), key=lambda kv: -kv[1][2]
    )[:_TOP]:
        print(
            f"{gap:8.3f} {meas:8.3f} {count:5d}  {phase:9s}  {fqn or '-'}  {target}  {src}"
        )


if __name__ == "__main__":
    main()
