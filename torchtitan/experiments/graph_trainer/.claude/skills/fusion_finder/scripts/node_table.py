# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attribute the profiled step's GPU kernels to FX nodes.

    python node_table.py <capture dir>

Reads ``graph.jsonl.gz`` and ``profile/iteration_<step>/rank0_trace.json[.gz]``
(written by a ``configure_capture`` run). Writes ``node_table.csv`` (one row per
node, in graph order) and ``step_budget.json``, and prints the step budget.

Categories: ``gemm``, ``attention``, ``fused_moe`` (Dist-MoE's fused ops),
``collective`` (including the send-buffer packs of bucketed collectives), ``glue``
(any other node that launches kernels), ``no_kernel`` and ``input``.

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


def input_nodes(x) -> list[str]:
    """Names of the nodes referenced by an encoded args/kwargs structure, in order."""
    if isinstance(x, list):
        return [n for v in x for n in input_nodes(v)]
    if isinstance(x, dict):
        if set(x) == {"node"}:
            return [x["node"]]
        return [n for v in x.values() for n in input_nodes(v)]
    return []


def tensors(val) -> list[dict]:
    """The encoded tensors in a node's ``val``."""
    if isinstance(val, dict) and val.get("t") == "tensor":
        return [val]
    if isinstance(val, list):
        return [t for v in val for t in tensors(v)]
    return []


def _nbytes(t: dict, symbols: dict) -> int | None:
    """Bytes of an encoded tensor, broadcast (stride 0) dims counted once.

    None if a size is a symbol the trace did not resolve; 0 for host tensors.
    """
    if t["device"] != "cuda":
        return 0
    n = getattr(torch, t["dtype"]).itemsize
    for size, stride in zip(t["shape"], t["stride"]):
        size = symbols.get(size) if isinstance(size, str) else size
        if size is None:
            return None
        if stride != 0:
            n *= size
    return n


def _is_collective(target: str) -> bool:
    return (
        target.startswith(("_c10d_functional.", "c10d_functional.", "_dtensor."))
        or "wait_tensor" in target
        or (target.startswith("torchtitan.") and "token_exchange" in target)
    )


def _category(node: dict, user_targets: list[str], kernel_names: set[str]) -> str:
    target = node["target"]
    if node["op"] in ("placeholder", "get_attr"):
        return "input"
    if target.rsplit(".", 1)[0] in _GEMMS:
        return "gemm"
    if any(a in target.lower() for a in _ATTENTION) or any(
        "flex" in k for k in kernel_names
    ):
        return "attention"
    if target.startswith("dist_moe.") and not target.startswith("dist_moe.prepare_"):
        return "fused_moe"
    # A bucketed collective's send buffer is packed by an FSDP bucketing op, or by a
    # cat that only collectives read.
    if (
        _is_collective(target)
        or target.startswith("bucketing.")
        or (
            target == "aten.cat.default"
            and user_targets
            and all(_is_collective(t) for t in user_targets)
        )
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
        arg_nodes = input_nodes(nodes[name]["args"])
        if not isinstance(dims, list) or len(dims) != len(arg_nodes):
            continue
        for arg, sizes in zip(arg_nodes, dims):
            val = nodes[arg]["val"]
            if (
                tensors(val) == [val]
                and isinstance(sizes, list)
                and len(sizes) == len(val["shape"])
            ):
                for s, v in zip(val["shape"], sizes):
                    if isinstance(s, str):
                        symbols.setdefault(s, v)

    users = defaultdict(list)
    for name, n in nodes.items():
        for a in dict.fromkeys(input_nodes([n["args"], n["kwargs"]])):
            users[a].append(name)

    def out_bytes(name: str) -> int | None:
        sizes = [_nbytes(t, symbols) for t in tensors(nodes[name]["val"])]
        return None if None in sizes else sum(sizes)

    rows = []
    for name, n in nodes.items():
        if n["op"] == "output":
            continue
        category = _category(
            n, [nodes[u]["target"] for u in users[name]], kernel_names[name]
        )
        rows.append(
            {
                "name": name,
                "phase": n["phase"],
                "module_fqn": n["module_fqn"],
                "target": n["target"],
                "category": category,
                "kernel_us": round(kernel_us[name], 2),
                "num_kernels": num_kernels[name],
                "out_bytes": out_bytes(name),
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
        if r["category"] != "input":
            by_category[r["category"]] += r["kernel_us"] / 1e3
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
        # Nodes with a size the trace did not resolve: regions.py counts them as 0 bytes.
        "nodes_without_bytes": sum(r["out_bytes"] is None for r in rows),
    }
    with open(os.path.join(root, "step_budget.json"), "w") as f:
        json.dump(budget, f, indent=1)
    print(json.dumps(budget, indent=1))


if __name__ == "__main__":
    main()
