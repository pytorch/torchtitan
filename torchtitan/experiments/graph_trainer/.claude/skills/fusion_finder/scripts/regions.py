# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Group the glue nodes of a capture into fusion regions and rank them by gap.

    python regions.py <capture dir>

Reads ``graph.jsonl.gz``, ``node_table.csv`` and ``step_budget.json`` (from
``node_table.py``) and ``hbm_bandwidth.json``. Writes ``regions.json`` and prints
the regions above the profit floor.

A region instance is a connected set of glue nodes, plus the kernel-free nodes
(views, allocations) among them, such that no boundary lies on a path between two
members. Boundaries are the nodes a glue kernel cannot absorb: GEMMs, attention,
Dist-MoE fused ops, collectives, and nodes whose output is not a GPU tensor (host
values). Instances grow by merging along glue edges, in graph order, while that
holds. Hence:

- every glue node is in exactly one instance;
- no path leaves an instance and comes back: an instance can become one
  ``call_module``, and one kernel can replace it.

Forward glue never merges with recompute or backward glue: that edge is a saved
activation, and fusing across it would move backward work into the forward pass.

Instances with the same members up to the layer index (op, phase, module with
``layers.<N>`` as ``layers.*``, output shapes with symbolic sizes abstracted) are
one region. Ideal time of an instance: external input bytes (read once) plus the
bytes of its outputs used outside, over the HBM bandwidth. The gap of a region is
its measured time minus its ideal time, summed over instances. Regions below the
profit floor (max(0.5 ms, 0.05% of GPU busy time)) are not worth a kernel: the
gap is what a kernel can save at best.
"""

import argparse
import csv
import gzip
import json
import os
import re
from collections import Counter, defaultdict

from node_table import input_nodes, tensors

_FLOOR_MS = 0.5
_FLOOR_PCT_OF_BUSY = 0.05


def _short_op(target: str) -> str:
    return target.removeprefix("aten.").removesuffix(".default")


def _any_layer(fqn: str) -> str:
    return re.sub(r"\blayers\.\d+", "layers.*", fqn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    args = parser.parse_args()
    root = args.capture_dir
    with gzip.open(os.path.join(root, "graph.jsonl.gz"), "rt") as f:
        nodes = {r["name"]: r for r in map(json.loads, f)}
    with open(os.path.join(root, "node_table.csv")) as f:
        table = {r["name"]: r for r in csv.DictReader(f)}
    with open(os.path.join(root, "step_budget.json")) as f:
        budget = json.load(f)
    with open(os.path.join(root, "hbm_bandwidth.json")) as f:
        hbm_gbps = json.load(f)["hbm_gbps"]
    order = {name: i for i, name in enumerate(nodes)}

    inputs = {
        name: list(dict.fromkeys(input_nodes([n["args"], n["kwargs"]])))
        for name, n in nodes.items()
    }
    users = defaultdict(list)
    for name in nodes:
        for a in inputs[name]:
            users[a].append(name)
    # A write into a buffer has no FX edge to the buffer's later readers: make one, for
    # out= ops (the out kwarg) and in-place ops (their first argument).
    for name, n in nodes.items():
        op = n["target"].split(".")[-2] if n["target"].count(".") >= 2 else ""
        if n["target"].endswith(".out"):
            buffers = input_nodes(n["kwargs"].get("out"))
        elif op.endswith("_") and not op.endswith("__") and inputs[name]:
            buffers = inputs[name][:1]
        else:
            continue
        for b in buffers:
            for reader in users[b]:
                if order[reader] > order[name] and name not in inputs[reader]:
                    inputs[reader].append(name)
                    users[name].append(reader)

    def category(name: str) -> str:
        return table[name]["category"] if name in table else "input"

    def on_gpu(name: str) -> bool:
        ts = tensors(nodes[name]["val"])
        return bool(ts) and all(t["device"] == "cuda" for t in ts)

    candidates = sorted(
        (
            name
            for name in nodes
            if category(name) in ("glue", "no_kernel") and on_gpu(name)
        ),
        key=order.__getitem__,
    )
    is_candidate = set(candidates)
    boundaries = [
        name
        for name in nodes
        if name not in is_candidate
        and category(name) != "input"
        and nodes[name]["op"] != "output"
    ]
    bit = {name: 1 << i for i, name in enumerate(boundaries)}
    # Bitsets of the boundaries upstream and downstream of each node (graph order is
    # a topological order).
    upstream, downstream = {}, {}
    for name in nodes:
        mask = bit.get(name, 0)
        for a in inputs[name]:
            mask |= upstream[a]
        upstream[name] = mask
    for name in reversed(nodes):
        mask = bit.get(name, 0)
        for u in users[name]:
            mask |= downstream[u]
        downstream[name] = mask

    # Merge along glue edges unless a boundary would sit on a path between the two
    # sides. A forward value read by recompute or backward is a saved activation:
    # fusing across that edge would move backward work into the forward pass, so
    # those edges are never merged (backward nodes never feed forward ones, so this
    # keeps instances closed under paths).
    parent = {name: name for name in candidates}
    up = {name: upstream[name] for name in candidates}
    down = {name: downstream[name] for name in candidates}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    late = {name: nodes[name]["phase"] != "forward" for name in nodes}
    for name in candidates:
        for a in inputs[name]:
            if a not in is_candidate or late[a] != late[name]:
                continue
            ra, rb = find(a), find(name)
            if ra == rb or down[ra] & up[rb] or down[rb] & up[ra]:
                continue
            parent[ra] = rb
            up[rb] |= up[ra]
            down[rb] |= down[ra]
    groups = defaultdict(list)
    for name in candidates:
        groups[find(name)].append(name)
    kernel_us = {name: float(table[name]["kernel_us"]) for name in table}
    instances = [
        members for members in groups.values() if any(kernel_us[m] > 0 for m in members)
    ]

    def out_bytes(name: str) -> int:
        # Sizes the trace did not resolve count as 0 (node_table.py reports them).
        return int(table[name]["out_bytes"] or 0) if name in table else 0

    def measure(members: list[str]) -> dict:
        mset = set(members)
        nbytes = 0
        for a in dict.fromkeys(a for m in members for a in inputs[m] if a not in mset):
            readers = [m for m in users[a] if m in mset]
            # Read through views only: the views' bytes, e.g. one slice of a tensor.
            if all(kernel_us[m] == 0 for m in readers):
                nbytes += min(out_bytes(a), sum(out_bytes(m) for m in readers))
            else:
                nbytes += out_bytes(a)
        written = set()
        for m in members:
            if all(u in mset for u in users[m]):
                continue
            # A view used outside: the kernel that wrote its storage.
            while kernel_us[m] == 0 and inputs[m] and inputs[m][0] in mset:
                m = inputs[m][0]
            if kernel_us[m] > 0 and m not in written:
                written.add(m)
                nbytes += out_bytes(m)
        return {
            "measured_us": sum(kernel_us[m] for m in members),
            "ideal_us": nbytes / (hbm_gbps * 1e3),
            "kernels": sum(int(table[m]["num_kernels"]) for m in members),
        }

    def signature(members: list[str]) -> str:
        key = []
        for m in members:
            n = nodes[m]
            shapes = [
                [s if isinstance(s, int) else "?" for s in t["shape"]]
                for t in tensors(n["val"])
            ]
            key.append(
                (
                    n["phase"],
                    _any_layer(n["module_fqn"]),
                    n["target"],
                    json.dumps(shapes),
                )
            )
        return json.dumps(sorted(key))

    by_signature = defaultdict(list)
    for members in instances:
        by_signature[signature(members)].append(members)

    regions = []
    for group in by_signature.values():
        stats = [measure(members) for members in group]
        measured_ms = sum(s["measured_us"] for s in stats) / 1e3
        ideal_ms = sum(s["ideal_us"] for s in stats) / 1e3
        first = group[0]
        ops = Counter(_short_op(nodes[m]["target"]) for m in first if kernel_us[m] > 0)
        regions.append(
            {
                "gap_ms": round(max(measured_ms - ideal_ms, 0.0), 3),
                "measured_ms": round(measured_ms, 3),
                "ideal_ms": round(ideal_ms, 3),
                "instances": len(group),
                "kernels_per_instance": stats[0]["kernels"],
                "phases": sorted({nodes[m]["phase"] for m in first}),
                "modules": sorted({_any_layer(nodes[m]["module_fqn"]) for m in first}),
                "ops": dict(ops.most_common()),
                # The first instance as it appears in the dumped FX code, one line
                # per node.
                "code": [nodes[m]["code"] for m in first],
                "members": group,
            }
        )
    regions.sort(key=lambda r: -r["gap_ms"])
    for i, r in enumerate(regions):
        r["id"] = f"r{i:03d}"
    floor_ms = max(_FLOOR_MS, _FLOOR_PCT_OF_BUSY / 100 * budget["gpu_busy_ms"])
    above = [r for r in regions if r["gap_ms"] >= floor_ms]
    with open(os.path.join(root, "regions.json"), "w") as f:
        json.dump(
            {"profit_floor_ms": round(floor_ms, 3), "regions": regions}, f, indent=1
        )

    total_gap = sum(r["gap_ms"] for r in regions)
    print(
        f"{len(regions)} regions ({len(instances)} instances), gap {total_gap:.1f} ms; "
        f"{len(above)} above the profit floor of {floor_ms:.2f} ms hold "
        f"{sum(r['gap_ms'] for r in above):.1f} ms"
    )
    print(
        f"{'id':5s} {'gap_ms':>8s} {'meas_ms':>8s} {'inst':>5s} {'k/inst':>6s}  phases  modules  ops"
    )
    for r in above:
        ops = ", ".join(f"{op} x{c}" for op, c in list(r["ops"].items())[:4])
        more = f" (+{len(r['ops']) - 4} ops)" if len(r["ops"]) > 4 else ""
        print(
            f"{r['id']:5s} {r['gap_ms']:8.2f} {r['measured_ms']:8.2f} {r['instances']:5d} "
            f"{r['kernels_per_instance']:6d}  {','.join(r['phases'])}  "
            f"{' '.join(r['modules'][:2])}{' ...' if len(r['modules']) > 2 else ''}  {ops}{more}"
        )


if __name__ == "__main__":
    main()
