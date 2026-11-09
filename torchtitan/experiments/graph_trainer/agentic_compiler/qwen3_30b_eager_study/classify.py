# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Classify the Qwen3 30B-A3B fusion regions from the graph the lift passes see.

    PYTHONPATH=<fusion_region_finder scripts>:. python classify.py <report dir> <plan> <dump> <out.json>

Replays the plan's lift passes in order on the dump (as replay.py does) and records, per site:
phase of each member (forward, backward, or recompute: a forward op that GraphTrainer's memory
policy duplicated into the backward, named ``*_recomputed``), the layers and modules its members come from
(custom.module_fqn, else the forward node with the same seq_nr), the source lines, and what
produces its external inputs and consumes its outputs.
"""

import collections
import gzip
import importlib
import json
import os
import re
import sys

from lift_regions import find_matches, lift_hazard, load_dump_graph

GEMMS = (
    "_scaled_mm",
    "aten.mm",
    "aten.addmm",
    "aten.bmm",
    "_grouped_mm",
    "scaled_grouped_mm",
)
COLLECTIVES = (
    "_c10d_functional",
    "wait_tensor",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "all_reduce",
)


def fqn_of(rec):
    c = rec.get("custom") or {}
    f = c.get("module_fqn")
    return f.strip("'") if f else None


def frames(stack):
    if not stack:
        return []
    return re.findall(r'File "([^"]+)", line (\d+), in (\w+)', stack)


def short(path):
    for key in ("torchtitan/torchtitan/", "pytorch/torch/", "site-packages/"):
        if key in path:
            return (
                path.split(key, 1)[1]
                if key != "pytorch/torch/"
                else "torch/" + path.split(key, 1)[1]
            )
    return path


def phase(rec):
    if rec["name"].endswith("_recomputed"):
        return "recompute"
    return "backward" if rec["autograd_backward"] else "forward"


def kind(rec):
    t = rec["target"] if rec["op"] == "call_function" else rec["op"]
    if rec["op"] == "placeholder":
        return "graph input"
    if rec["op"] == "output":
        return "graph output"
    if any(g in t for g in GEMMS):
        return "GEMM"
    if any(c in t for c in COLLECTIVES):
        return "collective"
    if "flash" in t or "attention" in t or "flex" in t:
        return "attention"
    return "glue"


def main():
    root, plan_file, dump, out = sys.argv[1:5]
    recs = {r["name"]: r for r in (json.loads(l) for l in gzip.open(dump, "rt"))}
    fwd_fqn_by_seq = {}
    for r in recs.values():
        if not r["autograd_backward"] and fqn_of(r) and r["seq_nr"] is not None:
            fwd_fqn_by_seq.setdefault(r["seq_nr"], fqn_of(r))

    def module(name):
        r = recs[name]
        return fqn_of(r) or (
            fwd_fqn_by_seq.get(r["seq_nr"]) if r["autograd_backward"] else None
        )

    def norm(f):
        return re.sub(r"layers\.\d+", "layers.*", f) if f else "<none>"

    sys.path.insert(0, os.path.abspath(root))
    regions = importlib.import_module("fusion_regions")
    passes = [p for p in open(plan_file).read().strip().split(",") if p]
    folder = {p: (rid, f) for rid, (f, p) in regions.PASSES.items()}
    nodes = load_dump_graph(dump)
    taken, result = set(), {}
    for p in passes:
        rid, f = folder[p]
        mod = importlib.import_module(f"fusion_regions.{f}.lift_pass")
        sites = [
            s
            for s in find_matches(nodes, mod.MATCHERS, claimed=set(taken))
            if not lift_hazard(s)
        ]
        taken |= {n for s in sites for n in s}
        agg = collections.defaultdict(collections.Counter)
        per_site_layers = collections.Counter()
        for s in sites:
            names = {n.name for n in s}
            layers = set()
            for n in s:
                r = recs[n.name]
                if r["op"] != "call_function":
                    continue
                m = module(n.name)
                agg["phase"][phase(r)] += 1
                agg["module"][norm(m)] += 1
                if m and (mm := re.match(r"layers\.(\d+)", m)):
                    layers.add(int(mm.group(1)))
                fr = [
                    x for x in frames(r["stack_trace"]) if "fx/traceback.py" not in x[0]
                ]
                tt = [x for x in fr if "torchtitan/torchtitan/" in x[0]]
                if tt:
                    agg["torchtitan_line"][
                        f"{short(tt[-1][0])}:{tt[-1][1]} {tt[-1][2]}"
                    ] += 1
                if fr and fr[-1] not in tt:
                    agg["innermost_line"][
                        f"{short(fr[-1][0])}:{fr[-1][1]} {fr[-1][2]}"
                    ] += 1
                for a in r["args"] + list(r["kwargs"].values()):
                    for ref in re.findall(
                        r"'node': '([^']+)'", json.dumps(a).replace('"', "'")
                    ):
                        if ref not in names:
                            pr = recs[ref]
                            agg["input_from"][
                                f"{kind(pr)} | {norm(module(ref))} | "
                                f"{phase(pr)[:3]}"
                            ] += 1
                for u in r["users"]:
                    if u not in names:
                        ur = recs[u]
                        agg["output_to"][
                            f"{kind(ur)} | {norm(module(u))} | {phase(ur)[:3]}"
                        ] += 1
            per_site_layers[len(layers)] += 1
        result[rid] = {
            "pass": p,
            "sites": len(sites),
            "layers_per_site": dict(per_site_layers),
            **{k: dict(v.most_common()) for k, v in agg.items()},
        }
    json.dump(result, open(out, "w"), indent=1)
    for rid, x in result.items():
        print(
            f"\n== {rid} sites={x['sites']} layers/site={x['layers_per_site']} phase={x.get('phase')}"
        )
        for k in (
            "module",
            "torchtitan_line",
            "innermost_line",
            "input_from",
            "output_to",
        ):
            print(f"  {k}:")
            for v, c in list(x.get(k, {}).items())[:12]:
                print(f"    {c:6d}  {v}")


if __name__ == "__main__":
    main()
