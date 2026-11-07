# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Join the FX graph dump with attributed kernels into a node-level table.

Usage: ``python build_node_table.py <capture dir>``. Inputs:
``graph_dump/final_nodes.jsonl.gz``, ``roofline.json`` and ``analysis/{kernels,labels}.pkl``
(from parse_trace.py).
Outputs: ``analysis/node_table.{pkl,csv.gz}``, ``analysis/step_budget.json``,
``analysis/symbols.json``.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from graph_lib import (
    gemm_peak_tflops,
    GEMM_TARGETS,
    VIEW_TARGETS,
    gemm_flops,
    ideal_us,
    is_attention_target,
    is_fused_moe_target,
    is_comm_target,
    iter_arg_nodes,
    load_nodes,
    node_io_bytes,
    short_src,
    tensor_vals,
)

SYM_RE = re.compile(r"^[us]\d+$")


def interval_union(iv):
    iv = sorted(iv)
    total = 0.0
    cur_s, cur_e = None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def classify(n, kernel_us):
    t = n["target"]
    if n["op"] in ("placeholder", "output", "get_attr"):
        return "none"
    if t in GEMM_TARGETS:
        return "gemm"
    if is_fused_moe_target(t):
        return "fused_moe"
    if is_attention_target(t) or n.get("is_flex_region"):
        return "attention"
    if is_comm_target(t):
        return "fake_comm"
    if t in VIEW_TARGETS and kernel_us == 0:
        return "view"
    if kernel_us == 0:
        return "no_kernel"
    return "memory_bound_glue"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="capture directory with graph_dump/, roofline.json and analysis/")
    args = p.parse_args()
    analysis = os.path.join(args.root, "analysis")
    cal = json.load(open(os.path.join(args.root, "roofline.json")))
    bw = cal["roofline"]["hbm_gbps"]

    nodes, by_name = load_nodes(os.path.join(args.root, "graph_dump", "final_nodes.jsonl.gz"))
    kern = pd.read_pickle(os.path.join(analysis, "kernels.pkl"))
    labels = pd.read_pickle(os.path.join(analysis, "labels.pkl"))
    summ = json.load(open(os.path.join(analysis, "trace_summary.json")))

    # Profiled step window (CPU ProfilerStep#N); GPU work launched inside it.
    steps = sorted(summ["steps"])
    step_ts, step_end, step_name = steps[-1]
    in_step = kern["launch_ts"].between(step_ts, step_end) | kern["launch_ts"].isna()
    k = kern[in_step].copy()
    k["end"] = k["ts"] + k["dur"]
    gpu_first, gpu_last = k["ts"].min(), k["end"].max()
    busy = interval_union(list(zip(k["ts"], k["end"])))

    # ---- symbol resolution from runtime label input dims ----
    sym = {}
    conflicts = Counter()
    lab_first = labels.drop_duplicates("node")
    for _, r in lab_first.iterrows():
        n = by_name.get(r["node"])
        if n is None or not isinstance(r["input_dims"], list):
            continue
        arg_nodes = iter_arg_nodes(n["args"])
        dims = r["input_dims"]
        if len(dims) != len(arg_nodes):
            continue
        for a, d in zip(arg_nodes, dims):
            pn = by_name.get(a)
            if pn is None:
                continue
            tv = tensor_vals(pn["val"])
            if len(tv) != 1 or not isinstance(d, list) or len(d) != len(tv[0]["shape"]):
                continue
            for s, v in zip(tv[0]["shape"], d):
                if isinstance(s, str) and isinstance(v, int):
                    # Any symbolic size expression (unbacked u*, sums of them)
                    if s in sym and sym[s] != v:
                        conflicts[s] += 1
                    sym.setdefault(s, v)
    json.dump({"symbols": sym, "conflicts": dict(conflicts)},
              open(os.path.join(analysis, "symbols.json"), "w"), indent=1)

    # ---- per-node kernel aggregation ----
    gk = k[k["node"].notna()].groupby("node")
    node_kus = gk["dur"].sum().to_dict()
    node_kcnt = gk["dur"].count().to_dict()
    node_knames = gk["kernel"].agg(lambda s: "|".join(sorted(set(x[:80] for x in s))[:4])).to_dict()
    node_ncalls = labels.groupby("node")["ts"].count().to_dict()

    # flex attention compiled regions: mark call nodes whose kernels are flex
    for n in nodes:
        kn = node_knames.get(n["name"], "")
        n["is_flex_region"] = ("flex" in kn.lower()) or ("flex" in n["target"].lower())

    rows = []
    for n in nodes:
        if n["op"] in ("placeholder", "output", "get_attr"):
            continue
        kus = node_kus.get(n["name"], 0.0)
        cat = classify(n, kus)
        rd, wr = node_io_bytes(n, by_name, sym)
        fl = gemm_flops(n, by_name, sym) if cat == "gemm" else 0
        if cat == "attention":
            fl = None  # filled later from shapes (see analyze)
        tot = (rd or 0) + (wr or 0) if rd is not None and wr is not None else None
        peak = (
            gemm_peak_tflops(n, by_name, cal["roofline"])
            if cat == "gemm"
            else cal["roofline"]["bf16_gemm_tflops"]
        )
        ideal = ideal_us(tot, fl, bw, peak) if tot is not None else None
        outs = tensor_vals(n["val"])
        ins = []
        for a in n["input_nodes"]:
            pn = by_name.get(a)
            if pn is not None:
                tv = tensor_vals(pn["val"])
                ins.append(f"{a}:" + ";".join(f"{t['dtype'][6:]}{t['shape']}" for t in tv))
        rows.append(
            dict(
                idx=n["idx"],
                name=n["name"],
                target=n["target"],
                phase=n["phase"],
                module_fqn=n["module_fqn"],
                layer=n["layer"],
                category=cat,
                kernel_us=kus,
                n_kernels=node_kcnt.get(n["name"], 0),
                n_label_calls=node_ncalls.get(n["name"], 0),
                kernel_names=node_knames.get(n["name"], ""),
                inputs=" ".join(ins),
                users=" ".join(n["users"]),
                out_shapes=";".join(f"{t['dtype'][6:]}{t['shape']}" for t in outs),
                bytes_read=rd,
                bytes_written=wr,
                flops=fl,
                ideal_us=ideal,
                ratio=(kus / ideal) if ideal else None,
                src=short_src(n.get("stack_trace")),
                recompute_tag=n.get("recompute"),
            )
        )
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(analysis, "node_table.pkl"))
    df.to_csv(os.path.join(analysis, "node_table.csv.gz"), index=False)

    # ---- step budget ----
    graph_kernel_us = k.loc[k["node"].notna(), "dur"].sum()
    nongraph = k[k["node"].isna()]
    cat_of_node = dict(zip(df["name"], df["category"]))
    by_cat = defaultdict(float)
    for nm, us in node_kus.items():
        c = cat_of_node.get(nm, "unknown_node")
        if c in ("view", "no_kernel"):
            c = "memory_bound_glue"
        by_cat[c] += us
    nongraph_by_outer = nongraph.groupby(nongraph["outer_cpu_op"].fillna("<no runtime/outer op>"))["dur"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    for outer, r in nongraph_by_outer.iterrows():
        o = str(outer)
        if "Optimizer" in o or "adam" in o.lower() or "_foreach" in o or "clip" in o.lower() or "norm" in o.lower():
            by_cat["optimizer"] += r["sum"]
        else:
            by_cat["other"] += r["sum"]
    attributed_us = graph_kernel_us + nongraph.loc[nongraph["outer_cpu_op"].notna(), "dur"].sum()
    total_kernel_us = k["dur"].sum()
    budget = dict(
        step_name=step_name,
        step_cpu_window_ms=(step_end - step_ts) / 1e3,
        gpu_window_ms=(gpu_last - gpu_first) / 1e3,
        step_gpu_busy_ms=busy / 1e3,
        step_gpu_idle_ms=(gpu_last - gpu_first - busy) / 1e3,
        sum_kernel_ms=total_kernel_us / 1e3,
        graph_attributed_kernel_ms=graph_kernel_us / 1e3,
        graph_attributed_pct_of_kernel_time=100 * graph_kernel_us / total_kernel_us,
        attributed_kernel_ms=attributed_us / 1e3,
        attributed_pct=100 * attributed_us / total_kernel_us,
        attributed_pct_of_busy=100 * min(attributed_us, busy) / busy,
        by_category_ms={c: v / 1e3 for c, v in sorted(by_cat.items())},
        nongraph_top=[(str(i), r["sum"] / 1e3, int(r["count"])) for i, r in nongraph_by_outer.head(30).iterrows()],
        num_symbols_resolved=len(sym),
    )
    json.dump(budget, open(os.path.join(analysis, "step_budget.json"), "w"), indent=1)
    print(json.dumps(budget, indent=1))


if __name__ == "__main__":
    main()
