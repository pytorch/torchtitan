# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Glue roofline: how much of a GraphTrainer step fusion could remove, and how much the loop removed.

    python roofline.py <model>=<base run dir>,<pack run dir> [...] [--json out.json] [--top N]

Each run dir holds one profiled step (rank0_trace.json.gz anywhere below it). Every GPU kernel, memcpy and
memset of the step is put in one category, first match wins:
- communication: launched under a collective CPU op (c10d / functional collectives); with the fake PG these
  are copies that emulate the collectives, so they are reported but excluded from the step time, as in the
  loops' reports;
- fused by the loop: the kernels substitute_kernels / a kernel pack called;
- attention: FA4 (flash_attn.cute) forward, backward and their pre/post-processing;
- GEMM: nvjet / cuBLAS(Lt) / CUTLASS GEMMs, incl. Dist-MoE's grouped GEMMs with fused dispatch / combine;
- glue: everything else, split into multi_tensor_apply (optimizer step, grad norm, foreach copies),
  memcpy / memset, sorts and indexing_backward (deterministic index_put / index_add, MoE argsort), and
  other kernels.

Roofline: if every glue kernel took no time, the step would shrink by the glue share. The loop captured
(base - pack) / glue(base) of that. Times are sums of kernel durations; programmatic dependent launch
overlaps make sums slightly larger than busy time.
"""

import collections
import glob
import gzip
import json
import re
import sys

COMM = ("c10d::", "_c10d_functional::", "nccl", "record_param_comms")
CATEGORIES = [
    ("fused by the loop", re.compile(r"_substitute_kernels_|_kernel_pack_pat_")),
    ("attention", re.compile(r"flash_attn|FlashAttention|fmha")),
    ("GEMM", re.compile(r"nvjet|gemm|Gemm|GEMM|cublas|cutlass13device_kernel|xmma")),
    ("glue: multi_tensor_apply", re.compile(r"multi_tensor_apply")),
    ("glue: memcpy / memset", re.compile(r"^Memcpy|^Memset")),
    (
        "glue: sort + indexing_backward",
        re.compile(r"indexing_backward|RadixSort|radix_sort|bitonicSort"),
    ),
    ("glue: other kernels", re.compile(r".")),
]
NON_GLUE = ("attention", "GEMM")


def trace_of(run):
    paths = sorted(glob.glob(f"{run}/**/rank0_trace.json.gz", recursive=True))
    if len(paths) != 1:
        raise ValueError(f"{run}: expected one rank0 trace, found {paths}")
    return paths[0]


def classify(run):
    """ms per category, and ms per (category, kernel name)."""
    ev = json.load(gzip.open(trace_of(run), "rt"))["traceEvents"]
    comm = collections.defaultdict(list)
    for e in ev:
        if (
            e.get("ph") == "X"
            and e.get("cat") == "cpu_op"
            and any(k in e["name"] for k in COMM)
        ):
            comm[e["tid"]].append((e["ts"], e["ts"] + e["dur"]))
    launch = {
        e["args"].get("correlation"): e
        for e in ev
        if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "args" in e
    }
    cat_ms, name_ms = collections.Counter(), collections.Counter()
    for k in ev:
        if k.get("ph") != "X" or k.get("cat") not in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ):
            continue
        name = k["name"]
        lau = launch.get(k.get("args", {}).get("correlation"))
        if lau and any(s <= lau["ts"] <= t for s, t in comm.get(lau["tid"], ())):
            cat = "communication (fake-PG emulation)"
        else:
            cat = next(c for c, pat in CATEGORIES if pat.search(name))
        cat_ms[cat] += k["dur"] / 1e3
        name_ms[(cat, re.sub(r"0x[0-9a-f]+", "", name)[:100])] += k["dur"] / 1e3
    return cat_ms, name_ms


def summary(cat_ms):
    step = sum(v for c, v in cat_ms.items() if not c.startswith("communication"))
    glue = sum(
        v for c, v in cat_ms.items() if c.startswith("glue") or c == "fused by the loop"
    )
    return step, glue


def main():
    args = sys.argv[1:]
    out_json = args.pop(args.index("--json") + 1) if "--json" in args else None
    if "--json" in args:
        args.remove("--json")
    top = int(args.pop(args.index("--top") + 1)) if "--top" in args else 0
    if "--top" in args:
        args.remove("--top")
    results = {}
    for spec in args:
        model, runs = spec.split("=", 1)
        base_run, pack_run = runs.split(",")
        base, base_names = classify(base_run)
        pack, pack_names = classify(pack_run)
        step_b, glue_b = summary(base)
        step_p, glue_p = summary(pack)
        nonglue_b = sum(base[c] for c in NON_GLUE)
        nonglue_p = sum(pack[c] for c in NON_GLUE)
        mta_b = base["glue: multi_tensor_apply"]
        idx_b = base["glue: sort + indexing_backward"]
        r = {
            "base_run": base_run,
            "pack_run": pack_run,
            "base_ms": dict(base),
            "pack_ms": dict(pack),
            "step_ms": [step_b, step_p],
            "glue_ms": [glue_b, glue_p],
            "gemm_attention_ms": [nonglue_b, nonglue_p],
            "glue_share": glue_b / step_b,
            "glue_share_excl_multi_tensor_apply": (glue_b - mta_b) / step_b,
            "achieved_reduction": (step_b - step_p) / step_b,
            "captured_of_glue": (step_b - step_p) / glue_b,
            "captured_of_glue_excl_multi_tensor_apply": (step_b - step_p)
            / (glue_b - mta_b),
            "glue_share_excl_multi_tensor_apply_and_sort_index": (
                glue_b - mta_b - idx_b
            )
            / step_b,
        }
        results[model] = r
        print(f"\n== {model}")
        cats = [c for c, _ in CATEGORIES] + ["communication (fake-PG emulation)"]
        print(f"  {'category':40s} {'base ms':>9s} {'share':>6s} {'pack ms':>9s}")
        for c in cats:
            share = (
                ""
                if c.startswith("communication")
                else f"{100 * base[c] / step_b:5.1f}%"
            )
            print(f"  {c:40s} {base[c]:9.1f} {share:>6s} {pack[c]:9.1f}")
        print(
            f"  {'step (excl. communication)':40s} {step_b:9.1f} {'':6s} {step_p:9.1f}"
        )
        print(
            f"  glue share (roofline) {100 * r['glue_share']:.1f}% "
            f"({100 * r['glue_share_excl_multi_tensor_apply']:.1f}% without multi_tensor_apply); "
            f"achieved -{100 * r['achieved_reduction']:.1f}%; captured "
            f"{100 * r['captured_of_glue']:.0f}% of the glue "
            f"({100 * r['captured_of_glue_excl_multi_tensor_apply']:.0f}% without multi_tensor_apply); "
            f"GEMM + attention {nonglue_b:.1f} -> {nonglue_p:.1f} ms"
        )
        if top:
            print("  top remaining glue in the pack run:")
            rows = [(v, n) for (c, n), v in pack_names.items() if c.startswith("glue")]
            for v, n in sorted(rows, reverse=True)[:top]:
                print(f"    {v:8.2f}  {n}")
    if out_json:
        json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
