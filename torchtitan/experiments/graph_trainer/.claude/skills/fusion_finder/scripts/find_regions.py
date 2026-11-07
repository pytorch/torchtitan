# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Discover fusion-region candidates from the FX graph + node table.

Method (graph substrate only):
  * "Glue" nodes = call_function nodes that are not GEMM, attention or
    collectives: memory-bound kernels plus kernel-free views / allocations.
  * Glue->glue *tensor* dataflow edges are classified:
      - same phase, same layer, same module FQN   -> L1 edge (intra-module)
      - same phase, same layer, different FQN      -> XM edge (cross-module)
      - same phase, different layer / root module  -> XL edge (cross-layer)
      - recompute -> backward                      -> RB edge
      - forward -> recompute/backward              -> saved activation (no fusion)
    Cut rules: SymInt and CPU-tensor edges, edges out of allocation nodes
    (out= writers have no FX edge to readers), and edges into FSDP bucket
    pack ops except from casts that feed only the pack.
  * Candidate kinds:
      L1      connected components over L1 edges
      XM-max  connected components over L1+XM edges with >=2 kernel parts
      XM      pairs of L1 components joined by XM edges
      XL      pairs of L1 components joined by XL edges
      RB      pairs (recompute L1, backward L1) joined by RB edges
      T3-epi  L1 component + the GEMM whose output it consumes
      T3-pro  L1 component + the GEMM(s) that consume all its outputs
  * Instances: candidates with the same signature (kind, normalized module
    FQNs, op targets, shapes; layer index and unbacked sizes abstracted) are
    one region with N instances per step.

Outputs analysis/clusters.pkl and analysis/candidates.json.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from graph_lib import (
    gemm_peak_tflops,
    iter_arg_nodes,
    load_nodes,
    normalize_fqn,
    tensor_bytes,
    tensor_vals,
)

GLUE_CATS = {"memory_bound_glue", "view", "no_kernel"}
ALLOC_TARGETS = {
    "aten.empty.memory_format",
    "aten.new_empty.default",
    "aten.empty_strided.default",
}
# Consumers that read only shape/dtype/device of their input (no data).
SHAPE_ONLY_TARGETS = {
    "aten.zeros_like.default",
    "aten.ones_like.default",
    "aten.empty_like.default",
    "aten.full_like.default",
    "aten.new_zeros.default",
    "aten.new_ones.default",
    "aten.new_full.default",
    "aten.new_empty.default",
    "aten.sym_size.int",
    "aten.sym_numel.default",
}
# Ops that end a fusible group in practice: data-dependent output sizes,
# host syncs, opaque custom ops, sorting/selection, sort-based deterministic
# accumulation. XM-max components do not grow across them.
BARRIER_TARGETS = {
    "aten.sort.stable",
    "aten.sort.default",
    "aten.topk.default",
    "aten.repeat_interleave.Tensor",
    "aten.nonzero.default",
    "aten._local_scalar_dense.default",
    "torchtitan.deterministic_scatter_add.default",
    "aten.index_put.default",
    "aten.index_put_.default",
    "aten.embedding_dense_backward.default",
}


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def group_key(n):
    if n["layer"] is not None:
        return ("L", n["layer"])
    root = (n["module_fqn"] or "<none>").split(".")[0]
    return ("R", root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="capture directory (after build_node_table.py)")
    args = ap.parse_args()
    analysis = os.path.join(args.root, "analysis")
    cal = json.load(open(os.path.join(args.root, "roofline.json")))["roofline"]
    BW = cal["hbm_gbps"]

    nodes, by = load_nodes(
        os.path.join(args.root, "graph_dump", "final_nodes.jsonl.gz")
    )
    nt = pd.read_pickle(os.path.join(analysis, "node_table.pkl"))
    sym = json.load(open(os.path.join(analysis, "symbols.json")))["symbols"]
    info = nt.set_index("name").to_dict("index")
    output_node = next(n for n in nodes if n["op"] == "output")
    graph_outputs = set(output_node["input_nodes"])
    order = {n["name"]: n["idx"] for n in nodes}

    def cat(name):
        r = info.get(name)
        return r["category"] if r else "none"

    def kus(name):
        r = info.get(name)
        return float(r["kernel_us"]) if r else 0.0

    def is_kernel_node(name):
        return kus(name) > 0

    def is_tensor(name):
        n = by.get(name)
        if n is None:
            return False
        tv = tensor_vals(n["val"])
        return len(tv) > 0 and all(
            t.get("device", "cuda").startswith("cuda") for t in tv
        )

    def vbytes(name, unique=True):
        n = by[name]
        tot = 0
        for t in tensor_vals(n["val"]):
            b = tensor_bytes(t, unique=unique, sym=sym)
            if b is None:
                return None
            tot += b
        if tot == 0 and n["target"].endswith(".out") and isinstance(n["kwargs"], dict):
            for o in iter_arg_nodes(n["kwargs"].get("out") or []):
                tot += vbytes(o, unique) or 0
        return tot

    # dict (graph order) rather than set, so component and candidate ids are reproducible.
    glue = dict.fromkeys(
        n["name"]
        for n in nodes
        if n["op"] == "call_function" and cat(n["name"]) in GLUE_CATS
    )

    # Backward accumulation nodes carry no module_fqn: infer it from the first
    # input (else user) that has an original annotation.
    for n in nodes:
        n["fqn_inferred"] = False
    for n in nodes:
        if n["op"] != "call_function" or n["module_fqn"]:
            continue
        src = next(
            (
                by[i]
                for i in n["input_nodes"]
                if by[i]["module_fqn"] and not by[i]["fqn_inferred"]
            ),
            None,
        )
        if src is None:
            src = next((by[u] for u in n["users"] if by[u]["module_fqn"]), None)
        if src is None:
            src = next((by[i] for i in n["input_nodes"] if by[i]["module_fqn"]), None)
        if src is not None:
            n["module_fqn"] = src["module_fqn"]
            n["layer"] = src["layer"]
            n["fqn_inferred"] = True
    json.dump(
        {n["name"]: n["module_fqn"] for n in nodes if n.get("fqn_inferred")},
        open(os.path.join(analysis, "inferred_fqns.json"), "w"),
    )

    pack_nodes = {n["name"] for n in nodes if n["target"].startswith("bucketing.")}
    pack_feeders = {
        u
        for b in pack_nodes
        for u in by[b]["input_nodes"]
        if u in glue and len(by[u]["users"]) == 1
    }

    # ---- fusion stage: number of non-glue ops on the longest path ----
    # Two glue nodes with the same stage have no GEMM/attention/collective on
    # any path between them, so a group whose members share one stage is a
    # legal single-kernel fusion with respect to non-glue ops.
    stage = {}
    for n in nodes:
        s_ = 0
        for p in n["input_nodes"]:
            if p in stage:
                s_ = max(s_, stage[p] + (0 if p in glue else 1))
        stage[n["name"]] = s_

    # ---- classify edges ----
    uf1 = UF()  # L1
    uf2 = UF()  # L1 + XM
    edges = defaultdict(list)  # kind -> [(u, v)]
    for v in glue:
        uf1.find(v)
        uf2.find(v)
        nv = by[v]
        for u in nv["input_nodes"]:
            if u not in glue or not is_tensor(u):
                continue
            nu = by[u]
            if nu["target"] in ALLOC_TARGETS or nv["target"] in SHAPE_ONLY_TARGETS:
                continue
            if v in pack_feeders or (v in pack_nodes and u not in pack_feeders):
                continue
            pu, pv = nu["phase"], nv["phase"]
            same_stage = stage[u] == stage[v]
            if pu == pv:
                if group_key(nu) == group_key(nv):
                    if not same_stage:
                        edges["cross-stage"].append((u, v))
                    elif nu["module_fqn"] == nv["module_fqn"]:
                        uf1.union(u, v)
                        uf2.union(u, v)
                    else:
                        edges["XM"].append((u, v))
                        if (
                            nu["target"] not in BARRIER_TARGETS
                            and nv["target"] not in BARRIER_TARGETS
                        ):
                            uf2.union(u, v)
                else:
                    edges["XL" if same_stage else "XL-cross-stage"].append((u, v))
            elif pu == "recompute" and pv == "backward":
                edges["RB" if same_stage else "RB-cross-stage"].append((u, v))
            else:
                edges[f"saved:{pu}->{pv}"].append((u, v))

    def components(uf):
        mem = defaultdict(list)
        for v in glue:
            mem[uf.find(v)].append(v)
        out = list(mem.values())
        for m in out:
            m.sort(key=order.__getitem__)
        return out

    def absorb(uf, same_fqn):
        """Merge a group into its only same-phase glue consumer group when it
        has no non-glue same-phase users and a stage <= the consumer's: no
        path can leave and re-enter, so the merge stays legal (e.g. a RoPE
        frequency gather on graph inputs into the RoPE multiply)."""
        merged = 0
        for _ in range(4):
            comp_of = {}
            for mem in components(uf):
                r = uf.find(mem[0])
                for m in mem:
                    comp_of[m] = r
            changed = False
            groups = defaultdict(list)
            for m, r in comp_of.items():
                groups[r].append(m)
            for r, mem in groups.items():
                ph = by[mem[0]]["phase"]
                gk = group_key(by[mem[0]])
                st = stage[mem[0]]
                targets = set()
                ok = True
                for m in mem:
                    for x in by[m]["users"]:
                        nx = by[x]
                        if comp_of.get(x) == r:
                            continue
                        if nx["phase"] != ph:
                            continue  # value saved for a later phase: just an output
                        if x not in glue:
                            if nx["op"] == "output":
                                continue
                            ok = False
                            break
                        if nx["target"] in SHAPE_ONLY_TARGETS:
                            continue
                        targets.add(comp_of[x])
                    if not ok:
                        break
                if not ok or len(targets) != 1:
                    continue
                t = next(iter(targets))
                tm = groups[t]
                if group_key(by[tm[0]]) != gk or stage[tm[0]] < st:
                    continue
                if same_fqn and by[tm[0]]["module_fqn"] != by[mem[0]]["module_fqn"]:
                    continue
                if not same_fqn and any(
                    by[m]["target"] in BARRIER_TARGETS for m in mem
                ):
                    continue
                uf.union(mem[0], tm[0])
                merged += 1
                changed = True
            if not changed:
                break
        return merged

    n_abs1 = absorb(uf1, same_fqn=True)
    # every (absorbed) L1 group must be contained in its XM-max group
    for mem in components(uf1):
        for m in mem[1:]:
            uf2.union(mem[0], m)
    n_abs2 = absorb(uf2, same_fqn=False)
    print("absorbed groups: L1", n_abs1, "XM", n_abs2)

    def base_of(m, mset, seen=None):
        if is_kernel_node(m) or cat(m) == "memory_bound_glue":
            return m
        seen = seen if seen is not None else set()
        if m in seen:
            return None
        seen.add(m)
        for u in by[m]["input_nodes"]:
            if u in mset and is_tensor(u):
                return base_of(u, mset, seen)
        return None

    def stats(mem, gemm_compute_us=0.0):
        mset = set(mem)
        measured = sum(kus(m) for m in mem)

        def read_bytes_via(m, seen):
            """Bytes a cluster reads through view member ``m``: the views that
            kernel members (inside the cluster) actually consume."""
            if m in seen:
                return 0
            seen.add(m)
            tot = 0
            for x in by[m]["users"]:
                if x not in mset:
                    continue
                if is_kernel_node(x) or cat(x) == "memory_bound_glue":
                    b = vbytes(m)
                    return None if b is None else b  # the view itself is read
                r = read_bytes_via(x, seen)
                if r is None:
                    return None
                tot += r
            return tot

        ext_in = {}
        for m in mem:
            for u in by[m]["input_nodes"]:
                if u in mset or not is_tensor(u):
                    continue
                if by[u]["target"] in ALLOC_TARGETS:
                    continue  # buffer is written, not read
                if cat(m) in ("view", "no_kernel") and not is_kernel_node(m):
                    b = read_bytes_via(m, set())
                else:
                    b = vbytes(u)
                cap = vbytes(u)
                if b is None or cap is None:
                    ext_in[u] = None
                    continue
                prev = ext_in.get(u) or 0
                ext_in[u] = min(prev + b, cap)
        ext_out = {}
        for m in mem:
            users_out = [x for x in by[m]["users"] if x not in mset]
            if not users_out and m not in graph_outputs:
                continue
            if not is_tensor(m) and not by[m]["target"].endswith(".out"):
                continue
            b = base_of(m, mset)
            if b is None:
                continue
            ext_out[b] = vbytes(b, unique=False)
        # FSDP all-gather pack writes only the local shards into the bucket
        # (its val is the full all-gather buffer).
        for b_ in list(ext_out):
            if by[b_]["target"] == "bucketing._pre_bucket_all_gather.default":
                ext_out[b_] = sum(
                    vbytes(i) or 0 for i in by[b_]["input_nodes"] if is_tensor(i)
                )
        # out= / in-place writers into external buffers always write
        for m in mem:
            t = by[m]["target"]
            if (
                t.endswith(".out") or t.endswith("_.default") or t.endswith("_.Tensor")
            ) and m not in ext_out:
                if is_kernel_node(m):
                    ext_out[m] = vbytes(m, unique=False)
        unknown = any(v is None for v in ext_in.values()) or any(
            v is None for v in ext_out.values()
        )
        bin_ = sum(v or 0 for v in ext_in.values())
        bout = sum(v or 0 for v in ext_out.values())
        ideal = max((bin_ + bout) / (BW * 1e3), gemm_compute_us)
        return dict(
            n_members=len(mem),
            n_kernel_nodes=sum(1 for m in mem if is_kernel_node(m)),
            n_kernels=int(sum(info[m]["n_kernels"] for m in mem if m in info)),
            measured_us=measured,
            bytes_in=bin_,
            bytes_out=bout,
            bytes_unknown=unknown,
            ideal_us=ideal,
            ext_inputs=sorted(ext_in, key=order.__getitem__),
            ext_outputs=sorted(ext_out, key=order.__getitem__),
        )

    def op_sig(mem):
        sig = []
        for m in mem:
            if not is_kernel_node(m) and cat(m) not in ("memory_bound_glue", "gemm"):
                continue
            n = by[m]
            shp = ";".join(str(t["shape"]) for t in tensor_vals(n["val"]))
            shp = re.sub(r"'[^']*u\d+[^']*'", "U", shp)
            sig.append((normalize_fqn(n["module_fqn"]), n["target"], n["phase"], shp))
        return tuple(sig)

    def describe(mem, kind, sig, **extra):
        kmem = [
            m
            for m in mem
            if is_kernel_node(m) or cat(m) in ("memory_bound_glue", "gemm")
        ]
        fq = sorted({by[m]["module_fqn"] for m in (kmem or mem)})
        d = dict(
            kind=kind,
            members=mem,
            phases=sorted({by[m]["phase"] for m in mem}),
            layers=sorted({by[m]["layer"] for m in mem if by[m]["layer"] is not None}),
            groups=sorted({str(group_key(by[m])) for m in mem}),
            fqns=fq,
            fqn_patterns=sorted({normalize_fqn(f) for f in fq}),
            sig=sig,
            first_idx=order[mem[0]],
        )
        d.update(extra)
        return d

    allc = []

    def add(c):
        c["cid"] = len(allc)
        allc.append(c)

    # L1
    l1_of = {}
    for mem in components(uf1):
        cid = len(allc)
        for m in mem:
            l1_of[m] = cid
        add(describe(mem, "L1", ("L1", op_sig(mem)), **stats(mem)))

    # XM-max
    for mem in components(uf2):
        parts = sorted({l1_of[m] for m in mem})
        kparts = [p for p in parts if allc[p]["measured_us"] > 0]
        if len(kparts) < 2:
            continue
        add(describe(mem, "XM-max", ("XM-max", op_sig(mem)), parts=parts, **stats(mem)))

    # pairs
    seen_pairs = set()
    for kind in ("XM", "XL", "RB"):
        for u, v in edges[kind]:
            a, b = l1_of[u], l1_of[v]
            if (kind, a, b) in seen_pairs or a == b:
                continue
            seen_pairs.add((kind, a, b))
            A, B = allc[a], allc[b]
            if A["measured_us"] <= 0 and B["measured_us"] <= 0:
                continue
            mem = sorted(set(A["members"]) | set(B["members"]), key=order.__getitem__)
            add(
                describe(
                    mem,
                    kind,
                    (kind, A["sig"], B["sig"]),
                    parts=[a, b],
                    link=[u, v],
                    **stats(mem),
                )
            )

    # T3: GEMM epilogue / prologue around L1 clusters
    for g in nodes:
        gname = g["name"]
        if cat(gname) != "gemm":
            continue
        gemm_compute_us = (info[gname]["flops"] or 0) / (
            gemm_peak_tflops(g, by, cal) * 1e6
        )
        for c in sorted({l1_of[x] for x in g["users"] if x in l1_of}):
            C = allc[c]
            if C["measured_us"] <= 0 or gname not in C["ext_inputs"]:
                continue
            mem = sorted(set(C["members"]) | {gname}, key=order.__getitem__)
            add(
                describe(
                    mem,
                    "T3-epi",
                    ("T3-epi", op_sig(mem)),
                    parts=[c],
                    gemm=gname,
                    gemm_us=kus(gname),
                    gemm_ideal_us=gemm_compute_us,
                    **stats(mem, gemm_compute_us),
                )
            )
        for c in sorted({l1_of[x] for x in g["input_nodes"] if x in l1_of}):
            C = allc[c]
            if C["measured_us"] <= 0:
                continue
            mset = set(C["members"])

            def real_consumers(x, seen):
                # follow kernel-free views (transpose/t/view) to their consumers
                if x in seen:
                    return set()
                seen.add(x)
                if cat(x) in ("view", "no_kernel") and not is_kernel_node(x):
                    out = set()
                    for y in by[x]["users"]:
                        out |= real_consumers(y, seen)
                    return out
                return {x}

            consumers = set()
            for m in C["members"]:
                for x in by[m]["users"]:
                    if x not in mset:
                        consumers |= real_consumers(x, set())
            if not consumers or not all(cat(x) == "gemm" for x in consumers):
                continue
            mem = sorted(mset | {gname}, key=order.__getitem__)
            add(
                describe(
                    mem,
                    "T3-pro",
                    ("T3-pro", op_sig(mem)),
                    parts=[c],
                    gemm=gname,
                    other_gemm_consumers=sorted(consumers - {gname}),
                    gemm_us=kus(gname),
                    gemm_ideal_us=gemm_compute_us,
                    **stats(mem, gemm_compute_us),
                )
            )

    df = pd.DataFrame(allc)
    df["sig_str"] = df["sig"].map(lambda s: json.dumps(s, default=str))
    df.to_pickle(os.path.join(analysis, "clusters.pkl"))
    json.dump(
        {k: len(v) for k, v in edges.items()},
        open(os.path.join(analysis, "edge_counts.json"), "w"),
        indent=1,
    )

    # ---- group by signature ----
    cands = []
    for sig_str, g in df.groupby("sig_str"):
        if g["measured_us"].sum() <= 0:
            continue
        g = g.sort_values("first_idx")
        r0 = g.iloc[0]
        inst = len(g)
        meas = g["measured_us"].sum()
        ideal = g["ideal_us"].sum()
        parts_sig = None
        if isinstance(r0.get("parts"), list):
            parts_sig = [df.loc[p, "sig_str"] for p in r0["parts"]]
        cands.append(
            dict(
                sig=sig_str,
                kind=r0["kind"],
                phases=r0["phases"],
                instances=inst,
                layers=sorted({l for ls in g["layers"] for l in ls}),
                groups=r0["groups"],
                fqn_patterns=r0["fqn_patterns"],
                example_cid=int(r0["cid"]),
                example_members=list(r0["members"]),
                n_members=int(r0["n_members"]),
                n_kernel_nodes=int(r0["n_kernel_nodes"]),
                kernels_per_instance=float(g["n_kernels"].mean()),
                measured_us_per_instance=meas / inst,
                measured_ms_total=meas / 1e3,
                ideal_us_per_instance=ideal / inst,
                ideal_ms_total=ideal / 1e3,
                gap_ms_total=(meas - ideal) / 1e3,
                bytes_per_instance=float((g["bytes_in"] + g["bytes_out"]).mean()),
                bytes_unknown=bool(g["bytes_unknown"].any()),
                parts_sig=parts_sig,
                gemm_example=r0.get("gemm")
                if isinstance(r0.get("gemm"), str)
                else None,
                gemm_us_per_instance=float(g["gemm_us"].mean())
                if "gemm_us" in g and g["gemm_us"].notna().any()
                else None,
                gemm_ideal_us_per_instance=float(g["gemm_ideal_us"].mean())
                if "gemm_ideal_us" in g and g["gemm_ideal_us"].notna().any()
                else None,
                link_example=r0.get("link")
                if isinstance(r0.get("link"), list)
                else None,
                cids=[int(x) for x in g["cid"]],
            )
        )
    cands.sort(key=lambda c: -c["gap_ms_total"])
    for i, c in enumerate(cands):
        c["cand_id"] = i
    json.dump(
        cands,
        open(os.path.join(analysis, "candidates.json"), "w"),
        indent=1,
        default=str,
    )
    print("clusters", Counter(df["kind"]), "candidates", len(cands))
    print(Counter(c["kind"] for c in cands))
    for c in cands[:50]:
        print(
            f"{c['cand_id']:4d} {c['gap_ms_total']:9.2f} gap meas {c['measured_ms_total']:8.2f} "
            f"x{c['instances']:5d} {c['kind']:7s} {','.join(c['phases']):20s} "
            f"k/i={c['kernels_per_instance']:.0f} {c['fqn_patterns'][:3]}"
        )


if __name__ == "__main__":
    main()
