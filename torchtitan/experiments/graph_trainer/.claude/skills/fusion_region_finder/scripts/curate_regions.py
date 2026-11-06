# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Turn discovery candidates into the region inventory and one lift pass per region.

Usage: ``python curate_regions.py <capture dir> [--annotations annotations.json] [--gpu-busy]``

Selection (no size threshold):
  * L1 components with >= 2 kernel nodes                   (intra-module base units)
  * L1 components with 1 kernel node that launches >= 2 kernels or runs at
    >= 2x its roofline                                       (single-op rewrites)
  * XM-max components                                      (cross-module, barrier-free)
  * XM pairs whose link touches a barrier op               (cross-module through T2 ops)
  * XL pairs, RB pairs                                     (cross-layer, recompute-backward)
  * T3 epilogue / prologue candidates                      (GEMM + adjacent glue)
Candidates whose kernel set equals one of their parts' and exact duplicates are dropped.
Signature variants that differ only in minor details (first/last layer) are merged
when their kernel members have the same ops and shapes.

Writes:
  regions.json              inventory (schema in references/regions_schema.md)
  regions_table.md          ranked table for the report
  fusion_regions/           one folder per region, e.g. g003_rope/:
                              region.json    the region's regions.json entry plus its lift status
                              lift_pass.py   generated topology matchers and the registered
                                             graph pass lift_fusion_region_g003 (liftable only)
                            __init__.py imports every lift pass and defines PASSES and
                            DEFAULT_SELECTION
  analysis/regions_side.json

``--annotations`` is an optional JSON file ``{region key: {title, notes, fusion_kind,
bitwise_notes, blockers}}`` with hand-written text; the key is printed for each region.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict

import pandas as pd
import torch

from find_regions import BARRIER_TARGETS
from graph_lib import load_nodes, normalize_fqn, numel, short_src, tensor_vals
from lift_regions import (
    D,
    DumpNode,
    find_matches,
    is_op,
    lift_hazard,
    load_dump_graph,
    N,
    op_schema,
    S,
    user_of,
    V,
    writes_outside,
)

T2_OPS = {
    "aten.sort.stable",
    "aten.sort.default",
    "aten.topk.default",
    "aten.scatter_.value",
    "aten.scatter_.src",
    "aten.scatter_add.default",
    "aten.scatter_add_.default",
    "aten.index_add.default",
    "aten.index_put.default",
    "aten.index_put_.default",
    "torchtitan.deterministic_scatter_add.default",
    "aten.embedding_dense_backward.default",
    "aten.repeat_interleave.Tensor",
}
REDUCTION_OPS = {
    "aten.sum.dim_IntList",
    "aten.sum.default",
    "aten.mean.dim",
    "aten.mean.default",
    "aten.linalg_vector_norm.default",
    "aten._fused_rms_norm.default",
    "aten._fused_rms_norm_backward.default",
    "aten.native_layer_norm.default",
    "aten.native_layer_norm_backward.default",
    "aten._log_softmax.default",
    "aten._log_softmax_backward_data.default",
    "aten._softmax.default",
    "aten._softmax_backward_data.default",
    "aten.nll_loss_forward.default",
    "aten.nll_loss_backward.default",
    "aten.cumsum.default",
    "aten.amax.default",
    "aten.max.dim",
}
SCATTER_OPS = {
    "torchtitan.deterministic_scatter_add.default",
    "aten.index_put.default",
    "aten.index_put_.default",
    "aten.scatter_add.default",
    "aten.index_add.default",
}
KIND_PRIORITY = {
    "XM-max": 0,
    "XL": 1,
    "RB": 2,
    "XM": 3,
    "L1": 4,
    "T3-epi": 5,
    "T3-pro": 6,
}
HOW_FOUND = {
    "L1": "L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage",
    "L1-single": "L1 component with one kernel node that launches several kernels or runs far from its roofline",
    "XM-max": "XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges)",
    "XM": "XM pair: two L1 components of one phase/layer/stage joined by an edge through a barrier op",
    "XL": "XL pair: two L1 components of one phase and stage joined by an edge across a layer/root-module boundary",
    "RB": "RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage",
    "T3-epi": "T3 epilogue: L1 component consuming a GEMM output",
    "T3-pro": "T3 prologue: L1 component whose outputs feed only GEMMs",
}
TIER_NOTES = {
    "T0": "pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math "
    "and rounding points are kept.",
    "T1": "contains reductions ({ops}): the fused kernel must keep eager's reduction order/tree to be bitwise identical.",
    "T2": "order-sensitive data movement ({ops}): accumulation / selection order must match eager exactly.",
    "T3": "contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager "
    "plus an epilogue that rounds at the same points.",
}
SETUP_CATEGORIES = [
    "gemm",
    "attention",
    "fused_moe",
    "fake_comm",
    "memory_bound_glue",
    "optimizer",
    "other",
]


def short_op(t):
    return (
        t.replace("aten.", "")
        .replace(".default", "")
        .replace("torchtitan.", "tt.")
        .replace("bucketing.", "bkt.")
    )


def _abstract_shape(s):
    """Shape string with unbacked size expressions (u123 + ...) abstracted."""
    return re.sub(r"'[^']*u\d+[^']*'", "N", s)


def _reentry_path(member_names, by, order):
    """External nodes on a path that leaves the member set and re-enters it, or [] if none.

    Such a region cannot be one kernel (the external op runs in between), and lifting it
    into one call_module would create a cycle.
    """
    members = set(member_names)
    last = max(order[m] for m in members)
    parent = {}
    stack = [
        (u, None)
        for m in members
        for u in by[m]["users"]
        if u not in members and order[u] < last
    ]
    while stack:
        x, p = stack.pop()
        if x in parent:
            continue
        parent[x] = p
        for u in by[x]["users"]:
            if u in members:
                path = [x]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1]
            if order[u] < last and u not in parent:
                stack.append((u, x))
    return []


def region_family(r):
    """Short name of a region: its innermost module, plus its GEMM kind or single op.

    E.g. ``rope``, ``w13_grouped_gemm``, ``attention_cat``. Used in folder names.
    """
    fqns = [f for f in r["location"]["module_fqn_pattern"].split(", ") if f]
    parts = [[p for p in f.split(".") if p != "*"] for f in fqns]
    parts = sorted((p for p in parts if p), key=lambda p: (-len(p), p))
    name = parts[0][-1] if parts else "model"
    ops = set(r["ops"])
    if "aten._grouped_mm.default" in ops:
        name += "_grouped_gemm"
    elif ops & {"aten.mm.default", "aten.mm.dtype", "aten.addmm.default"}:
        name += "_gemm"
    elif len(ops) == 1:
        name += "_" + short_op(next(iter(ops))).split(".")[0]
    return re.sub(r"[^a-z0-9_]", "_", name.lower())


_UNMATCHABLE = object()


# Size arguments typed int[] instead of SymInt[] in their schema.
# TODO: make _fused_rms_norm's normalized_shape SymInt[] upstream, as rms_norm's is.
_INT_LIST_SIZE_ARGS = {
    ("aten::_fused_rms_norm", "normalized_shape"),
    ("aten::_fused_rms_norm_backward", "normalized_shape"),
}


def _literal_slot(schema, arg):
    """S if ``arg`` holds sizes, V if it is a scalar operand, D if it is a device, None if
    literals must be equal.

    Sizes and scalar operands carry the shapes and config constants of the capture (hidden
    size, number of experts, FSDP group size, ...) and the rank (bucket rank, shard
    offsets); devices carry the rank's device index. Matching them as any value (devices:
    any index of the same type) finds the region in other model variants and layouts and
    on every rank. Lifting keeps numerics either way.
    """
    t = arg.real_type
    while isinstance(t, (torch._C.OptionalType, torch._C.ListType)):
        t = t.getElementType()
    if (
        isinstance(t, torch._C.SymIntType)
        or (schema.name, arg.name) in _INT_LIST_SIZE_ARGS
    ):
        return S
    if isinstance(t, (torch._C.TensorType, torch._C.NumberType)):
        return V
    if isinstance(t, torch._C.DeviceObjType):
        return D
    return None


def _template(v, slot=None):
    """is_op template of a dumped argument: N for nodes, ``slot`` (S or V) for numbers and
    D(type) for devices if given, other literals as dumped."""
    if isinstance(v, DumpNode):
        return N
    if isinstance(v, list):
        items = [_template(x, slot) for x in v]
        return _UNMATCHABLE if any(x is _UNMATCHABLE for x in items) else items
    if isinstance(v, dict):
        if set(v) == {"repr"}:  # a literal the dump could not encode
            return _UNMATCHABLE
        items = {k: _template(x, slot) for k, x in v.items()}
        return _UNMATCHABLE if any(x is _UNMATCHABLE for x in items.values()) else items
    if isinstance(v, bool):
        return v
    # Graph dumps write symbolic sizes as strings.
    if slot is S and isinstance(v, (int, str)):
        return S
    if slot is V and isinstance(v, (int, float)):
        return V
    if slot is D and isinstance(v, str):
        return D(v.split(":")[0])
    if isinstance(v, float) and math.isnan(v):
        return _UNMATCHABLE
    return v


def _templates(node):
    """is_op templates of ``node``'s args and kwargs, with slots from its op schema."""
    schema = op_schema(node)
    if schema is None:
        return _template(node.args), _template(node.kwargs)
    slots = [_literal_slot(schema, a) for a in schema.arguments]
    by_name = {a.name: slot for a, slot in zip(schema.arguments, slots)}
    args = [
        _template(v, slots[i] if i < len(slots) else None)
        for i, v in enumerate(node.args)
    ]
    kwargs = {k: _template(v, by_name.get(k)) for k, v in node.kwargs.items()}
    if any(x is _UNMATCHABLE for x in [*args, *kwargs.values()]):
        return _UNMATCHABLE, _UNMATCHABLE
    return args, kwargs


def _captured(n, ta, tk):
    """Comment naming the captured node, with its literals where slots stand for them."""
    ea, ek = _template(n.args), _template(n.kwargs)
    if _UNMATCHABLE in (ea, ek) or (ta, tk) == (ea, ek):
        return f"    # {n.name} (as captured)"
    kw = f", {_py(ek)}" if ek else ""
    return f"    # {n.name} (as captured: {_py(ea)}{kw})"


def _py(v):
    """Python source of a template value."""
    if v is N or v is S or v is V or isinstance(v, D):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_py(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_py(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float) and math.isinf(v):
        return 'float("inf")' if v > 0 else 'float("-inf")'
    return repr(v)


def _slots(value, expr, path):
    """(node, access expression, path) for every node inside a dumped argument structure."""
    if isinstance(value, DumpNode):
        yield value, expr, path
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _slots(v, f"{expr}[{i}]", path + (i,))
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _slots(v, f"{expr}[{k!r}]", path + (k,))


def _node_slots(n, var):
    return list(_slots(n.args, f"{var}.args", ())) + list(
        _slots(n.kwargs, f"{var}.kwargs", ())
    )


def _gen_matcher(inst, dn, fname):
    """Python matcher for one instance (given by node names): (source, anchor op) or (None, reason).

    The anchor is the sink with the most ancestors inside the instance. The matcher walks
    producers from there; members that are not ancestors of visited nodes are reached as
    users of visited ones. Each op and literal is checked, a member reached twice must be
    the same node, and so must an external input used twice.
    """
    members = sorted(dn[m] for m in inst)
    mset = set(members)

    def num_ancestors(x):
        seen, stack = set(), [x]
        while stack:
            for s, _, _ in _node_slots(stack.pop(), ""):
                if s in mset and s not in seen:
                    seen.add(s)
                    stack.append(s)
        return len(seen)

    sinks = [m for m in members if not any(u in mset for u in m.users)]
    anchor = max(sinks, key=lambda m: (num_ancestors(m), m.idx))
    var, ext, body, work = {anchor: "n0"}, {}, [], [anchor]
    while True:
        while work:
            m = work.pop(0)
            ta, tk = _templates(m)
            if ta is _UNMATCHABLE or tk is _UNMATCHABLE:
                return None, f"{m.name} has a literal argument that cannot be matched"
            kw = f", {_py(tk)}" if tk else ""
            body += [
                _captured(m, ta, tk),
                f"    if not is_op({var[m]}, {m.target!r}, {_py(ta)}{kw}):",
                "        return None",
            ]
            for s, expr, _ in _node_slots(m, var[m]):
                known = var if s in mset else ext
                if s in known:
                    body += [f"    if {expr} is not {known[s]}:", "        return None"]
                elif s in mset:
                    var[s] = f"n{len(var)}"
                    body.append(f"    {var[s]} = {expr}")
                    work.append(s)
                else:
                    ext[s] = f"x{len(ext)}"
                    body.append(f"    {ext[s]} = {expr}")
        rest = [m for m in members if m not in var]
        if not rest:
            break
        for u in rest:
            link = next(
                ((s, path) for s, _, path in _node_slots(u, "") if s in var), None
            )
            ta, tk = _templates(u)
            if link is not None and ta is not _UNMATCHABLE and tk is not _UNMATCHABLE:
                skip = ", ".join(var.values())
                var[u] = f"n{len(var)}"
                kw = f", {_py(tk) if tk else None}"
                body += [
                    f"    {var[u]} = user_of({var[link[0]]}, {link[1]!r}, {u.target!r}, {_py(ta)}{kw}, skip=({skip},))",
                    f"    if {var[u]} is None:",
                    "        return None",
                ]
                work.append(u)
                break
        else:
            return None, "the instance is not connected"
    body.append(
        f"    return [{', '.join(var[m] for m in members)}], [{', '.join(ext.values())}]"
    )
    doc = f'    """One instance, anchored at its {anchor.target} node: (members, external inputs) or None."""'
    return "\n".join([f"def {fname}(n0):", doc] + body) + "\n", anchor.target


def _lift_imports(matchers):
    """Names the generated lift pass imports from lift_regions."""
    calls = [
        line
        for _, src, _ in matchers
        for line in src.splitlines()
        if "is_op(" in line or "user_of(" in line
    ]
    slots = [
        x for x in ("S", "V") if any(re.search(rf"[\[ ]{x}[\],]", c) for c in calls)
    ]
    if any(re.search(r"[\[ ]D\(", c) for c in calls):
        slots.append("D")
    return ", ".join(
        sorted(["is_op", "lift_matches", "N", "user_of", *slots], key=str.lower)
    )


def _region_matchers(instances, dn, dump_nodes):
    """Matchers that find every finder instance: ([(anchor op, source, fn)], sites) or (None, reason)."""
    matchers = []
    wanted = [frozenset(i) for i in instances]
    uncovered = list(wanted)
    while uncovered:
        fname = f"match_v{len(matchers)}"
        src, anchor = _gen_matcher(sorted(uncovered[0]), dn, fname)
        if src is None:
            return None, anchor
        ns = {"D": D, "N": N, "S": S, "V": V, "is_op": is_op, "user_of": user_of}
        exec(src, ns)  # noqa: S102  (generated matcher source)
        matchers.append((anchor, src, ns[fname]))
        sites = find_matches(dump_nodes, [(a, fn) for a, _, fn in matchers])
        claimed = {n.name for site in sites for n in site}
        # An equivalent overlapping site can claim an instance's nodes first; that counts.
        if not uncovered[0] & claimed:
            return None, "a matcher does not find the instance it was generated from"
        uncovered = [i for i in wanted if not i & claimed]
    return matchers, sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="capture directory (after find_regions.py)")
    ap.add_argument(
        "--annotations",
        help="JSON file of hand-written region text keyed by region key",
    )
    ap.add_argument(
        "--gpu-busy",
        action="store_true",
        help="another process used the GPU during the capture",
    )
    args = ap.parse_args()
    R = args.root
    A = os.path.join(R, "analysis")
    cands = json.load(open(os.path.join(A, "candidates.json")))
    df = pd.read_pickle(os.path.join(A, "clusters.pkl")).set_index("cid", drop=False)
    nt = pd.read_pickle(os.path.join(A, "node_table.pkl")).set_index("name", drop=False)
    budget = json.load(open(os.path.join(A, "step_budget.json")))
    roof = json.load(open(os.path.join(R, "roofline.json")))
    cal = roof["roofline"]
    nodes, by = load_nodes(os.path.join(R, "graph_dump", "final_nodes.jsonl.gz"))
    order = {n["name"]: n["idx"] for n in nodes}
    for name, fqn in json.load(open(os.path.join(A, "inferred_fqns.json"))).items():
        by[name]["module_fqn"] = fqn
        m = re.match(r"^layers\.(\d+)", fqn)
        by[name]["layer"] = int(m.group(1)) if m else None
    step_ms = budget["gpu_window_ms"]
    kus = nt["kernel_us"].to_dict()
    cat = nt["category"].to_dict()

    def kset(cid):
        return frozenset(m for m in df.loc[cid, "members"] if kus.get(m, 0) > 0)

    # ---- select ----
    sel = []
    for c in cands:
        if c["measured_ms_total"] <= 0:
            continue
        kind = c["kind"]
        r0 = df.loc[c["example_cid"]]
        if kind == "L1" and c["n_kernel_nodes"] < 2:
            slow = c["measured_us_per_instance"] >= 2 * max(
                c["ideal_us_per_instance"], 1e-9
            )
            if (
                c["n_kernel_nodes"] < 1
                or not (c["kernels_per_instance"] >= 2 or slow)
                or c["gap_ms_total"] <= 0
            ):
                continue
            c["single_op"] = True
        if kind == "XM":
            u, v = c["link_example"]
            if (
                by[u]["target"] not in BARRIER_TARGETS
                and by[v]["target"] not in BARRIER_TARGETS
            ):
                continue
        if kind in ("XM", "XL", "RB", "XM-max"):
            ex = kset(r0["cid"])
            if any(ex == kset(p) for p in r0["parts"]):
                continue
        c["kset_all"] = frozenset().union(*[kset(cid) for cid in c["cids"]])
        sel.append(c)
    sel.sort(key=lambda c: (KIND_PRIORITY[c["kind"]], -c["gap_ms_total"]))
    seen = {}
    kept = []
    for c in sel:
        if c["kset_all"] in seen:
            seen[c["kset_all"]].setdefault("duplicates", []).append(c["cand_id"])
            continue
        seen[c["kset_all"]] = c
        kept.append(c)
    # XM pairs strictly contained in an XM-max of the same instance set -> drop.
    xmmax = [c for c in kept if c["kind"] == "XM-max"]
    final = [
        c
        for c in kept
        if not (c["kind"] == "XM" and any(c["kset_all"] < x["kset_all"] for x in xmmax))
    ]
    # T3 candidates that share the same glue part and GEMM module pattern -> one.
    t3seen = {}
    final2 = []
    for c in sorted(final, key=lambda x: -x["gap_ms_total"]):
        if c["kind"].startswith("T3"):
            gfqn = normalize_fqn(by[c["gemm_example"]]["module_fqn"])
            k = (c["kind"], json.dumps(c["parts_sig"]), gfqn)
            if k in t3seen:
                t3seen[k].setdefault("duplicates", []).append(c["cand_id"])
                continue
            t3seen[k] = c
        final2.append(c)
    final = final2

    # Merge signature variants that differ only in minor details (first/last layer, a
    # different input producer): same kind, phases, module patterns, ops and shapes.
    # Shapes are part of the key so instances of different sizes (e.g. q and k paths)
    # stay separate regions with honest per-instance numbers.
    def loose_key(c):
        r0 = df.loc[c["example_cid"]]
        km = [m for m in r0["members"] if kus.get(m, 0) > 0]
        return (
            c["kind"],
            tuple(sorted({by[m]["phase"] for m in km})),
            tuple(sorted({normalize_fqn(by[m]["module_fqn"]) for m in km})),
            tuple(sorted({by[m]["target"] for m in km})),
            tuple(
                sorted(
                    {
                        _abstract_shape(
                            ";".join(str(t["shape"]) for t in tensor_vals(by[m]["val"]))
                        )
                        for m in km
                    }
                )
            ),
        )

    groups_ = defaultdict(list)
    for c in final:
        groups_[loose_key(c)].append(c)
    merged = []
    for cs in groups_.values():
        cs.sort(key=lambda c: -c["instances"])
        base = dict(cs[0])
        if len(cs) > 1:
            inst = sum(c["instances"] for c in cs)
            base["merged_from"] = [c["cand_id"] for c in cs[1:]]
            base["kernels_per_instance"] = (
                sum(c["kernels_per_instance"] * c["instances"] for c in cs) / inst
            )
            base["bytes_per_instance"] = (
                sum(c["bytes_per_instance"] * c["instances"] for c in cs) / inst
            )
            for f in ("measured_ms_total", "ideal_ms_total", "gap_ms_total"):
                base[f] = sum(c[f] for c in cs)
            base["instances"] = inst
            base["measured_us_per_instance"] = base["measured_ms_total"] * 1e3 / inst
            base["ideal_us_per_instance"] = base["ideal_ms_total"] * 1e3 / inst
            base["cids"] = [x for c in cs for x in c["cids"]]
            base["kset_all"] = frozenset().union(*[c["kset_all"] for c in cs])
            base["bytes_unknown"] = any(c["bytes_unknown"] for c in cs)
            if all(c.get("gemm_us_per_instance") is not None for c in cs):
                base["gemm_us_per_instance"] = (
                    sum(c["gemm_us_per_instance"] * c["instances"] for c in cs) / inst
                )
                base["gemm_ideal_us_per_instance"] = (
                    sum(c["gemm_ideal_us_per_instance"] * c["instances"] for c in cs)
                    / inst
                )
        merged.append(base)
    final = sorted(merged, key=lambda c: -c["gap_ms_total"])

    ann = json.load(open(args.annotations)) if args.annotations else {}

    regions = []
    for c in final:
        r0 = df.loc[c["example_cid"]]
        mem = list(r0["members"])
        kmem = [m for m in mem if kus.get(m, 0) > 0]
        targets = [by[m]["target"] for m in kmem]
        tset = set(targets)
        phases = sorted({by[m]["phase"] for m in kmem}) or c["phases"]
        fq = sorted({by[m]["module_fqn"] for m in kmem})
        layer_ids = {by[m]["layer"] for m in kmem if by[m]["layer"] is not None}
        roots = {
            (by[m]["module_fqn"] or "").split(".")[0]
            for m in kmem
            if by[m]["layer"] is None
        }
        if len(layer_ids) > 1 or (layer_ids and roots):
            scope = ["cross-layer"]
        elif len(fq) > 1:
            scope = ["cross-module"]
        else:
            scope = ["intra-module"]
        if "recompute" in phases and "backward" in phases:
            scope.append("recompute-backward")
        if any(cat.get(m) in ("gemm", "attention", "fused_moe") for m in kmem):
            tier = "T3"
        elif tset & T2_OPS:
            tier = "T2"
        elif tset & REDUCTION_OPS:
            tier = "T1"
        else:
            tier = "T0"
        if c["kind"] == "T3-epi":
            fk = "GEMM epilogue"
        elif c["kind"] == "T3-pro":
            fk = "GEMM prologue"
        elif c.get("single_op"):
            fk = "single-op rewrite"
        elif any(t.startswith("bucketing.") for t in tset):
            fk = "dtype cast + FSDP bucket pack"
        elif tset & SCATTER_OPS:
            fk = "gather/scatter + pointwise"
        elif tset & {"aten.topk.default", "aten.sort.stable", "aten.sort.default"}:
            fk = "selection (top-k/sort) + pointwise"
        elif any("norm" in t for t in tset):
            fk = "normalization + pointwise"
        elif any("log_softmax" in t or "nll_loss" in t for t in tset):
            fk = "cross-entropy (log_softmax + nll)"
        elif tset & REDUCTION_OPS:
            fk = "reduction + pointwise"
        elif tset <= {
            "aten.cat.default",
            "aten.clone.default",
            "aten._to_copy.default",
            "aten.stack.default",
            "aten.constant_pad_nd.default",
        }:
            fk = "copy/concat + cast"
        else:
            fk = "pointwise chain"
        shp = Counter()
        big_numel = 0
        big_fp32 = False
        for m in kmem:
            for t in tensor_vals(by[m]["val"]):
                shp[f"{t['dtype'][6:]}{_abstract_shape(str(t['shape']))}"] += 1
                n_ = numel(t) or 0
                big_numel = max(big_numel, n_)
                big_fp32 |= t["dtype"] == "torch.float32" and n_ * 4 >= (1 << 30)
        main_shapes = ", ".join(s for s, _ in shp.most_common(4))
        srcs = Counter(short_src(by[m].get("stack_trace")) for m in kmem)
        srcs.pop("", None)
        by_file = defaultdict(list)
        for s_ in srcs:
            mm = re.match(r"(.+):(\d+) in .*", s_)
            if mm:
                by_file[mm.group(1)].append(int(mm.group(2)))
        src = "; ".join(
            f"{f}:{min(v)}-{max(v)}" if min(v) != max(v) else f"{f}:{v[0]}"
            for f, v in by_file.items()
        )
        incr = None
        if isinstance(r0.get("parts"), list) and r0["parts"]:
            tot = 0.0
            for cid in c["cids"]:
                rr = df.loc[cid]
                parts_ideal = sum(df.loc[p, "ideal_us"] for p in rr["parts"])
                if rr["kind"].startswith("T3"):
                    parts_ideal += rr["gemm_ideal_us"]
                tot += parts_ideal - rr["ideal_us"]
            incr = tot / 1e3
        gemm_note = ""
        if c["kind"].startswith("T3"):
            gus = c["gemm_us_per_instance"] or 0
            gid = c["gemm_ideal_us_per_instance"] or 0
            hidden = max(
                0.0,
                c["measured_us_per_instance"] - max(gus, c["ideal_us_per_instance"]),
            )
            gemm_note = (
                f"GEMM {c['gemm_example']}: {gus:.1f} us measured vs {gid:.1f} us flops-ideal "
                f"(GEMM-own gap {(gus - gid) * c['instances'] / 1e3:.1f} ms of this region's gap); glue part "
                f"{c['measured_us_per_instance'] - gus:.1f} us/instance. Fusion-attributable gap (glue time not "
                f"hidden behind the GEMM's measured time) = {hidden * c['instances'] / 1e3:.1f} ms."
            )
        blk = []
        if "torchtitan.deterministic_scatter_add.default" in tset:
            blk.append(
                "torchtitan.deterministic_scatter_add is an opaque custom op (sort-based deterministic scatter-add); "
                "a fused kernel must reproduce its accumulation order."
            )
        if tset & {"aten.index_put.default", "aten.index_put_.default"}:
            blk.append(
                "aten.index_put(_) under deterministic mode runs a sort-based path (memset/memcpy of the full "
                "destination); accumulate=True needs a deterministic segmented reduction. If the indices are a "
                "permutation (from sort/argsort), there are no duplicates and a gather/scatter is order-free."
            )
        if tset & {"aten.scatter_add.default", "aten.index_add.default"}:
            blk.append(
                "scatter_add/index_add accumulation order (atomics) must be fixed for bitwise results."
            )
        if tset & {"aten.topk.default", "aten.sort.stable", "aten.sort.default"}:
            blk.append(
                "top-k / sort selection: tie-breaking order must match; produces data-dependent indices."
            )
        if any("N" in s for s in shp):
            blk.append(
                "a dimension is data-dependent (unbacked SymInts): kernels need dynamic shapes."
            )
        if big_numel >= (1 << 31):
            blk.append(
                "tensors exceed 2^31 elements: eager TensorIterator splits each op into several launches "
                "(32-bit indexing)."
            )
        if big_fp32:
            blk.append(
                "fp32 intermediates of >= 1 GiB (type promotion) double the bytes of every pass in eager."
            )
        if "cross-layer" in scope:
            blk.append(
                "crosses a decoder-layer boundary: the value handed between layers is also read elsewhere "
                "(saved for backward, or the other residual branch), so the fused kernel must still write it; "
                "graph passes treat layers as units (per-layer FSDP buckets and reordering)."
            )
        if "recompute-backward" in scope:
            blk.append(
                "recompute copies are inserted by selective_activation_remat_pass right before their first "
                "backward consumer; fusing them into the backward kernel needs a remat-aware fusion step."
            )
        if tier == "T3":
            blk.append(
                "needs a GEMM with a custom prologue/epilogue (CUTLASS/Triton template or cuBLASLt epilogue)."
            )
        if any(t.startswith("bucketing.") for t in tset):
            blk.append(
                "the FSDP pack is a custom bucketing op; casts must fold into it or into the producer."
            )
        if not blk:
            blk.append("none structural: same phase, same fusion stage, no barrier op.")
        est = []
        if c["bytes_unknown"]:
            est.append(
                "some external tensor sizes were symbolic and unresolved; their bytes counted as 0"
            )
        if c["kind"].startswith("T3"):
            est.append("ideal uses flops/peak for the GEMM")
        key = hashlib.sha1(c["sig"].encode()).hexdigest()[:10]
        a = ann.get(key, {})
        how = "L1-single" if c.get("single_op") else c["kind"]
        kind_label = {
            "T3-epi": "T3 GEMM epilogue",
            "T3-pro": "T3 GEMM prologue",
            "L1": "Single op" if c.get("single_op") else "Intra-module",
            "XM-max": "Cross-module",
            "XM": "Cross-module via barrier op",
            "XL": "Cross-layer",
            "RB": "Recompute->backward",
        }[c["kind"]]
        ops_label = " + ".join(
            dict.fromkeys(
                short_op(t).replace(".Tensor", "").replace(".dim_IntList", "")
                for t in targets
            )
        )
        fqn_label = ", ".join(normalize_fqn(f).replace("layers.*.", "") for f in fq)
        title = (
            a.get("title")
            or f"{kind_label} ({'/'.join(phases)}): {fqn_label[:80]}: {ops_label[:90]}"
        )
        notes = []
        if incr is not None:
            notes.append(
                "incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) "
                f"= {max(0.0, incr):.1f} ms"
            )
        if gemm_note:
            notes.append(gemm_note)
        if c.get("duplicates"):
            notes.append(f"identical kernel set as candidate(s) {c['duplicates']}")
        if c.get("merged_from"):
            notes.append(
                f"merged signature variants (same ops and shapes): candidates {c['merged_from']}"
            )
        if a.get("notes"):
            notes.append(a["notes"])
        if est:
            notes.append("estimated: " + "; ".join(est))
        regions.append(
            dict(
                id=None,
                key=key,
                title=title,
                phases=phases,
                scope=scope,
                fusion_kind=a.get("fusion_kind", fk),
                location=dict(
                    module_fqn_pattern=", ".join(
                        sorted({normalize_fqn(f) for f in fq})
                    ),
                    graph_nodes_example=kmem[:12],
                    source=src or "(no torchtitan frame in stack_trace)",
                ),
                ops=sorted(dict.fromkeys(targets)),
                kernels_per_instance=round(c["kernels_per_instance"], 2),
                instances_per_step=c["instances"],
                shapes_dtypes=main_shapes,
                external_bytes_per_instance=int(c["bytes_per_instance"]),
                measured_us_per_instance=round(c["measured_us_per_instance"], 2),
                measured_ms_total=round(c["measured_ms_total"], 3),
                ideal_us_per_instance=round(c["ideal_us_per_instance"], 2),
                ideal_ms_total=round(c["ideal_ms_total"], 3),
                gap_ms_total=round(c["gap_ms_total"], 3),
                gap_pct_of_step=round(100 * c["gap_ms_total"] / step_ms, 3),
                bitwise_tier=tier,
                bitwise_notes=a.get(
                    "bitwise_notes",
                    TIER_NOTES[tier].format(
                        ops=", ".join(
                            sorted(
                                short_op(t)
                                for t in tset
                                & (T2_OPS if tier == "T2" else REDUCTION_OPS)
                            )
                        )
                    ),
                ),
                blockers=a.get("blockers", " ".join(blk)),
                evidence="estimated" if c["bytes_unknown"] else "measured",
                how_found=HOW_FOUND[how] + f" (candidate #{c['cand_id']}, {c['kind']})",
                overlaps_with=[],
                notes=" | ".join(notes),
                _kind=c["kind"],
                _kset=c["kset_all"],
                _incr=incr,
                _instances=[list(df.loc[cid, "members"]) for cid in c["cids"]],
            )
        )
    width = 2 if len(regions) < 100 else 3
    for i, r in enumerate(regions):
        r["id"] = f"G-{i + 1:0{width}d}"
    owner = defaultdict(list)
    for r in regions:
        for m in r["_kset"]:
            owner[m].append(r["id"])
    for r in regions:
        ov = set()
        for m in r["_kset"]:
            ov.update(owner[m])
        ov.discard(r["id"])
        r["overlaps_with"] = sorted(ov)

    # ---- no_opportunity: memory-bound node classes not in any region ----
    covered = set()
    for r in regions:
        covered |= set(r["_kset"])
    glue = nt[(nt["category"] == "memory_bound_glue") & (nt["kernel_us"] > 0)].copy()
    glue["fqn_pat"] = glue["module_fqn"].map(normalize_fqn)
    glue["covered"] = glue["name"].isin(covered)
    noop = []
    for (t, ph, fp), g in glue[~glue["covered"]].groupby(
        ["target", "phase", "fqn_pat"]
    ):
        ratio = g["kernel_us"].sum() / max(g["ideal_us"].sum(), 1e-9)
        if g["ideal_us"].sum() < 1.0 * len(g):
            reason = (
                f"tiny launch-bound kernel ({g['kernel_us'].sum() / len(g):.1f} us each, ~0 bytes) "
                "with no same-stage glue neighbour."
            )
        elif t.startswith("bucketing._pre_bucket_all_gather"):
            reason = "FSDP all-gather pack of parameter shards (one copy kernel feeding the all_gather)."
        elif t == "aten.split_with_sizes_copy.out":
            reason = (
                "FSDP all-gather unpack: one copy kernel after the collective; removable only by gathering into "
                "the final layout (comm-side change)."
            )
        else:
            reason = (
                f"single kernel ({ratio:.2f}x of its roofline) with no same-stage glue neighbour (neighbours are "
                "GEMM/attention/collective); listed as T3 prologue/epilogue where one applies."
            )
        noop.append(
            dict(
                what=f"{short_op(t)} [{ph}] {fp} (x{len(g)})",
                time_ms=round(g["kernel_us"].sum() / 1e3, 3),
                reason=reason,
            )
        )
    noop.append(
        dict(
            what="optimizer step + grad-norm clipping (eager, outside the traced graph)",
            time_ms=round(budget["by_category_ms"].get("optimizer", 0), 3),
            reason="runs outside the FX graph; invisible to the graph substrate.",
        )
    )
    noop.append(
        dict(
            what="other kernels outside the graph",
            time_ms=round(budget["by_category_ms"].get("other", 0), 3),
            reason="outside the traced graph; not attributable to FX nodes.",
        )
    )
    noop.sort(key=lambda x: -x["time_ms"])

    # ---- liftability: every instance checked on the graph extra passes see ----
    pre_path = os.path.join(R, "graph_dump", "pre_inductor_nodes.jsonl.gz")
    if not os.path.exists(pre_path):
        raise ValueError(
            f"{pre_path} is missing; capture with a fusion_finder pass pipeline"
        )
    dump_nodes = load_dump_graph(pre_path)
    dn = {n.name: n for n in dump_nodes}
    not_liftable = {}
    for r in regions:
        # A getitem of a tuple produced outside the instance is an input, so the
        # submodule receives the tensor it reads rather than the whole tuple. An
        # in-place write into outside storage that nothing in the instance reads
        # (e.g. a buffer accumulate) stays outside, after the call.
        r["_lift_instances"] = []
        for inst in r["_instances"]:
            lift = [
                m
                for m in inst
                if not (
                    by[m]["target"] == "_operator.getitem"
                    and by[m]["input_nodes"][0] not in inst
                )
            ]
            dset = {dn[m] for m in lift if m in dn}
            lift = [
                m
                for m in lift
                if m not in dn
                or not (
                    writes_outside(dn[m], dset)
                    and not any(u in dset for u in dn[m].users)
                )
            ]
            r["_lift_instances"].append(lift)
        reason = ""
        counts = Counter(m for inst in r["_lift_instances"] for m in inst)
        shared = [m for m, n in counts.items() if n > 1]
        if shared:
            reason = (
                f"instances share nodes ({shared[0]} ...): one group is paired with several neighbours, so the "
                "instances cannot all be lifted"
            )
        for inst in [] if reason else r["_lift_instances"]:
            if not inst:
                reason = "no member is left once outside reads and writes stay outside"
                break
            missing = [
                m for m in inst if m not in dn or dn[m].target != by[m]["target"]
            ]
            if missing:
                reason = f"nodes not in the pre_inductor graph with the same target: {missing[:3]}"
                break
            path = _reentry_path(inst, by, order)
            if path:
                ops = " -> ".join(
                    dict.fromkeys(short_op(by[x]["target"]) for x in path)
                )
                hard = [
                    x
                    for x in path
                    if cat.get(x) in ("gemm", "attention", "fused_moe", "fake_comm")
                ]
                if hard:
                    reason = (
                        f"not one kernel: {short_op(by[hard[0]]['target'])} ({hard[0]}) runs between region "
                        f"members (path out and back in: {ops})"
                    )
                else:
                    reason = (
                        f"incomplete: a path of glue ops leaves the region and re-enters it ({ops}, from "
                        f"{path[0]}); add those nodes to make it one kernel"
                    )
                r["blockers"] = f"NOT FUSABLE AS FOUND: {reason}. " + r["blockers"]
                break
            hazard = lift_hazard([dn[m] for m in inst])
            if hazard:
                reason = f"lifting is unsafe: {hazard}"
                break
        not_liftable[r["id"]] = reason

    # ---- one folder per region; a registered lift pass for each liftable one ----
    pkg = os.path.join(R, "fusion_regions")
    shutil.rmtree(pkg, ignore_errors=True)
    os.makedirs(pkg)
    folders = {
        r["id"]: f"{r['id'].replace('-', '').lower()}_{region_family(r)}"
        for r in regions
    }
    pass_names = {}
    num_sites = {}
    region_matchers = {}
    for r in regions:
        os.makedirs(os.path.join(pkg, folders[r["id"]]))
        if not_liftable[r["id"]]:
            continue
        matchers, sites = _region_matchers(r["_lift_instances"], dn, dump_nodes)
        if matchers is None:
            not_liftable[r["id"]] = f"no topology matcher: {sites}"
            continue
        mod = r["id"].replace("-", "").lower()
        pass_names[r["id"]] = f"lift_fusion_region_{mod}"
        num_sites[r["id"]] = len(sites)
        region_matchers[r["id"]] = [(a, fn) for a, _, fn in matchers]
        with open(os.path.join(pkg, folders[r["id"]], "lift_pass.py"), "w") as f:
            f.write(
                REGION_PASS_TEMPLATE.format(
                    id=r["id"],
                    title=r["title"],
                    num_instances=r["instances_per_step"],
                    gap=r["gap_ms_total"],
                    pct=r["gap_pct_of_step"],
                    tier=r["bitwise_tier"],
                    fusion_kind=r["fusion_kind"],
                    root=os.path.abspath(R),
                    num_sites=len(sites),
                    num_found=len(r["_instances"]),
                    pass_name=pass_names[r["id"]],
                    attr=f"fusion_{r['id'].replace('-', '')}",
                    lift_imports=_lift_imports(matchers),
                    matchers="\n\n".join(src for _, src, _ in matchers),
                    matcher_list="["
                    + ", ".join(
                        f"({a!r}, match_v{k})" for k, (a, _, _) in enumerate(matchers)
                    )
                    + "]",
                )
            )
    # Default selection: replay the passes in gap order on the dumped graph. A pass lifts
    # every safe match of its pattern, so a region joins only if safe matches remain after
    # the regions before it took theirs.
    taken = set()
    default_sel = []
    for r in regions:
        if r["id"] not in pass_names or r["gap_ms_total"] <= 0:
            continue
        sites = find_matches(dump_nodes, region_matchers[r["id"]], claimed=set(taken))
        safe = [site for site in sites if not lift_hazard(site)]
        if safe:
            default_sel.append(pass_names[r["id"]])
            taken |= {n for site in safe for n in site}
    for r in regions:
        entry = {k: v for k, v in r.items() if not k.startswith("_") and k != "key"}
        entry["lift"] = dict(
            pass_name=pass_names.get(r["id"]),
            num_sites=num_sites.get(r["id"]),
            in_default_selection=pass_names.get(r["id"]) in default_sel,
            not_liftable_reason=not_liftable[r["id"]],
        )
        with open(os.path.join(pkg, folders[r["id"]], "region.json"), "w") as f:
            json.dump(entry, f, indent=1)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write(
            PACKAGE_TEMPLATE.format(
                root=os.path.abspath(R),
                passes="{\n"
                + "".join(
                    f"    {rid!r}: ({folders[rid]!r}, {pn!r}),\n"
                    for rid, pn in pass_names.items()
                )
                + "}",
                default=json.dumps(default_sel, indent=4),
            )
        )
    repo = os.path.dirname(os.path.abspath(__file__))
    git = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    setup_path = os.path.join(R, "setup.json")
    setup_json = json.load(open(setup_path)) if os.path.exists(setup_path) else {}
    cmd_path = os.path.join(R, "command.txt")
    out = dict(
        substrate="graph",
        setup=dict(
            git_commit=git,
            command=open(cmd_path).read().strip() if os.path.exists(cmd_path) else "",
            config_deltas=setup_json,
            gpu=roof.get("device", ""),
            gpu_busy_during_timing=args.gpu_busy,
            hbm_gbps=cal["hbm_gbps"],
            bf16_gemm_tflops=cal["bf16_gemm_tflops"],
        ),
        step_budget=dict(
            step_gpu_busy_ms=round(budget["step_gpu_busy_ms"], 2),
            step_gpu_idle_ms=round(budget["step_gpu_idle_ms"], 2),
            attributed_pct=round(budget["attributed_pct_of_busy"], 3),
            by_category_ms={
                k: round(budget["by_category_ms"].get(k, 0.0), 2)
                for k in SETUP_CATEGORIES
            },
        ),
        regions=[
            {k: v for k, v in r.items() if not k.startswith("_") and k != "key"}
            for r in regions
        ],
        no_opportunity=noop,
    )
    json.dump(out, open(os.path.join(R, "regions.json"), "w"), indent=1)
    side = [
        dict(
            id=r["id"],
            key=r["key"],
            kind=r["_kind"],
            incr=r["_incr"],
            kset=sorted(r["_kset"]),
            folder=folders[r["id"]],
            lift_pass=pass_names.get(r["id"]),
            not_liftable_reason=not_liftable[r["id"]],
        )
        for r in regions
    ]
    json.dump(side, open(os.path.join(A, "regions_side.json"), "w"), indent=1)

    hdr = "| id | title | phases | scope | kind | inst | measured ms | ideal ms | gap ms | % step | tier | lift pass |"
    lines = [hdr, "|" + "---|" * 12]
    for r in regions:
        if r["id"] in pass_names:
            pn = pass_names[r["id"]]
            lift = f"`{pn}`, {num_sites[r['id']]} sites" + (
                " (default)" if pn in default_sel else ""
            )
        else:
            lift = "no: " + not_liftable[r["id"]].split(" (")[0].split(":")[0]
        lines.append(
            f"| {r['id']} | {r['title']} | {', '.join(r['phases'])} | {', '.join(r['scope'])} | {r['fusion_kind']} | "
            f"{r['instances_per_step']} | {r['measured_ms_total']:.1f} | {r['ideal_ms_total']:.1f} | "
            f"{r['gap_ms_total']:.1f} | {r['gap_pct_of_step']:.2f} | {r['bitwise_tier']} | {lift} |"
        )
    open(os.path.join(R, "regions_table.md"), "w").write("\n".join(lines) + "\n")

    cov = glue.groupby("covered")["kernel_us"].sum() / 1e3
    print(
        "regions",
        len(regions),
        "no_opportunity",
        len(noop),
        "lift passes",
        len(pass_names),
    )
    print("default selection", len(default_sel), "passes")
    print("glue ms covered/uncovered", cov.to_dict())
    print(Counter(t for r in regions for t in r["scope"]))
    for r in regions:
        print(
            f"{r['id']} [{r['key']}] {r['_kind']:6s} gap={r['gap_ms_total']:8.1f} x{r['instances_per_step']} "
            f"{r['bitwise_tier']} {r['scope']} {r['title'][:110]}"
        )


REGION_PASS_TEMPLATE = """\"\"\"Fusion region {id}: {title}

{num_instances} instances per step; gap {gap} ms ({pct}% of step); bitwise tier {tier}; {fusion_kind}.
Generated by fusion_region_finder/scripts/curate_regions.py from {root}. In that capture
the matchers below find {num_sites} site(s); the finder reported {num_found} instance(s).

Each matcher walks the region's topology from an anchor node and checks every op and
literal argument, except sizes (``S``) and scalar operands (``V``), which match any value,
and devices (``D``), which match by type. So the region is found wherever the same subgraph
appears: at other shapes and model configs, and on every rank. The node names and literals
in comments are from the capture, for reference only.

Add "{pass_name}" to config.compile.extra_passes after importing this module or the
fusion_regions package; the directory of lift_regions.py must be on PYTHONPATH. Every
instance becomes ``call_module {attr}`` (``{attr}_v<k>`` for structural variants) whose
body runs the region's original aten ops. To replace it with a kernel, assign a module
with the same signature to ``gm.{attr}`` in a pass listed after this one.
\"\"\"

from lift_regions import {lift_imports}

from torchtitan.experiments.graph_trainer.registry import register_graph_pass

REGION_ID = "{id}"


{matchers}

# (anchor op, matcher) for each structural variant.
MATCHERS = {matcher_list}


@register_graph_pass
def {pass_name}(gm, example_inputs=None):
    return lift_matches(gm, REGION_ID, MATCHERS)
"""

PACKAGE_TEMPLATE = """\"\"\"Fusion regions found in {root}, one folder per region.

Generated by fusion_region_finder/scripts/curate_regions.py. Importing this package
registers every region's lift pass (<folder>/lift_pass.py); enable regions by listing
their passes in config.compile.extra_passes. A region's pass lifts every safe match of
its pattern; after one region is lifted, an overlapping region's pass finds fewer
matches.
\"\"\"

import importlib

# Region id -> (folder, pass name), for every liftable region (largest gap first).
PASSES = {passes}

# Passes in gap order that each still find a safe site after the ones before them.
DEFAULT_SELECTION = {default}

for _folder, _ in PASSES.values():
    importlib.import_module(f"{{__name__}}.{{_folder}}.lift_pass")
"""


if __name__ == "__main__":
    main()
