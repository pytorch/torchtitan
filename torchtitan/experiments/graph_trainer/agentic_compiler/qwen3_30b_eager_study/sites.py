# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Print one site per region in full: members (op, module, phase, source line) and boundary edges
with the producer's / consumer's layer relative to the site's layer. Same arguments as classify.py
(prints instead of writing JSON), plus an optional comma-separated list of region ids. The site
printed is the one in layer 5."""

import gzip
import importlib
import json
import os
import re
import sys

from classify import fqn_of, frames, kind, short
from lift_regions import find_matches, lift_hazard, load_dump_graph

root, plan_file, dump = sys.argv[1:4]
want = sys.argv[4].split(",") if len(sys.argv) > 4 else None
recs = {r["name"]: r for r in (json.loads(l) for l in gzip.open(dump, "rt"))}
fwd_by_seq = {}
for r in recs.values():
    if not r["autograd_backward"] and fqn_of(r) and r["seq_nr"] is not None:
        fwd_by_seq.setdefault(r["seq_nr"], fqn_of(r))


def module(n):
    r = recs[n]
    return fqn_of(r) or (
        fwd_by_seq.get(r["seq_nr"]) if r["autograd_backward"] else None
    )


def layer(n):
    m = module(n)
    mm = re.match(r"layers\.(\d+)", m or "")
    return int(mm.group(1)) if mm else None


def src(n):
    fr = [x for x in frames(recs[n]["stack_trace"]) if "fx/traceback.py" not in x[0]]
    tt = [x for x in fr if "torchtitan/torchtitan/" in x[0]]
    return f"{short(tt[-1][0])}:{tt[-1][1]}" if tt else "-"


sys.path.insert(0, os.path.abspath(root))
regions = importlib.import_module("fusion_regions")
folder = {p: (rid, f) for rid, (f, p) in regions.PASSES.items()}
nodes = load_dump_graph(dump)
taken = set()
for p in open(plan_file).read().strip().split(","):
    rid, f = folder[p]
    mod = importlib.import_module(f"fusion_regions.{f}.lift_pass")
    sites = [
        s
        for s in find_matches(nodes, mod.MATCHERS, claimed=set(taken))
        if not lift_hazard(s)
    ]
    taken |= {n for s in sites for n in s}
    if want and rid not in want:
        continue
    site = next((s for s in sites if 5 in {layer(n.name) for n in s}), sites[0])
    names = {n.name for n in site}
    L = next((layer(n.name) for n in site if layer(n.name) is not None), None)
    rel = (
        lambda n: "-"
        if layer(n) is None or L is None
        else ("same" if layer(n) == L else f"{layer(n) - L:+d}")
    )
    print(f"\n== {rid} ({p}), site in layer {L}: {len(site)} nodes")
    for member in sorted(site, key=lambda x: x.idx):
        r = recs[member.name]
        if r["op"] != "call_function" or r["target"] == "_operator.getitem":
            continue
        phase = "B" if r["autograd_backward"] else "F"
        target = r["target"].replace("aten.", "")
        print(
            f"  {phase} {target:42s} {str(module(member.name)):45s} "
            f"seq={r['seq_nr']} {src(member.name)}"
        )
    ins, outs = set(), set()
    for n in site:
        r = recs[n.name]
        for ref in re.findall(
            r"'node': '([^']+)'",
            json.dumps(r["args"] + list(r["kwargs"].values())).replace('"', "'"),
        ):
            if ref not in names:
                ins.add(ref)
        outs |= {u for u in r["users"] if u not in names}
    for tag, group in (("in ", ins), ("out", outs)):
        for e in sorted(group, key=lambda x: recs[x]["idx"]):
            r = recs[e]
            target = (
                r["target"].replace("aten.", "")
                if r["op"] == "call_function"
                else r["op"]
            )
            phase = "B" if r["autograd_backward"] else "F"
            print(
                f"  {tag} {kind(r):12s} {target:36s} {str(module(e)):42s} "
                f"layer={rel(e)} {phase}"
            )
