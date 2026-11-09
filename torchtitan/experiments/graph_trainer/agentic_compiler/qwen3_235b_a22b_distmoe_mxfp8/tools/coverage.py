# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which finder regions does a plan already cover? Replays <plan> from <report dir> on <dump>,
then runs every region matcher of <capture dir>/fusion_regions and prints, per region, its gap
and the share of its matched nodes the plan claims.

    python coverage.py <report dir> <plan file> <capture dir> <dump> [top N]
"""

import importlib
import json
import os
import sys

from lift_regions import find_matches, lift_hazard, load_dump_graph


def load_pkg(root, name):
    sys.path.insert(0, os.path.abspath(root))
    for k in [
        k
        for k in sys.modules
        if k == "fusion_regions" or k.startswith("fusion_regions.")
    ]:
        del sys.modules[k]
    pkg = importlib.import_module("fusion_regions")
    sys.path.pop(0)
    return pkg


def main():
    root, plan_file, cap, dump = sys.argv[1:5]
    top = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    nodes = load_dump_graph(dump)
    pkg = load_pkg(root, "fusion_regions")
    folders = {p: f for f, p in pkg.PASSES.values()}
    mods = {
        p: importlib.import_module(f"fusion_regions.{folders[p]}.lift_pass")
        for p in folders
    }
    taken, per_pass = set(), {}
    for p in [x for x in open(plan_file).read().strip().split(",") if x]:
        sites = [
            s
            for s in find_matches(nodes, mods[p].MATCHERS, claimed=set(taken))
            if not lift_hazard(s)
        ]
        per_pass[p] = len(sites)
        taken |= {n for s in sites for n in s}
    print("plan sites:", per_pass)
    regions_dir = os.path.join(cap, "fusion_regions")
    # The capture's passes reuse the plan's pass names; this process only matches, never runs passes.
    from torchtitan.experiments.graph_trainer.registry import GRAPH_PASS_REGISTRY

    GRAPH_PASS_REGISTRY.clear()
    load_pkg(cap, "fusion_regions")
    rows = []
    for f in sorted(os.listdir(regions_dir)):
        rj = os.path.join(regions_dir, f, "region.json")
        if not os.path.exists(rj):
            continue
        r = json.load(open(rj))
        lp = os.path.join(regions_dir, f, "lift_pass.py")
        cov = "-"
        if os.path.exists(lp):
            m = importlib.import_module(f"fusion_regions.{f}.lift_pass")
            sites = find_matches(nodes, m.MATCHERS, claimed=set())
            members = {n for s in sites for n in s}
            cov = f"{len(members & taken)}/{len(members)}" if members else "0/0"
        rows.append((r["gap_ms_total"], r["id"], f, cov, r["title"][:90]))
    rows.sort(reverse=True)
    for gap, rid, f, cov, title in rows[:top]:
        print(f"{rid:5s} gap {gap:7.1f}  covered {cov:>11s}  {f:28s} {title}")


if __name__ == "__main__":
    main()
