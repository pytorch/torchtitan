# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check regions.json against the schema and the done criteria.

Usage: ``python validate_regions.py <capture dir>``. Prints VALID or the failures.
"""

import argparse
import json
import os
import sys

import pandas as pd

from curate_regions import short_op
from graph_lib import normalize_fqn

NUM = (int, float)
SETUP_KEYS = {
    "git_commit": str,
    "command": str,
    "config_deltas": dict,
    "gpu": str,
    "gpu_busy_during_timing": bool,
    "hbm_gbps": NUM,
    "bf16_gemm_tflops": NUM,
}
REGION_KEYS = {
    "id": str,
    "title": str,
    "phases": list,
    "scope": list,
    "fusion_kind": str,
    "location": dict,
    "ops": list,
    "kernels_per_instance": NUM,
    "instances_per_step": NUM,
    "shapes_dtypes": str,
    "external_bytes_per_instance": NUM,
    "measured_us_per_instance": NUM,
    "measured_ms_total": NUM,
    "ideal_us_per_instance": NUM,
    "ideal_ms_total": NUM,
    "gap_ms_total": NUM,
    "gap_pct_of_step": NUM,
    "bitwise_tier": str,
    "bitwise_notes": str,
    "blockers": str,
    "evidence": str,
    "how_found": str,
    "overlaps_with": list,
    "notes": str,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    R = ap.parse_args().root
    d = json.load(open(os.path.join(R, "regions.json")))
    failures = []

    def need(cond, msg):
        if not cond:
            failures.append(msg)

    need(d.get("substrate") == "graph", "substrate")
    for k, t in SETUP_KEYS.items():
        need(isinstance(d["setup"].get(k), t), f"setup.{k}")
    sb = d["step_budget"]
    for k in ("step_gpu_busy_ms", "step_gpu_idle_ms", "attributed_pct"):
        need(isinstance(sb.get(k), NUM), f"step_budget.{k}")
    need(sb["attributed_pct"] >= 95, f"attribution {sb['attributed_pct']:.2f}% < 95%")
    ids = set()
    for r in d["regions"]:
        rid = r.get("id")
        need(
            set(r) == set(REGION_KEYS),
            f"{rid} keys differ: {set(r) ^ set(REGION_KEYS)}",
        )
        for k, t in REGION_KEYS.items():
            need(isinstance(r.get(k), t), f"{rid}.{k} type")
        need(
            set(r["location"])
            == {"module_fqn_pattern", "graph_nodes_example", "source"},
            f"{rid} location",
        )
        need(set(r["phases"]) <= {"forward", "recompute", "backward"}, f"{rid} phases")
        need(
            bool(r["scope"])
            and set(r["scope"])
            <= {"intra-module", "cross-module", "cross-layer", "recompute-backward"},
            f"{rid} scope",
        )
        need(r["bitwise_tier"] in {"T0", "T1", "T2", "T3"}, f"{rid} tier")
        need(r["evidence"] in {"measured", "estimated"}, f"{rid} evidence")
        need(
            abs(r["gap_ms_total"] - (r["measured_ms_total"] - r["ideal_ms_total"]))
            < 1e-2,
            f"{rid} gap arithmetic",
        )
        ids.add(rid)
    for r in d["regions"]:
        need(set(r["overlaps_with"]) <= ids, f"{r['id']} overlaps unknown ids")
    gaps = [r["gap_ms_total"] for r in d["regions"]]
    need(gaps == sorted(gaps, reverse=True), "regions not ranked by gap")
    for x in d["no_opportunity"]:
        need(set(x) == {"what", "time_ms", "reason"}, "no_opportunity keys")

    # Every memory-bound node with kernel time is in a region or in a no_opportunity class.
    side = json.load(open(os.path.join(R, "analysis", "regions_side.json")))
    covered = {m for s in side for m in s["kset"]}
    nt = pd.read_pickle(os.path.join(R, "analysis", "node_table.pkl"))
    glue = nt[(nt.category == "memory_bound_glue") & (nt.kernel_us > 0)]
    listed = {x["what"].rsplit(" (x", 1)[0] for x in d["no_opportunity"]}
    missing = sorted(
        {
            f"{short_op(g.target)} [{g.phase}] {normalize_fqn(g.module_fqn)}"
            for g in glue.itertuples()
            if g.name not in covered
        }
        - listed
    )
    need(not missing, f"uncovered glue classes: {missing[:10]}")

    # One folder per region with its region.json; a lift pass exactly for the liftable ones.
    pkg = os.path.join(R, "fusion_regions")
    folders = {s["id"]: s["folder"] for s in side}
    have = {d for d in os.listdir(pkg) if os.path.isdir(os.path.join(pkg, d))} - {
        "__pycache__"
    }
    need(
        have == set(folders.values()),
        f"region folders differ from regions: {sorted(have ^ set(folders.values()))[:5]}",
    )
    for s in side:
        d = os.path.join(pkg, s["folder"])
        need(
            os.path.exists(os.path.join(d, "region.json")),
            f"{s['folder']}: no region.json",
        )
        has_pass = os.path.exists(os.path.join(d, "lift_pass.py"))
        need(
            has_pass == bool(s["lift_pass"]),
            f"{s['folder']}: lift_pass.py {'present' if has_pass else 'missing'} but the region is "
            f"{'liftable' if s['lift_pass'] else 'not liftable'}",
        )
    print(
        f"regions {len(ids)}, glue nodes {len(glue)}, covered {int(glue.name.isin(covered).sum())}"
    )
    for f in failures:
        print("FAIL:", f)
    print("VALID" if not failures else "INVALID")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
