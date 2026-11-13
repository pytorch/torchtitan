# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize caching / expandable / planned runs for each memory policy.

Steady state = steps >= 2 for unplanned runs, and steps with plan hits for
planned runs. Efficiency = peak allocated / peak reserved (the paper's E).
"""

import json
import os
import statistics
import sys

G = 1024**3
ROOT = "/data/users/bahuang/local/torchtitan_2/outputs/memplan"


def _recs(run):
    path = os.path.join(ROOT, run, "memstat.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def _steady(run, planned=False):
    recs = _recs(run)
    if planned:
        recs = [r for r in recs if r.get("hits", 0) > 0]
    else:
        recs = [r for r in recs if r["step"] >= 2]
    return recs


def row(policy, runs):
    out = {"policy": policy}
    c = _steady(runs["caching"])
    if c:
        last = c[-1]
        out["live_peak"] = last["peak_allocated"]
        out["caching_reserved"] = max(r["peak_reserved"] for r in c)
        out["caching_step_s"] = statistics.median(r["time_s"] for r in c[1:] or c)
    e = _steady(runs.get("expandable", ""))
    if e:
        out["expandable_reserved"] = max(r["peak_reserved"] for r in e)
    p = _steady(runs.get("planned", ""), planned=True)
    if p:
        out["planned_reserved"] = max(r["peak_reserved"] for r in p)
        out["planned_live_peak"] = p[-1]["peak_allocated"]
        out["planned_arena"] = p[-1]["arena_size"]
        out["planned_pool_reserved"] = max(r["pool_reserved_high"] for r in p)
        out["planned_pool_used"] = max(r["pool_used_high"] for r in p)
        out["planned_hits"] = p[-1]["hits"]
        out["planned_misses"] = sum(r["misses"] for r in p)
        out["planned_overlaps"] = sum(r["overlaps"] for r in p)
        out["planned_window_fallbacks"] = p[-1]["window_fallbacks"]
        out["planned_step_s"] = statistics.median(r["time_s"] for r in p)
        out["planned_steps"] = len(p)
    s = runs.get("summary")
    if s and os.path.exists(os.path.join(ROOT, s)):
        summ = json.load(open(os.path.join(ROOT, s)))
        out["transient_lower_bound"] = summ["lower_bound_bytes"]
        out["arena_planned"] = summ["arena_bytes"]
        out["num_plannable"] = summ["num_plannable"]
        out["num_wrapping"] = summ["num_wrapping"]
        out["plan_s"] = summ["plan_s"]
    return out


def fmt(rows):
    lines = [
        "| policy | live peak | caching reserved (E) | expandable reserved (E) "
        "| planned reserved (E) | saving vs caching | saving vs expandable |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lp = r.get("live_peak")

        def cell(key):
            v = r.get(key)
            if v is None or lp is None:
                return "-"
            return f"{v / G:.2f} GiB ({100 * lp / v:.1f}%)"

        def save(key):
            if r.get("planned_reserved") is None or r.get(key) is None:
                return "-"
            d = r[key] - r["planned_reserved"]
            return f"{d / G:+.2f} GiB ({100 * d / r[key]:+.1f}%)"

        lines.append(
            f"| {r['policy']} | {lp / G:.2f} GiB | {cell('caching_reserved')} | "
            f"{cell('expandable_reserved')} | {cell('planned_reserved')} | "
            f"{save('caching_reserved')} | {save('expandable_reserved')} |"
        )
    return "\n".join(lines)


POLICIES = {
    "none": {
        "caching": "none_caching",
        "expandable": "none_expandable",
        "planned": "none_planned",
        "summary": "none_trace/summary_stalloc.json",
    },
    "default (SAC)": {
        "caching": "base_caching",
        "expandable": "base_expandable",
        "planned": "sac_planned",
        "summary": "planned_trace2/summary_stalloc.json",
    },
    "full": {
        "caching": "full_caching",
        "expandable": "full_expandable",
        "planned": "full_planned",
        "summary": "full_trace/summary_stalloc.json",
    },
}

if __name__ == "__main__":
    rows = [row(k, v) for k, v in POLICIES.items() if _recs(v["caching"])]
    print(fmt(rows))
    json.dump(rows, open(os.path.join(ROOT, "summary.json"), "w"), indent=1)
    if len(sys.argv) > 1:
        for r in rows:
            print(
                json.dumps(
                    {
                        k: (round(v / G, 3) if isinstance(v, int) and v > 1e6 else v)
                        for k, v in r.items()
                    }
                )
            )
