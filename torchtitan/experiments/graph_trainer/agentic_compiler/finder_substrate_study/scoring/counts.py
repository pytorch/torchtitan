# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Raw opportunity counts per arm (no saving estimates), overall and by phase.

    python counts.py <out.json>

An arm "found" a real opportunity if its best card coverage is >= 0.5 (the judge's coverage).
"Shared program" drops the opportunities that exist only in the eager program (FSDP2 backward
re-gather, persistent MXFP8 operand copy; see robustness.py). Qwen3's eager-only work is part of
two larger opportunities, so nothing is dropped there.
Phase comes from the judge's ops_summary prefix; opportunities spanning both phases are "mixed".
"""

import collections
import json
import sys

ROOT = "/home/bahuang/local/step_analysis"
KEYS = "/home/bahuang/local/torchtitan/outputs/finder_substrate/scoring/keys"
MODELS = ("llama3_8b", "qwen3_30b", "dsv3")
EAGER_ONLY = {
    "llama3_8b": {"O-28", "O-29", "O-31"},
    "qwen3_30b": set(),
    "dsv3": {"O-42", "O-44", "O-45"},
}


def phase(o):
    p = o["ops_summary"].split("/")[0].strip().lower()
    return p if p in ("forward", "backward") else "mixed"


def main():
    out = {}
    for m in MODELS:
        key = json.load(open(f"{KEYS}/{m}.json"))
        opportunities = json.load(open(f"{ROOT}/judge_{m}/out/judgement.json"))[
            "opportunities"
        ]
        res = {
            "regions": dict(collections.Counter(v["arm"] for v in key.values())),
            "opportunities": len(opportunities),
            "not_real": len([o for o in opportunities if not o["real"]]),
        }
        for label, excl in (("all", set()), ("shared_program", EAGER_ONLY[m])):
            tot = collections.Counter()
            by_phase = collections.defaultdict(collections.Counter)
            for o in opportunities:
                if not o["real"] or o["id"] in excl:
                    continue
                cov = {
                    a: max(
                        [
                            float(c.get("coverage") or 0)
                            for c in o["cards"]
                            if key[c["card"]]["arm"] == a
                        ]
                        or [0]
                    )
                    for a in ("eager", "graph")
                }
                e, g = cov["eager"] >= 0.5, cov["graph"] >= 0.5
                k = (
                    "both"
                    if e and g
                    else "eager_only"
                    if e
                    else "graph_only"
                    if g
                    else "neither"
                )
                tot[k] += 1
                by_phase[phase(o)][k] += 1
            res[label] = {
                **dict(tot),
                "eager_found": tot["both"] + tot["eager_only"],
                "graph_found": tot["both"] + tot["graph_only"],
                "by_phase": {p: dict(c) for p, c in by_phase.items()},
            }
        out[m] = res
        print(m, json.dumps(res))
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
