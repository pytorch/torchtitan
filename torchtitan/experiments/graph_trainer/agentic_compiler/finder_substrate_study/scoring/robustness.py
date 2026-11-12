# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Robustness checks on top of score.py (post hoc; see RESULTS.md for the evidence behind each adjustment).

    python robustness.py <out.json>

1. Verified eager-only program work removed: opportunities (or the stated part of one) that exist only in the
   eager program. Evidence: the graph programs have no backward all-gather (all FSDP all-gathers are forward
   nodes) and no `_build_operands` copy into persistent MXFP8 operand storage (graph quantizes inside the
   linear's forward).
2. Category-based, assigned blind by the judge: recall excluding `communication-glue` and `weight-prep`,
   where the eager (FSDP2) and graph (GraphTrainer FSDP) programs differ by construction.
"""

import json
import sys

ROOT = "/home/bahuang/local/step_analysis"
KEYS = "/home/bahuang/local/torchtitan/outputs/finder_substrate/scoring/keys"
ARMS = ("eager", "graph")

# ms of saving that exists only in the eager program, per opportunity id (from the judge's saving_basis).
EAGER_ONLY = {
    "llama3_8b": {
        "O-31": "all",
        "O-28": "all",
        "O-29": "all",
    },  # backward re-gather copy-out/in, reuse
    "qwen3_30b": {
        "O-32": 14.415 / 2,  # 96 unshards = 48 forward + 48 backward re-gather
        "O-33": (66.29 - 10.41) * 48 / 1000,  # the backward term of the basis
    },
    "dsv3": {
        "O-44": "all",
        "O-42": "all",
        "O-45": "all",
    },  # backward re-unshard; persistent copy_
}


def recall(opportunities, key, saving_of):
    tot = {a: 0.0 for a in ARMS}
    total = 0.0
    for o in opportunities:
        s = saving_of(o)
        if s <= 0:
            continue
        total += s
        for a in ARMS:
            cov = max(
                [
                    min(1.0, float(c.get("coverage", 0) or 0))
                    for c in o["cards"]
                    if key[c["card"]]["arm"] == a
                ]
                or [0.0]
            )
            tot[a] += s * cov
    return {"total_ms": round(total, 2), **{a: round(tot[a] / total, 4) for a in ARMS}}


def main():
    out = {}
    for m, adj in EAGER_ONLY.items():
        key = json.load(open(f"{KEYS}/{m}.json"))
        opportunities = [
            o
            for o in json.load(open(f"{ROOT}/judge_{m}/out/judgement.json"))[
                "opportunities"
            ]
            if o.get("real") and float(o.get("saving_ms") or 0) > 0
        ]

        def adjusted(o):
            s = float(o["saving_ms"])
            a = adj.get(o["id"])
            return 0.0 if a == "all" else s - (a or 0.0)

        def adjusted_no_t3(o):
            return 0.0 if o.get("tier") == "T3" else adjusted(o)

        removed = sum(float(o["saving_ms"]) - adjusted(o) for o in opportunities)
        out[m] = {
            "eager_only_removed_ms": round(removed, 2),
            "recall_shared_program": recall(opportunities, key, adjusted),
            "recall_shared_program_excl_t3": recall(opportunities, key, adjusted_no_t3),
            "recall_excl_comm_and_weight_prep": recall(
                opportunities,
                key,
                lambda o: 0.0
                if o.get("category") in ("communication-glue", "weight-prep")
                else float(o["saving_ms"]),
            ),
            "recall_excl_comm_weight_prep_t3": recall(
                opportunities,
                key,
                lambda o: 0.0
                if o.get("category") in ("communication-glue", "weight-prep")
                or o.get("tier") == "T3"
                else float(o["saving_ms"]),
            ),
        }
        print(m, json.dumps(out[m]))
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
