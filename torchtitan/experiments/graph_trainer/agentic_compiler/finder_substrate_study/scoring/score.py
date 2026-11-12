# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Score the finder arms from the judge's verdicts (see ../PROTOCOL.md).

    python score.py <out.json> <model>=<judgement.json>,<key.json> [...]

Recall: sum over real opportunities of saving x the arm's best coverage, over the total real saving;
reported overall, without T3, without communication glue, and without both. Precision: share of an arm's
cards in at least one real opportunity, and covered judged saving over the arm's claimed gap. Unique:
real opportunities one arm covers at >= 0.5 while the other covers < 0.5.
"""

import json
import sys

ARMS = ("eager", "graph")


def _coverage(opp, key):
    cov = {a: 0.0 for a in ARMS}
    for c in opp.get("cards", []):
        arm = key[c["card"]]["arm"]
        cov[arm] = max(cov[arm], min(1.0, float(c.get("coverage", 0) or 0)))
    return cov


def score_model(judgement, key):
    opportunities = judgement["opportunities"]
    real = [
        o for o in opportunities if o.get("real") and float(o.get("saving_ms") or 0) > 0
    ]
    cov = {o["id"]: _coverage(o, key) for o in opportunities}

    def recall(pred):
        sel = [o for o in real if pred(o)]
        total = sum(float(o["saving_ms"]) for o in sel)
        return {
            "total_real_saving_ms": round(total, 2),
            "num_real": len(sel),
            **{
                a: round(
                    sum(float(o["saving_ms"]) * cov[o["id"]][a] for o in sel) / total, 4
                )
                if total
                else None
                for a in ARMS
            },
        }

    out = {
        "num_opportunities": len(opportunities),
        "num_real": len(real),
        "num_not_real": len(opportunities) - len(real),
        "recall": recall(lambda o: True),
        "recall_excl_t3": recall(lambda o: o.get("tier") != "T3"),
        "recall_excl_comm": recall(lambda o: o.get("category") != "communication-glue"),
        "recall_excl_t3_comm": recall(
            lambda o: o.get("tier") != "T3"
            and o.get("category") != "communication-glue"
        ),
    }
    real_ids = {o["id"] for o in real}
    for a in ARMS:
        cards = [c for c, v in key.items() if v["arm"] == a]
        in_real = {
            c["card"]
            for o in opportunities
            if o["id"] in real_ids
            for c in o.get("cards", [])
            if key[c["card"]]["arm"] == a and float(c.get("coverage", 0) or 0) > 0
        }
        claimed = sum(float(key[c]["claimed_gap_ms"] or 0) for c in cards)
        covered = sum(float(o["saving_ms"]) * cov[o["id"]][a] for o in real)
        out[a] = {
            "num_cards": len(cards),
            "cards_in_real_share": round(len(in_real) / len(cards), 4)
            if cards
            else None,
            "claimed_gap_ms": round(claimed, 2),
            "covered_real_saving_ms": round(covered, 2),
            "covered_over_claimed": round(covered / claimed, 4) if claimed else None,
        }
    uniq = {a: [] for a in ARMS}
    for o in real:
        c = cov[o["id"]]
        for a, b in (ARMS, ARMS[::-1]):
            if c[a] >= 0.5 and c[b] < 0.5:
                uniq[a].append(
                    {
                        "id": o["id"],
                        "title": o.get("title"),
                        "category": o.get("category"),
                        "tier": o.get("tier"),
                        "saving_ms": o["saving_ms"],
                        f"coverage_{b}": c[b],
                    }
                )
    out["unique"] = {
        a: {
            "count": len(v),
            "saving_ms": round(sum(float(x["saving_ms"]) for x in v), 2),
            "items": sorted(v, key=lambda x: -float(x["saving_ms"])),
        }
        for a, v in uniq.items()
    }
    by_cat = {}
    for o in real:
        d = by_cat.setdefault(
            o.get("category", "other"), {"saving_ms": 0.0, **{a: 0.0 for a in ARMS}}
        )
        d["saving_ms"] += float(o["saving_ms"])
        for a in ARMS:
            d[a] += float(o["saving_ms"]) * cov[o["id"]][a]
    out["by_category_ms"] = {
        k: {kk: round(vv, 2) for kk, vv in v.items()}
        for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]["saving_ms"])
    }
    return out


def decision(results):
    """Protocol rule on recall without T3: graph better if >= 10 points on >= 2 of 3 models and eager
    not >= 10 points better on the remaining one."""
    diffs = {
        m: (r["recall_excl_t3"]["graph"] or 0) - (r["recall_excl_t3"]["eager"] or 0)
        for m, r in results.items()
    }
    g = sum(d >= 0.10 for d in diffs.values())
    e = sum(d <= -0.10 for d in diffs.values())
    if g >= 2 and e == 0:
        call = "graph better for finding"
    elif e >= 2 and g == 0:
        call = "eager better for finding"
    else:
        call = "inconclusive"
    return {"recall_excl_t3_diff_graph_minus_eager": diffs, "call": call}


def main():
    out_path, specs = sys.argv[1], sys.argv[2:]
    results = {}
    for spec in specs:
        model, files = spec.split("=", 1)
        jpath, kpath = files.split(",")
        results[model] = score_model(json.load(open(jpath)), json.load(open(kpath)))
    final = {"models": results, "decision": decision(results)}
    json.dump(final, open(out_path, "w"), indent=1)
    for m, r in results.items():
        print(f"\n== {m}: {r['num_real']} real / {r['num_not_real']} not real")
        for k in (
            "recall",
            "recall_excl_t3",
            "recall_excl_comm",
            "recall_excl_t3_comm",
        ):
            v = r[k]
            print(
                f"  {k:22s} total {v['total_real_saving_ms']:8.1f} ms  eager {v['eager']}  graph {v['graph']}"
            )
        for a in ARMS:
            print(f"  {a:6s} {r[a]}")
        print(
            f"  unique: eager {r['unique']['eager']['count']} ({r['unique']['eager']['saving_ms']} ms), "
            f"graph {r['unique']['graph']['count']} ({r['unique']['graph']['saving_ms']} ms)"
        )
    print("\n", final["decision"])


if __name__ == "__main__":
    main()
