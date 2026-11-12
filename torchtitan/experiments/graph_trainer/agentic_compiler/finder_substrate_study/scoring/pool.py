# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pool both arms' regions for one model into blind cards for the judge.

    python pool.py <model> <judge sandbox> <key out.json>

Reads <ROOT>/<model>_{eager,graph}/out/regions.json. Writes <judge sandbox>/cards.json (random card IDs,
random order, structured fields only, substrate vocabulary scrubbed) and the card -> (arm, region id) key,
which stays outside the judge's sandbox. Also writes <judge sandbox>/overlaps.json: card pairs whose
footprints share (phase, module, op) keys, as hints for grouping.
"""

import json
import os
import random
import re
import sys

ROOT = "/home/bahuang/local/step_analysis"
REPO = f"{ROOT}/repo/"

# Words and names that would tell the judge which artifacts a finder worked from.
_SCRUB = [
    (re.compile(r"aten::(\w+)"), r"\1"),
    (re.compile(r"aten\.(\w+)(\.\w+)?"), r"\1"),
    (re.compile(r"_c10d_functional[.:]+(\w+)"), r"\1"),
    (
        re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*_\d+\b"),
        "<value>",
    ),  # FX node names
    (re.compile(r"\bSimpleFSDP\b|\bFSDP2\b|\bfully_shard\b"), "FSDP"),
    (
        re.compile(r"\bGraphTrainer\b|\bjoint graph\b|\bFX graph\b|\bFX\b", re.I),
        "<program>",
    ),
    (re.compile(r"\b(graph )?nodes?\b", re.I), "op(s)"),
    (re.compile(r"\bgraph\b", re.I), "program"),
    (re.compile(r"\beager\b", re.I), "<program>"),
    (
        re.compile(
            r"\bnn_module_stack\b|\bstack_trace\b|\bseq_nr\b|\bseq_id\b|\brf_id\b"
        ),
        "<metadata>",
    ),
    (
        re.compile(
            r"\b_RecordFunctionFast\b|\brecord_function\b|\bcustom_codegen\w*\b"
        ),
        "<label>",
    ),
    (
        re.compile(
            r"\bexecution trace\b|\bKineto\b|\bprofiler trace\b|\btrace\b", re.I
        ),
        "<profile>",
    ),
    (re.compile(r"\bhooks?\b", re.I), "<callback>"),
    (re.compile(r"\bcodegen\b|\bpass(es)?\b(?= \w+_pass)|\w+_pass\b"), "<transform>"),
    (re.compile(re.escape(REPO)), ""),
]


def scrub(text):
    if not isinstance(text, str):
        return text
    for pat, rep in _SCRUB:
        text = pat.sub(rep, text)
    return text


def norm_module(m):
    m = (m or "").strip()
    m = re.sub(r"\[['\"]?(\d+)['\"]?\]", r".\1", m)  # layers['3'] -> layers.3
    m = re.sub(r"^L\['self'\]\.?|^self\.?", "", m)
    m = re.sub(r"\.\d+(?=\.|$)", ".*", m)
    return m


def _is_overload(s):
    return (
        s in ("default", "int", "dim", "out", "self", "names", "using_ints")
        or s[:1].isupper()
    )


def norm_op(o):
    """aten::add / aten.add.Tensor / add.Tensor -> add; ns::op / ns.op.default -> ns.op."""
    parts = o.strip().replace("::", ".").split(".")
    if parts[0] == "aten":
        parts = parts[1:]
        return parts[0] if parts else "aten"
    if len(parts) >= 2 and _is_overload(parts[-1]):
        parts = parts[:-1]
    return ".".join(parts)


def norm_footprint(fp):
    out = []
    for e in fp or []:
        if not isinstance(e, dict):
            continue
        out.append(
            {
                "phase": e.get("phase"),
                "module": norm_module(e.get("module", "")),
                "layers": e.get("layers"),
                "ops": sorted(
                    {norm_op(o) for o in e.get("ops", []) if isinstance(o, str)}
                ),
            }
        )
    return out


CARD_FIELDS = [
    "title",
    "description",
    "phases",
    "scope",
    "shapes_dtypes",
    "kernels_per_instance",
    "instances_per_step",
    "external_bytes_per_instance",
    "measured_us_per_instance",
    "measured_ms_total",
    "ideal_us_per_instance",
    "ideal_ms_total",
    "gap_ms_total",
    "bitwise_tier",
    "bitwise_notes",
    "dataflow",
    "blockers",
    "source",
]


def main():
    model, judge_box, key_out = sys.argv[1:4]
    cards, key = [], {}
    rng = random.Random(f"{model}-pool")
    ids = rng.sample(range(1000, 10000), 2000)
    n = 0
    for arm in ("eager", "graph"):
        path = f"{ROOT}/{model}_{arm}/out/regions.json"
        regions = json.load(open(path))["regions"]
        for r in regions:
            cid = f"C-{ids[n]}"
            n += 1
            card = {"card": cid}
            for f in CARD_FIELDS:
                if f in r:
                    card[f] = scrub(r[f]) if isinstance(r[f], str) else r[f]
            card["footprint"] = norm_footprint(r.get("footprint"))
            cards.append(card)
            key[cid] = {
                "arm": arm,
                "region": r.get("id"),
                "claimed_gap_ms": r.get("gap_ms_total"),
            }
    rng.shuffle(cards)
    # Overlap hints: shared (phase, module, op) keys.
    keys = {
        c["card"]: {
            (e["phase"], e["module"], o) for e in c["footprint"] for o in e["ops"]
        }
        for c in cards
    }
    overlaps = []
    cl = [c["card"] for c in cards]
    for i, a in enumerate(cl):
        for b in cl[i + 1 :]:
            shared = keys[a] & keys[b]
            if shared:
                overlaps.append(
                    {
                        "cards": [a, b],
                        "shared_keys": sorted("/".join(map(str, s)) for s in shared)[
                            :12
                        ],
                    }
                )
    os.makedirs(judge_box, exist_ok=True)
    json.dump(cards, open(os.path.join(judge_box, "cards.json"), "w"), indent=1)
    json.dump(overlaps, open(os.path.join(judge_box, "overlaps.json"), "w"), indent=1)
    json.dump(key, open(key_out, "w"), indent=1)
    print(f"{model}: {len(cards)} cards, {len(overlaps)} overlapping pairs")


if __name__ == "__main__":
    main()
