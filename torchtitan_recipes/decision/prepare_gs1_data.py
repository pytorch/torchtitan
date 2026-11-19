# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Select and flatten native GS1 distribution records for the decision recipe."""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _contains_media(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("type") in {"image", "video"}:
            return True
        return any(_contains_media(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_media(item) for item in value)
    return False


def flatten_record(row: dict[str, Any]) -> list[dict[str, Any]]:
    request = row["request"]
    targets = row["targets"]
    questions = request["questions"]
    if set(questions) != set(targets):
        raise ValueError("GS1 question IDs must match target IDs")

    rows = []
    if _contains_media(request["state"]):
        return rows
    for question_id, question in questions.items():
        kind = question["type"]
        criteria = question.get("criteria") or {}
        if kind == "choice":
            candidate_ids = list(criteria)
            options = [
                _text(criteria[key] if criteria[key] is not None else key)
                for key in candidate_ids
            ]
        elif kind == "noul":
            candidate_ids = ["false", "true"]
            options = [_text(criteria.get(key) or key) for key in candidate_ids]
        elif kind == "score":
            candidate_ids = [str(index) for index in range(len(criteria))]
            options = [_text(value) for value in criteria]
        else:
            raise ValueError(f"Unsupported GS1 question type: {kind!r}")

        target = targets[question_id]
        target_ids = target["candidate_ids"]
        distribution = target["distribution"]
        if len(candidate_ids) < 2 or len(candidate_ids) > 26:
            continue
        if len(target_ids) != len(candidate_ids) or set(target_ids) != set(
            candidate_ids
        ):
            raise ValueError(f"Target candidates do not match question {question_id!r}")
        target_by_id = dict(zip(target_ids, distribution, strict=True))
        target_probs = [float(target_by_id[key]) for key in candidate_ids]
        if any(
            not math.isfinite(value) or value < 0 for value in target_probs
        ) or not math.isclose(sum(target_probs), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                f"Invalid target distribution for question {question_id!r}"
            )
        rows.append(
            {
                "state": _text(request["state"]),
                "kind": kind,
                "instructions": _text(question.get("instructions") or question_id),
                "candidate_ids": candidate_ids,
                "options": options,
                "target_probs": target_probs,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("./assets/hf/gs1-distribution/train.jsonl")
    )
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive")

    rng = random.Random(args.seed)
    rows = []
    num_seen = 0
    with args.input.open() as source:
        for line in source:
            for row in flatten_record(json.loads(line)):
                num_seen += 1
                if len(rows) < args.num_samples:
                    rows.append(row)
                else:
                    index = rng.randrange(num_seen)
                    if index < args.num_samples:
                        rows[index] = row
    if num_seen < args.num_samples:
        raise ValueError(
            f"Found {num_seen} eligible questions, requested {args.num_samples}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        output.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"Wrote {len(rows)} questions to {args.out}")


if __name__ == "__main__":
    main()
