# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flatten BEV decision rows into one letter-slot prompt per question.

BEV (``avbiswas/bev-decision-150K``) stores a ``state`` and a JSON object of
typed questions (``choice``, ``noul``, ``score``) with labels. Each question
becomes a prompt that lists its options as letters and ends in ``Answer:``;
the target is the gold letter with a leading space (one token). ``choice``
options and ``noul``'s Yes/No order are shuffled with a seeded RNG so the model
cannot learn letter-position priors; ``score`` levels stay ordered.

States over ``MAX_STATE_CHARS`` and questions with fewer than 2 or more than 26
options are dropped. Writes ``train.jsonl``, ``calib.jsonl`` (every question
from the first 250 test states, for the temperature fit), and ``eval.jsonl``
(1,500 questions from the remaining test states).

Usage: python -m torchtitan_recipes.decision.prepare_data [--out DIR]
"""

import argparse
import json
import math
import random
import string
from pathlib import Path
from typing import Any

LETTERS = string.ascii_uppercase
MAX_STATE_CHARS = 4000
MIN_CANDIDATES = 2


def render(state: str, question: dict, rng: random.Random) -> dict:
    kind = question["type"]
    if kind == "noul":
        options = [("true", "Yes"), ("false", "No")]
        label = "true" if question["label"] else "false"
    elif kind == "score":
        options = [(str(i), d) for i, d in enumerate(question["criteria"])]
        label = str(question["label"])
    else:
        options = [(k, v or k) for k, v in question["criteria"].items()]
        label = question["label"]
    if kind != "score":
        rng.shuffle(options)
    keys = [key for key, _ in options]
    gold = keys.index(label)
    return render_options(
        state,
        kind,
        question["instructions"],
        keys,
        [d for _, d in options],
        [float(i == gold) for i in range(len(keys))],
    )


def render_options(state, kind, instructions, keys, descriptions, target_probs):
    options = "\n".join(f"{LETTERS[i]}) {d}" for i, d in enumerate(descriptions))
    title = {"noul": "yes/no", "choice": "choice", "score": "ordered scale"}[kind]
    return {
        "prompt": f"Context:\n{state}\n\nQuestion ({title}): "
        f"{instructions}\nOptions:\n{options}\nAnswer:",
        "keys": keys,
        "target_probs": target_probs,
        "type": kind,
        "state": state,
        "instructions": instructions,
    }


def valid_option_count(count: int) -> bool:
    return MIN_CANDIDATES <= count <= len(LETTERS)


def flatten(split: str) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("avbiswas/bev-decision-150K", split=split)
    option_rng = random.Random(f"options:{split}")
    rows = []
    for state, questions in zip(dataset["state"], dataset["questions_json"]):
        if len(state) > MAX_STATE_CHARS:
            continue
        for question in json.loads(questions).values():
            num_options = 2 if question["type"] == "noul" else len(question["criteria"])
            if valid_option_count(num_options):
                rows.append(render(state, question, option_rng))
    random.Random(0).shuffle(rows)
    return rows


def write(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    print(f"{path}: {len(rows)} questions")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _contains_media(value: Any) -> bool:
    if isinstance(value, dict):
        kind = value.get("type")
        if kind in ("image", "video") and any(
            key in value for key in ("image", "path", "url", "data")
        ):
            return True
        return any(_contains_media(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_media(item) for item in value)
    return False


def flatten_record(
    row: dict[str, Any], rng: random.Random | None = None
) -> list[dict[str, Any]]:
    request = row["request"]
    targets = row["targets"]
    questions = request["questions"]
    if set(questions) != set(targets):
        raise ValueError("GS1 question IDs must match target IDs")

    if _contains_media(request["state"]):
        return []
    state_text = _text(request["state"])
    if len(state_text) > MAX_STATE_CHARS:
        return []
    rows = []
    for question_id, question in questions.items():
        kind = question["type"]
        criteria = question.get("criteria") or {}
        if kind == "choice":
            options = [
                _text(value if value is not None else key)
                for key, value in criteria.items()
            ]
            candidate_ids = list(criteria)
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
        if not valid_option_count(len(candidate_ids)):
            continue
        if len(target_ids) != len(candidate_ids) or set(target_ids) != set(
            candidate_ids
        ):
            raise ValueError(f"Target candidates do not match question {question_id!r}")
        target_by_id = dict(zip(target_ids, distribution, strict=True))
        target_probs = [float(target_by_id[key]) for key in candidate_ids]
        total = (
            math.fsum(target_probs)
            if all(math.isfinite(p) for p in target_probs)
            else math.nan
        )
        if any(
            not math.isfinite(value) or value < 0 for value in target_probs
        ) or not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                f"Invalid target distribution for question {question_id!r}"
            )
        order = list(range(len(candidate_ids)))
        if rng is not None and kind != "score":
            rng.shuffle(order)
        rows.append(
            render_options(
                state_text,
                kind,
                _text(question.get("instructions") or question_id),
                [candidate_ids[i] for i in order],
                [options[i] for i in order],
                [target_probs[i] / total for i in order],
            )
        )
    return rows


def sample_gs1(path: Path, count: int, seed: int, split: str) -> list[dict]:
    rng = random.Random(f"sample:{split}:{seed}")
    option_rng = random.Random(f"options:{split}:{seed}")
    rows = []
    with path.open() as source:
        questions = (
            row
            for line in source
            for row in flatten_record(json.loads(line), option_rng)
        )
        for seen, row in enumerate(questions, start=1):
            if seen <= count:
                rows.append(row)
            else:
                index = rng.randrange(seen)
                if index < count:
                    rows[index] = row
    if len(rows) != count:
        raise ValueError(
            f"Found {len(rows)} eligible questions in {path}, requested {count}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--input", type=Path, help="GS1 training JSONL")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--num-held-out", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.input and (not args.calibration or not args.eval):
        parser.error("GS1 input requires separate --calibration and --eval files")
    if args.input and (args.num_samples < 1 or args.num_held_out < 1):
        parser.error("Sample counts must be positive")
    if (
        args.input
        and len({path.resolve() for path in (args.input, args.calibration, args.eval)})
        != 3
    ):
        parser.error("Train, calibration and evaluation inputs must be distinct")
    args.out = args.out or Path("./assets/hf") / (
        "gs1-distribution" if args.input else "bev-decision-150K"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    if args.input:
        splits = {}
        states = set()
        for name, path, count in (
            ("train", args.input, args.num_samples),
            ("calib", args.calibration, args.num_held_out),
            ("eval", args.eval, args.num_held_out),
        ):
            rows = sample_gs1(path, count, args.seed, name)
            split_states = {row["state"] for row in rows}
            if states & split_states:
                raise ValueError(f"Sampled {name} states overlap an earlier split")
            states.update(split_states)
            splits[name] = rows
        for name, rows in splits.items():
            write(rows, args.out / f"{name}.jsonl")
        return
    write(flatten("train"), args.out / "train.jsonl")
    # BEV train and test share no states. Split test by state so the
    # temperature-fit questions and the reported questions share none either.
    test = flatten("test")
    calib, evals, calib_states = [], [], set()
    for row in test:
        if row["state"] in calib_states or len(calib_states) < 250:
            calib_states.add(row["state"])
            calib.append(row)
        else:
            evals.append(row)
    write(calib, args.out / "calib.jsonl")
    write(evals[:1500], args.out / "eval.jsonl")


if __name__ == "__main__":
    main()
