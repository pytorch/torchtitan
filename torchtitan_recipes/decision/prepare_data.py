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
import random
import string
from pathlib import Path

from datasets import load_dataset

LETTERS = string.ascii_uppercase
MAX_STATE_CHARS = 4000


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
    options = "\n".join(f"{LETTERS[i]}) {d}" for i, (_, d) in enumerate(options))
    title = {"noul": "yes/no", "choice": "choice", "score": "ordered scale"}[kind]
    gold = keys.index(label)
    return {
        "prompt": f"Context:\n{state}\n\nQuestion ({title}): "
        f"{question['instructions']}\nOptions:\n{options}\nAnswer:",
        "answer": " " + LETTERS[gold],
        "gold": gold,
        "keys": keys,
        "type": kind,
        "state": state,
        "instructions": question["instructions"],
    }


def flatten(split: str) -> list[dict]:
    dataset = load_dataset("avbiswas/bev-decision-150K", split=split)
    option_rng = random.Random(f"options:{split}")
    rows = []
    for state, questions in zip(dataset["state"], dataset["questions_json"]):
        if len(state) > MAX_STATE_CHARS:
            continue
        for question in json.loads(questions).values():
            num_options = 2 if question["type"] == "noul" else len(question["criteria"])
            if 2 <= num_options <= len(LETTERS):
                rows.append(render(state, question, option_rng))
    random.Random(0).shuffle(rows)
    return rows


def write(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    print(f"{path}: {len(rows)} questions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("./assets/hf/bev-decision-150K")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
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
