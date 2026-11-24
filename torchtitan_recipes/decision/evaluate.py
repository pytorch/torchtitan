# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Score letter-slot decisions and print answers in Jev's response format.

Reads the next-token logits at ``Answer:`` restricted to the valid option
letters, fits a temperature on ``calib.jsonl``, and reports accuracy, Brier
score, ECE, and the fraction of ``eval.jsonl`` that can be answered at a 5%
error budget.

Usage:
  python -m torchtitan_recipes.decision.evaluate \
    --hf-assets ./assets/hf/Qwen3-1.7B-Base [--checkpoint outputs/.../exports/step-600]
"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prepare_data import LETTERS


@torch.no_grad()
def letter_logits(model, tokenizer, rows: list[dict], batch_size: int = 16):
    letter_ids = [tokenizer.encode(" " + c, add_special_tokens=False) for c in LETTERS]
    if any(len(ids) != 1 for ids in letter_ids):
        raise ValueError("Each option letter must be a single ' X' token")
    letter_ids = torch.tensor([ids[0] for ids in letter_ids], device=model.device)
    out = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        enc = tokenizer(
            [r["prompt"] for r in batch], return_tensors="pt", padding=True
        ).to(model.device)
        # Left padding puts every prompt's answer slot at the last position;
        # position ids keep each prompt's RoPE positions starting at 0.
        position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        logits = model(**enc, position_ids=position_ids, logits_to_keep=1).logits
        logits = logits[:, -1, letter_ids]
        logits = logits.float().cpu()
        out += [lg[: len(r["keys"])] for r, lg in zip(batch, logits)]
    return out


def target_probs(row, like):
    if "target_probs" in row:
        target = like.new_tensor(row["target_probs"])
        return target / target.sum()
    target = torch.zeros_like(like)
    target[row["gold"]] = 1
    return target


def nll(logits, rows, temperature: float) -> float:
    return sum(
        -(target_probs(row, scores) * (scores / temperature).log_softmax(-1))
        .sum()
        .item()
        for scores, row in zip(logits, rows, strict=True)
    ) / len(rows)


def metrics(probs, rows, num_bins: int = 10) -> dict:
    n = len(rows)
    conf = [p.max().item() for p in probs]
    targets = [target_probs(row, p) for p, row in zip(probs, rows, strict=True)]
    correct = [
        target[p.argmax()].item() for p, target in zip(probs, targets, strict=True)
    ]
    brier = sum(
        ((p - target).square().sum() + 1 - target.square().sum()).item()
        for p, target in zip(probs, targets, strict=True)
    )
    ece = 0.0
    for b in range(num_bins):
        idx = [i for i, c in enumerate(conf) if b / num_bins < c <= (b + 1) / num_bins]
        if idx:
            gap = sum(correct[i] - conf[i] for i in idx)
            ece += abs(gap) / n
    # Largest k such that the k most confident answers have at most 5% errors.
    answered, errors = 0, 0.0
    order = sorted(range(n), key=lambda j: -conf[j])
    for k, i in enumerate(order, 1):
        errors += 1 - correct[i]
        if errors <= 0.05 * k:
            answered = k
    return {
        "n": n,
        "acc": sum(correct) / n,
        "brier": brier / n,
        "ece": ece,
        "answered_at_5pct_error": answered / n,
    }


def system_one_answer(p: torch.Tensor, row: dict) -> dict:
    """Format ``p`` as a Jev response: P(true) for noul, expected level for score, argmax key for choice."""
    probs = {k: round(v, 3) for k, v in zip(row["keys"], p.tolist())}
    if row["type"] == "noul":
        return {"type": "noul", "noul": probs["true"]}
    if row["type"] == "score":
        score = sum(i * v for i, v in enumerate(p.tolist()))
        return {"type": "score", "score": round(score, 3), "probabilities": probs}
    best = int(p.argmax())
    return {"type": "choice", "choice": row["keys"][best], "probabilities": probs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-assets", required=True)
    parser.add_argument("--checkpoint", help="TorchTitan HF-format checkpoint dir")
    parser.add_argument(
        "--data", type=Path, default=Path("./assets/hf/bev-decision-150K")
    )
    parser.add_argument("--num-examples", type=int, default=6)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_assets, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.hf_assets, dtype=torch.bfloat16)
    if args.checkpoint:
        state_dict = {}
        for path in Path(args.checkpoint).glob("*.safetensors"):
            state_dict.update(load_file(path))
        if "model.language_model.embed_tokens.weight" in state_dict:
            # TorchTitan exports text weights under the multimodal checkpoint prefix.
            state_dict = {
                "model." + key.removeprefix("model.language_model."): value
                for key, value in state_dict.items()
                if key.startswith("model.language_model.")
            } | {
                key: state_dict[key] for key in ("lm_head.weight",) if key in state_dict
            }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if model.config.tie_word_embeddings:
            missing = [key for key in missing if key != "lm_head.weight"]
        if missing or unexpected:
            raise ValueError(f"missing={missing} unexpected={unexpected}")
    model = model.cuda().eval()

    def load(name):
        return [
            json.loads(line) for line in (args.data / name).read_text().splitlines()
        ]

    calib, rows = load("calib.jsonl"), load("eval.jsonl")
    calib_logits = letter_logits(model, tokenizer, calib)
    grid = [math.exp(x / 20) for x in range(-40, 61)]
    temperature = min(grid, key=lambda t: nll(calib_logits, calib, t))
    probs = [
        torch.softmax(lg / temperature, -1)
        for lg in letter_logits(model, tokenizer, rows)
    ]

    report: dict[str, object] = {"temperature": round(temperature, 3)}
    for kind in ("all", "choice", "noul", "score"):
        idx = [i for i, r in enumerate(rows) if kind in ("all", r["type"])]
        if idx:
            report[kind] = metrics([probs[i] for i in idx], [rows[i] for i in idx])
    print(json.dumps(report, indent=1))
    for p, row in zip(probs[: args.num_examples], rows):
        print(f"\nstate: {row['state'][:200]!r}\nquestion: {row['instructions']}")
        print(f"target: {system_one_answer(target_probs(row, p), row)}")
        print(f"answer: {json.dumps(system_one_answer(p, row))}")


if __name__ == "__main__":
    main()
