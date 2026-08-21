# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inspect a synthetic length distribution without running training.

Example spec files:

  buckets.json:
    {"type": "buckets",
     "buckets": [{"min_len": 1, "max_len": 128, "weight": 3},
                 {"min_len": 2048, "max_len": 4096, "weight": 1}]}

  parametric.json:
    {"type": "parametric", "kind": "lognormal",
     "min_len": 1, "max_len": 8192, "mean": 6.0, "std": 1.0}

Usage:
  python -m scripts.preview_synthetic_lengths --spec buckets.json \\
      --seed 0 --samples 100000 --dp 8 --per-rank-batch 4
"""

import argparse
import json
from typing import Any

import numpy as np
from torchtitan.components.data.synthetic import (
    BucketLengthSpec,
    LengthBucket,
    LengthSpec,
    ParametricLengthSpec,
)


def build_spec(obj: dict[str, Any]) -> LengthSpec:
    spec_type = obj["type"]
    if spec_type == "buckets":
        return BucketLengthSpec(
            buckets=tuple(LengthBucket(**b) for b in obj["buckets"])
        )
    if spec_type == "parametric":
        params = {k: v for k, v in obj.items() if k != "type"}
        return ParametricLengthSpec(**params)
    raise ValueError(f"unknown spec type {spec_type!r}")


def summarize(lengths: np.ndarray) -> dict[str, float]:
    lengths = np.asarray(lengths)
    if lengths.size == 0:
        return {
            "count": 0,
            "min": 0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0,
            "mean": 0.0,
            "total_tokens": 0,
        }
    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "total_tokens": int(lengths.sum()),
    }


def histogram(lengths: np.ndarray, bins: int = 20, width: int = 50) -> str:
    counts, edges = np.histogram(lengths, bins=bins)
    peak = counts.max() or 1
    lines = []
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(width * count / peak)
        lines.append(f"[{int(lo):>7}, {int(hi):>7}) {count:>8} {bar}")
    return "\n".join(lines)


def dp_balance(
    lengths: np.ndarray, dp_world_size: int, per_rank_batch: int
) -> dict[str, float]:
    """Simulate batching+sharding and measure per-step token imbalance."""
    per_step = dp_world_size * per_rank_batch
    usable = (lengths.size // per_step) * per_step
    if usable == 0:
        return {
            "num_steps": 0,
            "max_over_mean": 0.0,
            "worst_step_max_over_mean": 0.0,
            "padding_waste_fraction": 0.0,
        }
    steps = lengths[:usable].reshape(-1, dp_world_size, per_rank_batch)
    rank_tokens = steps.sum(axis=2)  # (num_steps, dp_world_size)
    step_mean = rank_tokens.mean(axis=1)
    step_max = rank_tokens.max(axis=1)
    ratios = step_max / np.clip(step_mean, 1, None)
    # padded cost = per-step slowest rank drives the step
    padded = steps.max(axis=2) * per_rank_batch  # (num_steps, dp_world_size)
    waste = 1.0 - rank_tokens.sum() / np.clip(padded.sum(), 1, None)
    return {
        "num_steps": int(rank_tokens.shape[0]),
        "max_over_mean": float(ratios.mean()),
        "worst_step_max_over_mean": float(ratios.max()),
        "padding_waste_fraction": float(waste),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to JSON spec file.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--dp", type=int, default=0, help="DP world size (0 = skip).")
    parser.add_argument("--per-rank-batch", type=int, default=1)
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = build_spec(json.load(f))

    rng = np.random.default_rng(args.seed)
    lengths = spec.sample(rng, args.samples)

    stats = summarize(lengths)
    print(f"Spec: {args.spec}  seed={args.seed}  samples={args.samples}")
    for key, value in stats.items():
        print(f"  {key:>13}: {value}")
    print("\nHistogram:")
    print(histogram(lengths, bins=args.bins))

    if args.dp > 0:
        report = dp_balance(lengths, args.dp, args.per_rank_batch)
        print(f"\nDP balance (dp={args.dp}, per_rank_batch={args.per_rank_batch}):")
        for key, value in report.items():
            print(f"  {key:>25}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
