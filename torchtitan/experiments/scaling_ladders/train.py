# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torchrun entry point for API-driven and swept ladder runs.

Launched per rank by ``torchrun --nproc_per_node=<ws> -m
torchtitan.experiments.scaling_ladders.train --rung 100M [overrides]``. Overrides
are parsed from argv and threaded into ``LADDER.run`` so the policy re-runs from
them (a post-hoc tyro override could not re-derive LR/steps).
"""

import argparse
from dataclasses import replace

from . import LADDER


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one scaling-ladder rung.")
    parser.add_argument("--rung", required=True, help="Rung name, e.g. 100M.")
    parser.add_argument("--lr-multiplier", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--chinchilla-multiple", type=float)
    parser.add_argument("--tokens-per-param", type=int)
    parser.add_argument("--decay-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--compile", action="store_true", help="Enable torch.compile (model + loss)."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidates = {
        "lr_multiplier": args.lr_multiplier,
        "weight_decay": args.weight_decay,
        "chinchilla_multiple": args.chinchilla_multiple,
        "tokens_per_param": args.tokens_per_param,
        "decay_fraction": args.decay_fraction,
        "seed": args.seed,
    }
    overrides = {k: v for k, v in candidates.items() if v is not None}
    ladder = replace(LADDER, compile=True) if args.compile else LADDER
    ladder.run(args.rung, **overrides)


if __name__ == "__main__":
    main()
