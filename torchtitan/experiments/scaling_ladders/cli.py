# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin CLI over ``Llama3Ladder`` (precedent: OLMo-core ``internal/ladder.py``).

Read commands (dry-run, status, metrics, compare) run in-process and print JSON;
``run`` and ``sweep --execute`` spawn ``torchrun`` subprocesses via the ladder. Run::

    python -m torchtitan.experiments.scaling_ladders.cli dry-run --size 100M
"""

import argparse
import json

from . import LADDER
from .ladder import _spawn_run

_OVERRIDE_FLAGS = {
    "lr_multiplier": float,
    "weight_decay": float,
    "chinchilla_multiple": float,
    "tokens_per_param": int,
    "decay_fraction": float,
    "seed": int,
}


def _add_override_flags(parser: argparse.ArgumentParser) -> None:
    for name, cast in _OVERRIDE_FLAGS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=cast)


def _overrides(args: argparse.Namespace) -> dict:
    return {
        name: getattr(args, name)
        for name in _OVERRIDE_FLAGS
        if getattr(args, name, None) is not None
    }


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _parse_xc(value: str) -> float:
    """Parse a Chinchilla multiple written as ``1``, ``1xC``, or ``0.5x``."""
    return float(value.lower().removesuffix("xc").removesuffix("x"))


def _parse_grid(entries: list[str]) -> dict[str, list]:
    grid: dict[str, list] = {}
    for entry in entries:
        key, _, raw = entry.partition("=")
        cast = _OVERRIDE_FLAGS.get(key, float)
        grid[key] = [cast(v) for v in raw.split(",")]
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama3 scaling-ladder CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("dry-run", "run", "status", "metrics", "launch-command"):
        p = sub.add_parser(name)
        p.add_argument("--size", required=True)
        _add_override_flags(p)
    for name in ("dry-run-all", "status-all", "metrics-all"):
        sub.add_parser(name)

    sweep_p = sub.add_parser("sweep")
    sweep_p.add_argument("--sizes", required=True, help="Comma-separated rungs.")
    sweep_p.add_argument("--grid", nargs="+", required=True, help="key=v1,v2,...")
    sweep_p.add_argument("--execute", action="store_true")

    compare_p = sub.add_parser("compare")
    compare_p.add_argument("--runs", required=True, help="Path to sweep-output JSON.")
    compare_p.add_argument("--metric", default="val_loss")
    compare_p.add_argument(
        "--at",
        type=_parse_xc,
        required=True,
        help="Matched Chinchilla multiple, e.g. 1 or 1xC.",
    )

    args = parser.parse_args()
    cmd = args.command

    if cmd == "dry-run":
        _emit(LADDER.plan(args.size, **_overrides(args)))
    elif cmd == "dry-run-all":
        _emit({r: LADDER.plan(r) for r in LADDER.rungs()})
    elif cmd == "status":
        _emit(LADDER.status(args.size, **_overrides(args)))
    elif cmd == "status-all":
        _emit({r: LADDER.status(r) for r in LADDER.rungs()})
    elif cmd == "metrics":
        _emit(LADDER.metrics(args.size, **_overrides(args)))
    elif cmd == "metrics-all":
        _emit({r: LADDER.metrics(r) for r in LADDER.rungs()})
    elif cmd == "launch-command":
        world_size = LADDER.compute_for(args.size).world_size
        config = f"llama3_ladder_{args.size.lower()}"
        print(
            f"NGPU={world_size} MODULE=scaling_ladders CONFIG={config} ./run_train.sh"
        )
    elif cmd == "run":
        _spawn_run(
            args.size, _overrides(args), LADDER.compute_for(args.size).world_size
        )
    elif cmd == "sweep":
        specs = LADDER.sweep(
            args.sizes.split(","), _parse_grid(args.grid), execute=args.execute
        )
        _emit(specs)
    elif cmd == "compare":
        with open(args.runs) as f:
            runs = json.load(f)
        _emit(LADDER.compare(runs, args.metric, args.at))


if __name__ == "__main__":
    main()
