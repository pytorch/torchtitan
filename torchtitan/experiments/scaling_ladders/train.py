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

``--gpus``/``--fp8``/``--base-dump-folder`` let the concurrent launcher rebuild
the per-job ladder in the spawned worker: ``--gpus`` re-derives the rung's
``ComputeSpec`` for a GPU subset so the run() world-size assertion matches
``--nproc_per_node``; ``--fp8`` attaches the rowwise Float8 converter; and
``--base-dump-folder`` redirects outputs so the bf16 and fp8 arms do not collide.
"""

import argparse
from dataclasses import replace

from . import LADDER
from .planner import auto_compute_spec


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
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Attach the rowwise Float8 linear converter. Requires --compile.",
    )
    parser.add_argument(
        "--fp8-filter",
        type=str,
        help="Comma-separated fqn substrings to SKIP for fp8 (default: "
        "'lm_head,attention' = fp8 only the large MLP GEMMs).",
    )
    parser.add_argument(
        "--base-dump-folder",
        type=str,
        help="Override the ladder output root (keeps bf16/fp8 arms separate).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        help="Re-derive this rung's ComputeSpec for a GPU subset; must equal "
        "torchrun --nproc_per_node so the run() world-size assertion passes.",
    )
    parser.add_argument(
        "--attn-backend",
        type=str,
        help="Override the attention backend (flex, flex_flash, varlen).",
    )
    parser.add_argument(
        "--reduce-dtype",
        type=str,
        help="Gradient all-reduce dtype: float32 (default) or bfloat16.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        help="Override the per-device microbatch (memory knob; set by the OOM probe). "
        "Does not change the schedule -- it is snapped to a divisor of the per-rank batch.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Truncate training to this many steps (smoke test only; the final "
        "checkpoint is not written, so the run stays marked incomplete).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable the Kineto profiler (dumps a chrome trace under "
        "dump_folder/profiling/traces; for analysis runs).",
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

    ladder = LADDER
    if args.base_dump_folder is not None:
        ladder = replace(ladder, base_dump_folder=args.base_dump_folder)
    if args.attn_backend is not None:
        ladder = replace(ladder, attn_backend=args.attn_backend)
    if args.reduce_dtype is not None:
        ladder = replace(ladder, reduce_dtype=args.reduce_dtype)
    if args.compile:
        ladder = replace(ladder, compile=True)
    if args.fp8:
        from torchtitan.components.quantization.float8 import Float8LinearConverter

        # model_compile_enabled follows the ladder so rowwise fp8 gets compiled.
        # Manual filter: fp8 only the large MLP GEMMs, skipping the small attention
        # projections (and lm_head) that made fp8-everywhere a net loss. torchao's
        # auto_filter_small_kn cannot be used here -- it requires real nn.Linear
        # modules, but the converter swaps Config objects on meta, so it skips all.
        skip = (
            args.fp8_filter.split(",") if args.fp8_filter else ["lm_head", "attention"]
        )
        converter = Float8LinearConverter.Config(
            recipe_name="rowwise",
            filter_fqns=skip,
            model_compile_enabled=ladder.compile,
        )
        ladder = replace(ladder, converters=[converter])
    if args.gpus is not None:
        spec = auto_compute_spec(args.rung, ladder.policy, gpus=args.gpus)
        ladder = replace(
            ladder, compute_specs={**ladder.compute_specs, args.rung: spec}
        )

    ladder.run(
        args.rung,
        local_batch_size=args.local_batch_size,
        max_steps=args.max_steps,
        profile=args.profile,
        **overrides,
    )


if __name__ == "__main__":
    main()
