# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke script for the RL metrics module.

Run with no args for the console backend, or `--wandb <log_dir>` for the
W&B backend (uses `WANDB_MODE=offline` so no network call). Saved as a
file (not inlined) so it can be re-run.

    python -m torchtitan.experiments.rl.tests.smoke_metrics
    python -m torchtitan.experiments.rl.tests.smoke_metrics --wandb /tmp/wandb_smoke
"""

from __future__ import annotations

import argparse
import os
import shutil

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.tools.logging import init_logger, logger


def build_records(step: int) -> list[m.Metric]:
    """A representative payload that exercises every reduction type."""
    rewards = [0.1, 0.5, 0.5, 1.0]
    response_lens = [12, 18, 9, 27]
    return [
        # Mean + Max under one key.
        m.Metric("rollout/response_length", m.Mean.from_list(response_lens)),
        m.Metric("rollout/response_length", m.Max.from_list(response_lens)),
        # Stats — six sub-keys under one metric name.
        m.Metric("rollout/reward", m.Stats.from_list(rewards)),
        # Std on its own.
        m.Metric("rollout/group_reward_std", m.Std.from_list([0.0, 0.1, 0.2])),
        # Min on its own.
        m.Metric("rollout/group_reward_std", m.Min.from_list([0.0, 0.1, 0.2])),
        # NoReduce — already-reduced scalar.
        m.Metric("loss/total", m.NoReduce(0.345 + 0.01 * step)),
        m.Metric("train/grad_norm", m.NoReduce(1.21 - 0.005 * step)),
    ]


def smoke_console() -> int:
    logger.info("--- console smoke (allow_list=None, print everything) ---")
    log = m.MetricLogger.build(m.MetricsConfig())
    for step in range(3):
        log.log(step, build_records(step), console_allow_list=None)
    log.close()

    logger.info("--- console smoke (allow_list=['loss']) ---")
    filtered = m.MetricLogger.build(m.MetricsConfig())
    filtered.log(0, build_records(0), console_allow_list=["loss"])
    filtered.close()
    return 0


def smoke_wandb(log_dir: str) -> int:
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", "torchtitan-rl-smoke")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    logger.info("--- wandb (offline) smoke ---")
    log = m.MetricLogger.build(
        m.MetricsConfig(enable_wandb=True),
        log_dir=log_dir,
        config_dict={"smoke": True, "version": 1},
    )
    for step in range(3):
        log.log(step, build_records(step), console_allow_list=None)
    log.close()

    # Confirm the offline run was actually written.
    runs = [
        entry
        for entry in os.listdir(os.path.join(log_dir, "wandb"))
        if entry.startswith("offline-run-")
    ]
    if not runs:
        logger.error("no offline-run-* directory found under %s/wandb", log_dir)
        return 1
    logger.info("wandb offline run: %s", runs[0])
    return 0


def main() -> int:
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb",
        metavar="LOG_DIR",
        default=None,
        help="Run the W&B (offline) smoke; writes runs into LOG_DIR/wandb.",
    )
    args = parser.parse_args()

    if args.wandb is None:
        return smoke_console()
    return smoke_wandb(args.wandb)


if __name__ == "__main__":
    raise SystemExit(main())
