# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate a checkpoint and report its validation loss.

This is the eval runner :class:`~torchtitan.components.async_eval.AsyncEval`
launches by default, and it doubles as the reference implementation of the
runner contract: take ``--checkpoint-dir``, ``--output-dir``, ``--step`` and
``--result-path``, then write the metrics as JSON to ``--result-path``::

    {"step": 100, "metrics": {"loss": 2.31, "perplexity": 10.07}}

A runner is free to compute anything else (say, task accuracies from an
evaluation harness) as long as it follows that contract. All other arguments are
the regular torchtitan job arguments, so an eval job is configured exactly like
the training job it evaluates.

Standalone usage:

    torchrun --nproc_per_node=1 \\
        -m torchtitan_recipes.async_eval.example_loss_eval \\
        --module llama3 --config llama3_8b \\
        --checkpoint-dir outputs/checkpoint/step-1000 --step 1000
"""

import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import cast

import torch

from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.tools import filesystem
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer


@dataclass
class RunnerArgs:
    """The arguments every async eval runner receives."""

    checkpoint_dir: str
    """Model-only checkpoint to evaluate."""

    step: int
    """Training step the checkpoint was taken at, used as the metrics step."""

    output_dir: str = ""
    """Folder for this eval job's outputs. Defaults to the job's dump_folder."""

    result_path: str = ""
    """File to write the metrics JSON to. Empty means do not write results."""


def parse_runner_args(args: list[str]) -> tuple[RunnerArgs, list[str]]:
    """Split the runner contract arguments from the torchtitan job arguments."""

    flags = {
        "--checkpoint-dir": "checkpoint_dir",
        "--output-dir": "output_dir",
        "--step": "step",
        "--result-path": "result_path",
    }
    values: dict[str, str] = {}
    job_args: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        name = flags.get(arg.split("=", 1)[0])
        if name is None:
            job_args.append(arg)
        elif "=" in arg:
            values[name] = arg.split("=", 1)[1]
        elif i + 1 < len(args):
            values[name] = args[i + 1]
            i += 1
        else:
            raise ValueError(f"{arg} requires a value")
        i += 1

    for required in ("checkpoint_dir", "step"):
        if required not in values:
            raise ValueError(
                f"--{required.replace('_', '-')} is required. Example: "
                "--checkpoint-dir outputs/checkpoint/step-100 --step 100"
            )

    return (
        RunnerArgs(
            checkpoint_dir=values["checkpoint_dir"],
            step=int(values["step"]),
            output_dir=values.get("output_dir", ""),
            result_path=values.get("result_path", ""),
        ),
        job_args,
    )


def eval_config(config: Trainer.Config, runner_args: RunnerArgs) -> Trainer.Config:
    """Turn a training job config into a config that only evaluates.

    The eval job loads the checkpoint it was given, writes no checkpoint of its
    own, and stays out of the training job's metric backends: the trainer logs
    the results this job reports back.
    """

    checkpoint_dir = runner_args.checkpoint_dir
    if not filesystem.is_remote(checkpoint_dir):
        checkpoint_dir = os.path.abspath(checkpoint_dir)

    if runner_args.output_dir:
        config.dump_folder = runner_args.output_dir
    config.checkpoint = replace(
        config.checkpoint,
        enable=True,
        load_only=True,
        initial_load_path=checkpoint_dir,
        initial_load_model_only=True,
    )
    config.validator = replace(config.validator, enable=True)
    config.metrics = replace(
        config.metrics, enable_tensorboard=False, enable_wandb=False
    )
    config.async_eval = replace(config.async_eval, enable=False)
    return config


def main() -> None:
    """Main entry point for evaluation."""
    init_logger()

    runner_args, job_args = parse_runner_args(sys.argv[1:])
    config = eval_config(
        cast(Trainer.Config, ConfigManager().parse_args(job_args)), runner_args
    )

    sl.init_structured_logger(
        source="evaluation",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )

    trainer: Trainer | None = None
    try:
        trainer = config.build()
        if not trainer.checkpointer.load():
            raise FileNotFoundError(
                f"No checkpoint to evaluate at {config.checkpoint.initial_load_path}"
            )

        begin = time.monotonic()
        loss = trainer.validator.validate(trainer.model_parts, runner_args.step)
        elapsed = time.monotonic() - begin

        metrics = {"time(s)": elapsed}
        # A custom validator may report through its own channels instead of
        # returning a loss (e.g. the Flux validator, which saves images).
        if loss is not None:
            metrics["loss"] = loss
            metrics["perplexity"] = math.exp(min(loss, 100.0))
        logger.info(f"Evaluated step {runner_args.step}: {metrics}")

        is_rank_zero = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        if runner_args.result_path and is_rank_zero:
            write_result(runner_args.result_path, runner_args.step, metrics)
    finally:
        if trainer:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def write_result(result_path: str, step: int, metrics: dict[str, float]) -> None:
    """Report metrics back to whoever launched this job."""

    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    # Write and rename so that a reader never sees a partial result.
    tmp_path = f"{result_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump({"step": step, "metrics": metrics}, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, result_path)
    logger.info(f"Wrote eval result to {result_path}")


if __name__ == "__main__":
    main()
