# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate the DCP checkpoints of a training run with lm-evaluation-harness.

Runs as its own job next to (never inside) the training job. It reads the
training config with the same CLI as ``torchtitan.train``, loads each
``step-N`` checkpoint natively (no HF conversion), and writes lm-eval results
to ``<dump_folder>/evals/step-N/<suite>.json``.

Usage::

    torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \\
        --tasks trainstation_quick [--steps 1000 2000 | --watch] \\
        -- --module llama3 --config llama3_8b [training overrides]

Without ``--steps``, every completed checkpoint that has no results for the
suite yet is evaluated. ``--watch`` keeps polling for new checkpoints until
the final training step has been evaluated.
"""

import dataclasses
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import lm_eval
import torch
import torch.distributed as dist
import tyro
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager

from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.evals.checkpoint import (
    build_model,
    checkpoint_folder,
    eval_parallelism,
    list_checkpoint_steps,
    load_weights,
)
from torchtitan.experiments.evals.lm import TrainstationLM
from torchtitan.experiments.evals.wandb_logger import EvalWandBLogger
from torchtitan.observability.logging import init_logger
from torchtitan.tools import utils
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)

T = TypeVar("T")

TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")


@dataclass(kw_only=True, slots=True)
class EvalConfig:
    """Options of the eval job; the training config is passed after '--'."""

    tasks: list[str]
    """lm-eval task or group names, e.g. trainstation_quick."""

    steps: list[int] | None = None
    """Checkpoint steps to evaluate. Default: every completed checkpoint that
    has no results for these tasks yet."""

    watch: bool = False
    """Keep polling for new checkpoints until the final training step
    (training.steps) has been evaluated."""

    poll_interval: int = 300
    """Seconds between checks for new checkpoints in watch mode."""

    limit: float | None = None
    """Examples per task (a fraction if < 1). For quick checks only."""

    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    """Dtype the model is evaluated in."""

    tensor_parallel_degree: int = 1
    """Shard each model replica over this many GPUs, for models that do not
    fit on one. The remaining GPUs form data-parallel replicas."""

    wandb: bool = False
    """Also log results to W&B, next to the training run (see wandb_logger.py)."""

    num_tokens_per_batch: int | None = None
    """Tokens per packed forward pass. Default: max(16384, context length)."""

    output_folder: str | None = None
    """Where to write results. Default: <dump_folder>/evals."""

    def __post_init__(self) -> None:
        if self.watch and self.steps:
            raise ValueError("watch and steps are mutually exclusive.")


def parse_args(argv: list[str]) -> tuple[EvalConfig, list[str]]:
    if "--" not in argv:
        raise ValueError(
            "Pass the training config after '--', e.g. "
            "'-- --module llama3 --config llama3_8b'."
        )
    split = argv.index("--")
    return tyro.cli(EvalConfig, args=argv[:split]), argv[split + 1 :]


def init_distributed() -> torch.device:
    device = torch.device(utils.device_type, int(os.environ.get("LOCAL_RANK", 0)))
    utils.device_module.set_device(device)
    # gloo carries lm-eval's object gathers; nccl carries tensor collectives.
    dist.init_process_group(backend=f"cpu:gloo,{device.type}:nccl")
    return device


def broadcast_from_rank0(fn: Callable[[], T]) -> T:
    """Evaluate ``fn`` on rank 0 only and return its result on every rank."""
    value = [fn() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(value, src=0)
    return value[0]  # pyrefly: ignore [bad-return]


def suite_name(tasks: list[str]) -> str:
    return "+".join(tasks)


def results_path(output_folder: str, step: int, tasks: list[str]) -> str:
    return os.path.join(output_folder, f"step-{step}", f"{suite_name(tasks)}.json")


def pending_steps(args: EvalConfig, ckpt_folder: str, output_folder: str) -> list[int]:
    if args.steps:
        return args.steps
    # Rank 0 lists and broadcasts, so a checkpoint that appears mid-listing
    # cannot give ranks different work lists.
    return broadcast_from_rank0(
        lambda: [
            step
            for step in list_checkpoint_steps(ckpt_folder)
            if not os.path.exists(results_path(output_folder, step, args.tasks))
        ]
    )


def evaluate_step(
    step: int,
    *,
    args: EvalConfig,
    config: Trainer.Config,
    lm: TrainstationLM,
    task_manager: TaskManager,
    ckpt_folder: str,
    output_folder: str,
) -> dict[str, Any] | None:
    """Evaluate one checkpoint; returns the results record on global rank 0."""
    checkpoint = os.path.join(ckpt_folder, f"step-{step}")
    logger.info(f"Evaluating {checkpoint} on {args.tasks}")
    start = time.perf_counter()
    load_weights(lm.model, checkpoint)

    results = simple_evaluate(
        model=lm,
        tasks=args.tasks,
        limit=args.limit,
        task_manager=task_manager,
        log_samples=False,
    )
    # Every TP rank of data-parallel rank 0 gets results; one writes them.
    if dist.get_rank() != 0:
        return None
    assert results is not None
    record = {
        "step": step,
        "checkpoint": checkpoint,
        "module": config.model_spec.name,
        "flavor": config.model_spec.flavor,
        "dtype": args.dtype,
        "tensor_parallel_degree": args.tensor_parallel_degree,
        "limit": args.limit,
        "lm_eval_version": lm_eval.__version__,
        "eval_seconds": time.perf_counter() - start,
        "results": results["results"],
        "groups": results.get("groups", {}),
        "n-shot": results.get("n-shot", {}),
        "versions": results.get("versions", {}),
    }
    path = results_path(output_folder, step, args.tasks)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write then rename so pending_steps() never sees a partial file.
    with open(f"{path}.tmp", "w") as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(f"{path}.tmp", path)
    logger.info(f"Wrote {path} ({record['eval_seconds']:.0f}s)")
    return record


def main() -> None:
    init_logger()
    args, train_args = parse_args(sys.argv[1:])
    config = ConfigManager().parse_args(train_args)
    device = init_distributed()

    ckpt_folder = checkpoint_folder(config)
    output_folder = args.output_folder or os.path.join(config.dump_folder, "evals")
    max_context_length = config.training.max_context_length

    config.parallelism = eval_parallelism(
        config.parallelism,
        world_size=dist.get_world_size(),
        tensor_parallel_degree=args.tensor_parallel_degree,
    )
    parallel_dims = ParallelDims.from_config(config.parallelism, dist.get_world_size())
    lm = TrainstationLM(
        # Checkpoint values are cast into these parameters on every load.
        build_model(
            config,
            parallel_dims=parallel_dims,
            device=device,
            dtype=args.dtype,
        ),
        config.tokenizer.build(tokenizer_path=config.hf_assets_path),
        parallel_dims=parallel_dims,
        parallelism=config.parallelism,
        max_context_length=max_context_length,
        num_tokens_per_batch=args.num_tokens_per_batch
        or max(16384, max_context_length),
        max_num_documents=getattr(config.dataloader, "max_num_documents", None),
        device=device,
    )
    task_manager = TaskManager(include_path=TASKS_DIR)
    wandb_logger = None
    if args.wandb and dist.get_rank() == 0:
        wandb_logger = EvalWandBLogger(
            output_folder=output_folder,
            suite=suite_name(args.tasks),
            config={"eval": dataclasses.asdict(args), "train_args": train_args},
        )

    try:
        while True:
            steps = pending_steps(args, ckpt_folder, output_folder)
            for step in steps:
                record = evaluate_step(
                    step,
                    args=args,
                    config=config,
                    lm=lm,
                    task_manager=task_manager,
                    ckpt_folder=ckpt_folder,
                    output_folder=output_folder,
                )
                if wandb_logger is not None and record is not None:
                    wandb_logger.log(step, record)
            final_done = broadcast_from_rank0(
                lambda: os.path.exists(
                    results_path(output_folder, config.training.steps, args.tasks)
                )
            )
            if not args.watch or final_done:
                break
            if not steps:
                time.sleep(args.poll_interval)
    finally:
        if wandb_logger is not None:
            wandb_logger.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
