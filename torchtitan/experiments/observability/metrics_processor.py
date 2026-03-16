# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Toy MetricsProcessor for the observability experiment."""

import multiprocessing
from dataclasses import dataclass, field

import torch

from torchtitan.config import Configurable
from torchtitan.observability import record_event
from torchtitan.observability.aggregation import logging_worker
from torchtitan.observability.step_state import set_step


class MetricsProcessor(Configurable):
    """Step context and logging subprocess for the toy trainer.

    Mirrors the method order of components/metrics.py MetricsProcessor
    so the toy and production versions are easy to compare.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable_wandb: bool = True
        enable_tensorboard: bool = False
        console_log_metric_keys: list[str] = field(default_factory=list)

    def __init__(self, config: Config, *, dump_folder: str, rank: int):
        self._step: int = 0
        self._rank = rank

        # Spawn the logging subprocess if any output is enabled.
        # logging_worker reads experiment JSONL from all ranks, aggregates
        # metrics, and flushes to WandB/TB/console. Runs in a separate
        # process so it never blocks training.
        needs_subprocess = (
            config.enable_wandb
            or config.enable_tensorboard
            or config.console_log_metric_keys
        )
        self._log_queue: multiprocessing.Queue | None = None
        self._log_process: multiprocessing.Process | None = None
        if rank == 0 and needs_subprocess:
            self._log_queue = multiprocessing.Queue()
            self._log_process = multiprocessing.Process(
                target=logging_worker,
                args=(self._log_queue, dump_folder),
                kwargs={
                    "enable_wandb": config.enable_wandb,
                    "enable_tensorboard": config.enable_tensorboard,
                    "console_log_metric_keys": config.console_log_metric_keys,
                },
                daemon=True,
            )
            self._log_process.start()

    def set_step(self, step: int) -> None:
        """Set the current training step."""
        self._step = step
        set_step(step)
        record_event({"train.step": step})

    def log(self, step: int) -> None:
        """Signal the logging subprocess to aggregate and write.

        All ranks participate in the barrier so the subprocess can read
        all JSONL files. Non-blocking after barrier (~0.1ms).
        """
        torch.distributed.barrier()
        if self._log_queue is not None:
            self._log_queue.put(step)

    def close(self) -> None:
        """Shut down the logging subprocess."""
        if self._log_queue is not None:
            self._log_queue.put(None)
        if self._log_process is not None:
            self._log_process.join(timeout=10)
            if self._log_process.is_alive():
                self._log_process.terminate()
