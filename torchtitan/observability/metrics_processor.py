# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MetricsProcessor: step context, derived metrics, and logging subprocess."""

import multiprocessing
from dataclasses import dataclass, field

import torch

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.observability.aggregation import logging_worker
from torchtitan.observability.step_state import set_step
from torchtitan.observability.structured_logging import record_event


class MetricsProcessor(Configurable):
    """Step context and logging subprocess.

    Records training metrics to experiment JSONL via record_metric.
    A non-blocking background subprocess reads JSONL, aggregates across ranks,
    and writes to WandB/TB/console.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable_wandb: bool = True
        enable_tensorboard: bool = False
        console_log_metric_keys: list[str] = field(default_factory=list)

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        dump_folder: str,
    ):
        self._step: int = 0
        self.parallel_dims = parallel_dims

        # Rank 0 runs a background process that reads experiment JSONL
        # from all ranks, reduces metrics, and logs to WandB/TB/console.
        needs_subprocess = (
            config.enable_wandb
            or config.enable_tensorboard
            or config.console_log_metric_keys
        )
        self._log_queue: multiprocessing.Queue | None = None
        self._log_process: multiprocessing.Process | None = None
        if torch.distributed.get_rank() == 0 and needs_subprocess:
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
