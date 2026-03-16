# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Toy MetricsProcessor for the observability experiment."""

import multiprocessing
import time
from dataclasses import dataclass, field

import torch

from torchtitan.components.metrics import build_device_memory_monitor
from torchtitan.config import Configurable
from torchtitan.observability import record_event
from torchtitan.observability.aggregation import logging_worker
from torchtitan.observability.logging_boundary import EveryNSteps
from torchtitan.observability.metrics import (
    MaxMetric,
    MeanMetric,
    record_metric,
    SumMetric,
)
from torchtitan.observability.step_state import set_step
from torchtitan.tools import utils


class MetricsProcessor(Configurable):
    """Step context, derived metrics, and logging subprocess for the toy trainer.

    Mirrors the method order of components/metrics.py MetricsProcessor
    so the toy and production versions are easy to compare.
    """

    _TRAIN_PREFIX = "trainer"
    _VAL_PREFIX = "validator"

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to log to backends (e.g. wandb). Also gates expensive
        metrics computation (.item(), collectives)."""
        enable_wandb: bool = True
        enable_tensorboard: bool = False
        console_log_metric_keys: list[str] = field(default_factory=list)
        console_log_validation_keys: list[str] = field(default_factory=list)

    def __init__(self, config: Config, *, dump_folder: str, rank: int, non_data_parallel_size: int = 1):
        self.config = config
        self._step: int = 0
        self._rank = rank
        self._force_log = False
        self._non_data_parallel_size = non_data_parallel_size

        # Schedule
        self._log_schedule = EveryNSteps(
            every_n=config.log_freq, additional_steps={1}
        )

        # Device memory monitor
        self.device_memory_monitor = build_device_memory_monitor()

        # TFLOPS/MFU: set by trainer after construction (-1 = skip)
        self.num_flops_per_token: int = -1
        self._gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )

        self.reset_training_counters()
        self.reset_val_counters()

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
                    "console_log_validation_keys": config.console_log_validation_keys,
                },
                daemon=True,
            )
            self._log_process.start()

    # ----
    # Step management
    # ----

    def set_step(self, step: int, force_log: bool = False) -> None:
        """Set current step. Call before train_step().

        force_log: For example, can be used to ensure loss is computed
        on validation steps.
        """
        self._step = step
        self._force_log = force_log
        set_step(step)
        record_event({"train.step": step})

    # ----
    # Schedule queries
    # ----

    def should_log(self, step: int) -> bool:
        """Returns True on log steps or when force_log was set."""
        return self._log_schedule(step) or self._force_log

    # ----
    # Counter resets
    # ----

    def reset_training_counters(self) -> None:
        """Reset throughput/memory counters for a new training measurement."""
        self._time_at_reset = time.perf_counter()
        self.ntokens_since_reset = 0
        self.device_memory_monitor.reset_peak_stats()

    def reset_val_counters(self) -> None:
        """Reset throughput/memory counters for a new validation measurement."""
        self._val_time_at_reset = time.perf_counter()
        self.val_ntokens_since_reset = 0
        self.device_memory_monitor.reset_peak_stats()

    # ----
    # Derived metrics (called every step, outside should_log gate)
    # ----

    def record_throughput(self, is_validation: bool = False) -> None:
        """Compute and record throughput from tokens since last reset."""
        if is_validation:
            time_delta = time.perf_counter() - self._val_time_at_reset
            ntokens = self.val_ntokens_since_reset
            prefix = self._VAL_PREFIX
        else:
            time_delta = time.perf_counter() - self._time_at_reset
            ntokens = self.ntokens_since_reset
            prefix = self._TRAIN_PREFIX

        ndp = self._non_data_parallel_size
        tps = ntokens / (time_delta * ndp) if time_delta > 0 else 0
        record_metric(f"{prefix}_throughput/tps_mean", MeanMetric(sum=tps))

        if self.num_flops_per_token > 0:
            tflops = self.num_flops_per_token * tps / 1e12
            record_metric(f"{prefix}_throughput/tflops_mean", MeanMetric(sum=tflops))
            if self._gpu_peak_flops > 0:
                mfu = 100 * self.num_flops_per_token * tps / self._gpu_peak_flops
                record_metric(f"{prefix}_throughput/mfu_pct_mean", MeanMetric(sum=mfu))

    def record_memory(self, is_validation: bool = False) -> None:
        """Record GPU memory peak stats since last reset."""
        prefix = self._VAL_PREFIX if is_validation else self._TRAIN_PREFIX
        mem = self.device_memory_monitor.get_peak_stats()
        record_metric(
            f"{prefix}_memory/reserved_gib_max",
            MaxMetric(value=mem.max_reserved_gib),
        )
        record_metric(
            f"{prefix}_memory/active_gib_max",
            MaxMetric(value=mem.max_active_gib),
        )
        record_metric(
            f"{prefix}_memory/alloc_retries_sum",
            SumMetric(value=mem.num_alloc_retries),
        )

    # ----
    # Flush
    # ----

    def log(self, step: int, is_validation: bool = False) -> None:
        """Signal the logging subprocess to aggregate and write.

        All ranks participate in the barrier so the subprocess can read
        all JSONL files. Non-blocking after barrier (~0.1ms).
        """
        torch.distributed.barrier()
        if self._log_queue is not None:
            self._log_queue.put((step, is_validation))

    def close(self) -> None:
        """Shut down the logging subprocess."""
        if self._log_queue is not None:
            self._log_queue.put(None)
        if self._log_process is not None:
            self._log_process.join(timeout=10)
            if self._log_process.is_alive():
                self._log_process.terminate()
