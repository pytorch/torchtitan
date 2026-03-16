# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.components.metrics import build_device_memory_monitor
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.observability.aggregation import logging_worker
from torchtitan.observability.logging_boundary import EveryNSteps
from torchtitan.observability.metrics import (
    MaxMetric,
    MeanMetric,
    record_metric,
    SumMetric,
)
from torchtitan.observability.step_state import set_step
from torchtitan.observability.structured_logging import record_event
from torchtitan.tools import utils


class MetricsProcessor(Configurable):
    """Metrics lifecycle for distributed training.

    Handles three responsibilities:
    1. Step tracking and log scheduling (every N steps + validation steps).
    2. Derived metrics: GPU memory peak stats and throughput (TPS, TFLOPS, MFU).
    3. Background logging subprocess on rank 0 that reads per-rank experiment
       JSONL, aggregates across ranks, and writes to WandB/TB/console.

    The trainer calls methods in this order each step:
        set_step → reset_training_counters → [train_step] → record_memory →
        record_throughput → should_log → log

    Args:
        config: Controls log frequency, backend selection, and console keys.
        parallel_dims: Mesh topology. Used for throughput normalization
            (non_data_parallel_size) and rank-0 detection.
        dump_folder: Root output directory for JSONL and backend logs.
        ft_enable: Whether fault-tolerant training is enabled.
        ft_replica_id: FT replica ID, used to partition TB directories.
        config_dict: Passed to ``wandb.init(config=...)``.
        tag: Optional prefix for metric names in WandB/TB.
        has_quantization: Skip MFU when quantization changes effective FLOPS.

    Example:
        mp = MetricsProcessor.Config(log_freq=10, enable_wandb=True).build(
            parallel_dims=parallel_dims, dump_folder="./outputs",
        )
        mp.num_flops_per_token = model_num_flops
        for step in range(1, num_steps + 1):
            mp.set_step(step)
            mp.reset_training_counters()
            train_step(...)
            mp.record_memory()
            mp.record_throughput()
            if mp.should_log(step):
                mp.log(step)
        mp.close()
    """

    # Prefix for throughput and memory metric keys (e.g., "trainer_memory/...")
    _TRAIN_PREFIX = "trainer"
    _VAL_PREFIX = "validator"

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to log to backends (e.g. wandb). Also gates expensive
        metrics computation (.item(), collectives)."""

        enable_tensorboard: bool = False
        """Whether to log metrics to TensorBoard"""

        save_tb_folder: str = "tb"
        """Folder to dump TensorBoard states"""

        enable_wandb: bool = False
        """Whether to log metrics to Weights & Biases"""

        console_log_metric_keys: list[str] = field(
            default_factory=lambda: [
                "training/loss_mean",
                "training/grad_norm_max",
                "trainer_memory/reserved_gib_max",
                "trainer_throughput/tps_mean",
                "trainer_throughput/tflops_mean",
                "trainer_throughput/mfu_pct_mean",
            ]
        )
        """Training metric keys to print to console each log step."""

        console_log_validation_keys: list[str] = field(
            default_factory=lambda: [
                "validation/loss_mean",
                "validator_memory/reserved_gib_max",
                "validator_throughput/tps_mean",
            ]
        )
        """Validation metric keys to print to console."""

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        dump_folder: str = "./outputs",
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
        has_quantization: bool = False,
    ):
        self.config = config
        self._has_quantization = has_quantization
        self.parallel_dims = parallel_dims
        self._step = 0
        self._force_log = False

        # Schedule: log on step 1 + every log_freq steps
        self._log_schedule = EveryNSteps(every_n=config.log_freq, additional_steps={1})

        # Device memory monitor
        self.device_memory_monitor = build_device_memory_monitor()

        # TFLOPS/MFU: set by trainer after construction (-1 = skip)
        self.num_flops_per_token: int = -1
        self._gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )

        self.reset_training_counters()
        self.reset_val_counters()

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
                    "save_tb_folder": config.save_tb_folder,
                    "config_dict": config_dict,
                    "tag": tag,
                    "console_log_metric_keys": config.console_log_metric_keys,
                    "console_log_validation_keys": config.console_log_validation_keys,
                    "ft_enable": ft_enable,
                    "ft_replica_id": ft_replica_id,
                },
                daemon=True,
            )
            self._log_process.start()

    # ----
    # Step management
    # ----

    def set_step(self, step: int, force_log: bool = False) -> None:
        """Set current step. Call before train_step().

        Args:
            step: Current training step number.
            force_log: When True, should_log() returns True regardless of
                schedule. Used to ensure loss is computed on validation steps.
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
    # Derived metrics
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

        ndp = self.parallel_dims.non_data_parallel_size
        tps = ntokens / (time_delta * ndp) if time_delta > 0 else 0
        record_metric(f"{prefix}_throughput/tps_mean", MeanMetric(sum=tps))

        if self.num_flops_per_token > 0:
            tflops = self.num_flops_per_token * tps / 1e12
            record_metric(f"{prefix}_throughput/tflops_mean", MeanMetric(sum=tflops))
            if self._gpu_peak_flops > 0 and not self._has_quantization:
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
        record_metric(
            f"{prefix}_memory/ooms_sum",
            SumMetric(value=mem.num_ooms),
        )
        record_metric(
            f"{prefix}_memory/reserved_pct_max",
            MaxMetric(value=mem.max_reserved_pct),
        )
        record_metric(
            f"{prefix}_memory/active_pct_max",
            MaxMetric(value=mem.max_active_pct),
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
