# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module, device_type

# named tuple for passing device memory stats for logging
DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class DeviceMemoryMonitor:
    def __init__(self, device: str = f"{device_type}:0"):
        self.device = torch.device(device)  # device object
        self.device_name = device_module.get_device_name(self.device)
        self.device_index = device_module.current_device()
        self.device_capacity = device_module.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        device_module.reset_peak_memory_stats()
        device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        device_info = device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper()} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper()} OOM errors thrown.")

        return DeviceMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        device_module.reset_peak_memory_stats()


def build_device_memory_monitor():
    device_memory_monitor = DeviceMemoryMonitor(device_type)
    logger.info(
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


class BaseLogger:
    """Logger that does nothing, used when logging is disabled."""

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    def __init__(self, log_dir: str, tag: Optional[str] = None):
        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            self.writer.add_scalar(tag, v, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases."""

    def __init__(self, log_dir: str, tag: Optional[str] = None):
        # Import wandb here to avoid startup import
        import wandb

        self.wandb = wandb
        self.tag = tag

        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)

        self.wandb.init(
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            dir=log_dir,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wandb.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


def _get_metrics_rank(
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> int:
    """
    Determines which rank should log metrics.

    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel 'ZBVZeroBubble' schedule
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if job_config.experimental.pipeline_parallel_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)


def build_metric_logger(
    job_config: JobConfig, parallel_dims: ParallelDims, tag: Optional[str] = None
) -> BaseLogger:
    """
    Build an appropriate metric logger based on configuration.
    """
    metrics_config = job_config.metrics

    # Log initial config state
    logger.debug(
        f"Building logger with config: wandb={metrics_config.enable_wandb}, "
        f"tensorboard={metrics_config.enable_tensorboard}"
    )

    # Check if any logging backend is enabled
    has_logging_enabled = (
        metrics_config.enable_tensorboard or metrics_config.enable_wandb
    )

    # Determine if this rank should log
    should_log = has_logging_enabled
    if metrics_config.rank_0_only and should_log:
        metrics_rank = _get_metrics_rank(parallel_dims, job_config)
        should_log = torch.distributed.get_rank() == metrics_rank

    logger.debug(
        f"Logging decision: has_logging_enabled={has_logging_enabled}, should_log={should_log}"
    )

    if not should_log:
        logger.debug("Returning BaseLogger due to should_log=False")
        return BaseLogger()

    # Setup logging directory
    dump_dir = job_config.job.dump_folder
    base_log_dir = os.path.join(
        dump_dir, metrics_config.save_tb_folder, datetime.now().strftime("%Y%m%d-%H%M")
    )

    if not metrics_config.rank_0_only:
        base_log_dir = os.path.join(
            base_log_dir, f"rank_{torch.distributed.get_rank()}"
        )

    # Create loggers in priority order
    if metrics_config.enable_wandb:
        logger.debug("Attempting to create WandB logger")
        try:
            return WandBLogger(base_log_dir, tag)
        except Exception as e:
            if "No module named 'wandb'" in str(e):
                logger.error(
                    "Failed to create WandB logger: No module named 'wandb'. Please install it using 'pip install wandb'."
                )
            else:
                logger.error(f"Failed to create WandB logger: {e}")

    if metrics_config.enable_tensorboard:
        logger.debug("Creating TensorBoard logger")
        return TensorBoardLogger(base_log_dir, tag)

    logger.debug("No loggers enabled, returning BaseLogger")
    return BaseLogger()
