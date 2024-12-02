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
from torchtitan.logging import logger
from torchtitan.parallelisms import ParallelDims
from torchtitan.utils import device_module, device_type

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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

        max_active = device_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info["num_alloc_retries"]
        num_ooms = device_info["num_ooms"]

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper} OOM errors thrown.")

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
        f"{device_type.upper} capacity: {device_memory_monitor.device_name} ({device_memory_monitor.device_index}) "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


class MetricLogger:
    def __init__(self, log_dir, tag, enable_tb, enable_wandb=False, wandb_config=None):
        """Initialize metric logger with the configured backend."""
        self.tag = tag
        self.writer: Optional[SummaryWriter] = None
        self.use_wandb = False

        if enable_wandb and WANDB_AVAILABLE:
            self.use_wandb = True
            if wandb.run is None:
                project_name = (
                    wandb_config.get("project", "torchtitan")
                    if wandb_config
                    else "torchtitan"
                )
                wandb.init(
                    project=project_name,
                    config=wandb_config,
                    dir=log_dir,
                )
            logger.debug("WandB logging enabled")
        elif enable_tb:
            self.writer = SummaryWriter(log_dir, max_queue=1000)
            logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")
        else:
            logger.warning("Neither TensorBoard nor WandB logging is enabled.")

    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to the configured backend."""
        if self.use_wandb:
            wandb_metrics = {
                (k if self.tag is None else f"{self.tag}/{k}"): v
                for k, v in metrics.items()
            }
            wandb_metrics["step"] = step
            wandb.log(wandb_metrics)
        elif self.writer is not None:
            for k, v in metrics.items():
                tag = k if self.tag is None else f"{self.tag}/{k}"
                self.writer.add_scalar(tag, v, step)

    def close(self):
        """Clean up logging resources."""
        if self.writer is not None:
            self.writer.close()
        if self.use_wandb and wandb.run is not None:
            wandb.finish()


def _get_metrics_rank(parallel_dims: ParallelDims) -> int:
    """
    Returns global rank 0 in non-pipeline-parallel configs, and returns the global
    rank of the 0th rank in the last pipeline stage when pipeline parallelism is enabled.
    """
    if parallel_dims.pp_enabled:
        world_size = parallel_dims.world_size
        pp_size = parallel_dims.pp
        metrics_log_rank = (world_size // pp_size) * (pp_size - 1)
    else:
        metrics_log_rank = 0
    return metrics_log_rank


def build_metric_logger(
    job_config: JobConfig, parallel_dims: ParallelDims, tag: Optional[str] = None
):
    """
    Builds a metric logger based on the configuration.

    Args:
        job_config: Configuration object containing metrics settings
        parallel_dims: Parallel dimensions configuration
        tag: Optional tag to prefix all metrics

    Returns:
        MetricLogger instance configured based on the provided settings
    """
    dump_dir = job_config.job.dump_folder
    metrics_config = job_config.metrics
    log_dir = os.path.join(
        dump_dir, metrics_config.save_tb_folder, datetime.now().strftime("%Y%m%d-%H%M")
    )

    enable_tb = metrics_config.enable_tensorboard
    enable_wandb = metrics_config.enable_wandb
    wandb_config = (
        metrics_config.wandb_config if hasattr(metrics_config, "wandb_config") else None
    )

    if metrics_config.rank_0_only:
        metrics_rank = _get_metrics_rank(parallel_dims)
        is_metrics_rank = torch.distributed.get_rank() == metrics_rank
        enable_tb = enable_tb and is_metrics_rank
        enable_wandb = enable_wandb and is_metrics_rank
    else:
        rank_str = f"rank_{torch.distributed.get_rank()}"
        log_dir = os.path.join(log_dir, rank_str)

    return MetricLogger(log_dir, tag, enable_tb, enable_wandb, wandb_config)
