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
        self.tag = tag
        self.writer: Optional[SummaryWriter] = None
        self.use_wandb = enable_wandb and WANDB_AVAILABLE

        if enable_tb:
            self.writer = SummaryWriter(log_dir, max_queue=1000)

            # If wandb is enabled, set up tensorboard sync
            if self.use_wandb:
                wandb.tensorboard.patch(root_logdir=os.path.dirname(log_dir))

        # Initialize wandb if enabled and not already initialized
        if self.use_wandb and wandb.run is None:
            project_name = wandb_config.get("project", "torchtitan")
            wandb.init(
                project=project_name,
                config=wandb_config,
                sync_tensorboard=enable_tb,
                dir=log_dir,
            )

    def log(self, metrics: Dict[str, Any], step: int):
        if self.writer is not None:
            for k, v in metrics.items():
                tag = k if self.tag is None else f"{self.tag}/{k}"
                self.writer.add_scalar(tag, v, step)

        if self.use_wandb:
            # Transform metrics to include tag if present
            wandb_metrics = {}
            for k, v in metrics.items():
                tag = k if self.tag is None else f"{self.tag}/{k}"
                wandb_metrics[tag] = v
            wandb_metrics["step"] = step
            wandb.log(wandb_metrics)

    def log_memory_stats(self, memory_stats: DeviceMemStats, step: int):
        """Log device memory statistics"""
        metrics = {
            "memory/max_active_GiB": memory_stats.max_active_gib,
            "memory/max_active_pct": memory_stats.max_active_pct,
            "memory/max_reserved_GiB": memory_stats.max_reserved_gib,
            "memory/max_reserved_pct": memory_stats.max_reserved_pct,
            "memory/num_alloc_retries": memory_stats.num_alloc_retries,
            "memory/num_ooms": memory_stats.num_ooms,
        }
        self.log(metrics, step)

    def close(self):
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
    Builds a metric logger that can log to both TensorBoard and W&B (if enabled).
    W&B support is optional and controlled via the metrics config.

    Args:
        job_config: Configuration object containing metrics settings
        parallel_dims: Parallel dimensions configuration
        tag: Optional tag to prefix all metrics

    Returns:
        MetricLogger instance configured based on the provided settings
    """
    dump_dir = job_config.job.dump_folder
    tb_config = job_config.metrics
    save_tb_folder = tb_config.save_tb_folder
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(dump_dir, save_tb_folder, datetime_str)

    enable_tb = tb_config.enable_tensorboard
    enable_wandb = getattr(tb_config, "enable_wandb", False)
    wandb_config = getattr(tb_config, "wandb_config", None)

    if enable_tb:
        logger.info(
            f"Metrics logging active. Tensorboard logs will be saved at {log_dir}"
        )
        if tb_config.rank_0_only:
            metrics_rank = _get_metrics_rank(parallel_dims)
            enable_tb = torch.distributed.get_rank() == metrics_rank
            enable_wandb = enable_wandb and (
                torch.distributed.get_rank() == metrics_rank
            )
        else:
            rank_str = f"rank_{torch.distributed.get_rank()}"
            log_dir = os.path.join(log_dir, rank_str)

    if enable_wandb and not WANDB_AVAILABLE:
        logger.warning(
            "W&B logging requested but wandb package is not installed. Continuing without W&B logging."
        )
        enable_wandb = False
    elif enable_wandb:
        logger.info("W&B logging enabled")

    return MetricLogger(log_dir, tag, enable_tb, enable_wandb, wandb_config)
