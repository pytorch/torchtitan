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
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} ({device_memory_monitor.device_index}) "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )

    return device_memory_monitor


class MetricLogger:
    def __init__(self, log_dir, tag, enable_tb):
        self.tag = tag
        self.writer: Optional[SummaryWriter] = None
        if enable_tb:
            self.writer = SummaryWriter(log_dir, max_queue=1000)

    def log(self, metrics: Dict[str, Any], step: int):
        if self.writer is not None:
            for k, v in metrics.items():
                tag = k if self.tag is None else f"{self.tag}/{k}"
                self.writer.add_scalar(tag, v, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


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
    parallel_dims is used to determine the rank to log metrics from if 'tb_config.rank_0_only=True'.
    In that case, `_get_metrics_rank` will be used to calculate which rank acts as 'rank 0'. This is
    intended to allow logging from the 0th rank within the last pipeline stage group, in case pipeline
    parallelism is enabled, without forcing logging from all ranks to capture loss information.
    """
    dump_dir = job_config.job.dump_folder
    tb_config = job_config.metrics
    save_tb_folder = tb_config.save_tb_folder
    # since we don't have run id, use current minute as the identifier
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(dump_dir, save_tb_folder, datetime_str)

    enable_tb = tb_config.enable_tensorboard
    if enable_tb:
        logger.info(
            f"Metrics logging active. Tensorboard logs will be saved at {log_dir}"
        )
        if tb_config.rank_0_only:
            enable_tb = torch.distributed.get_rank() == _get_metrics_rank(parallel_dims)
        else:
            rank_str = f"rank_{torch.distributed.get_rank()}"
            log_dir = os.path.join(log_dir, rank_str)

    return MetricLogger(log_dir, tag, enable_tb)
