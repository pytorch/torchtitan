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
from torchtitan.logging_utils import logger

# named tuple for passing GPU memory stats for logging
GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class GPUMemoryMonitor:
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)  # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = cuda_info["num_alloc_retries"]
        num_ooms = cuda_info["num_ooms"]

        if num_retries > 0:
            logger.warning(f"{num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        torch.cuda.reset_peak_memory_stats()


def build_gpu_memory_monitor():
    gpu_memory_monitor = GPUMemoryMonitor("cuda")
    logger.info(
        f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
        f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )

    return gpu_memory_monitor


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


def build_metric_logger(
    config: JobConfig, metrics_log_rank: int = 0, tag: Optional[str] = None
):
    """
    metrics_log_rank controls which rank acts as 'rank 0' for logging metrics.

    If 'tb_config.rank_0_only' is set, then `metrics_log_rank` will be used as the rank to log metrics.
    This is intended to allow logging from the 0th rank within the last pipeline stage group, in case pipeline
    parallelism is enabled, without forcing logging from all ranks to capture loss information when using pipeline
    parallelism.
    """
    dump_dir = config.job.dump_folder
    tb_config = config.metrics
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
            enable_tb = torch.distributed.get_rank() == metrics_log_rank
        else:
            rank_str = f"rank_{torch.distributed.get_rank()}"
            log_dir = os.path.join(log_dir, rank_str)

    return MetricLogger(log_dir, tag, enable_tb)
