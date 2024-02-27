# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved

import os
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchtrain.config_manager import JobConfig

from torchtrain.logging_utils import rank0_log

# note that GiB (gibibyte) is 1024, vs GB is 1000
_gib_in_bytes = 1024 * 1024 * 1024
_mib_in_bytes = 1024 * 1024


def format_to_gib(item, precision=4):
    """quick function to format numbers to gibibyte and round to (default) 4 digit precision"""
    metric_num = item / _gib_in_bytes
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


def convert_to_gpu_pct(value, total_gpu_memory):
    return round(100 * (value / total_gpu_memory), 2)


# named tuple for passing memory stats (as % of device capacity) for Tensorboard logging
GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "allocated_curr",
        "allocated_peak",
        "reserved_curr",
        "reserved_peak",
        "active_curr",
        "active_peak",
        "num_retries",
    ],
)


class GPUMemoryMonitor:
    """
    Class to monitor GPU memory usage
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)  # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = format_to_gib(self.device_capacity)
        self.num_retries = 0
        self.num_ooms = 0
        self.peak_active_memory = 0
        self.peak_allocated_memory = 0
        self.peak_reserved_memory = 0
        self.curr_reserved_memory = 0

        self.device_reserved_memory_usage = 0
        self.device_reserved_memory_gib = 0
        self.device_reserved_memory_pct = 0

        self.device_active_memory_usage = 0
        self.device_active_memory_gib = 0
        self.device_active_memory_pct = 0

        # current stats
        self.device_alloc_memory_usage = torch.cuda.memory_allocated(self.device)
        self.device_alloc_memory_gib = format_to_gib(self.device_alloc_memory_usage)
        self.device_alloc_memory_pct = convert_to_gpu_pct(
            self.device_alloc_memory_usage, self.device_capacity
        )

        # reset stats, clear cache
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def get_pct_memory(self, memory_num):
        pct_memory = memory_num / self.device_capacity
        pct_memory = round(100 * (pct_memory), 2)
        return pct_memory

    def get_gib_memory(self, memory_num):
        gib_memory = memory_num / _gib_in_bytes
        gib_memory = round(gib_memory, 2)
        return gib_memory

    def get_current_stats(self, return_data: bool = False):
        """
        get the CudaCachingAllocator stats for the current device

        return_data: bool, if True, return the data as a named tuple
        """
        curr_mem = torch.cuda.memory_stats(self.device)

        self.device_alloc_memory_usage = curr_mem["allocated_bytes.all.current"]
        self.device_alloc_memory_gib = format_to_gib(self.device_alloc_memory_usage)
        self.device_alloc_memory_pct = convert_to_gpu_pct(
            self.device_alloc_memory_usage, self.device_capacity
        )

        self.device_reserved_memory_usage = curr_mem["reserved_bytes.all.current"]
        self.device_reserved_memory_gib = format_to_gib(
            self.device_reserved_memory_usage
        )
        self.device_reserved_memory_pct = convert_to_gpu_pct(
            self.device_reserved_memory_usage, self.device_capacity
        )

        self.device_active_memory_usage = curr_mem["active_bytes.all.current"]
        self.device_active_memory_gib = format_to_gib(self.device_active_memory_usage)
        self.device_active_memory_pct = convert_to_gpu_pct(
            self.device_active_memory_usage, self.device_capacity
        )

        display_str = ""
        display_str += f"Current Memory: {self.device_name} ({self.device_index}): Reserved: {self.device_reserved_memory_pct}%, "
        display_str += f"Alloc {self.device_alloc_memory_pct}%, Active: {self.device_active_memory_pct}%\n"

        self.get_peak_stats(curr_mem)

        peak_active_pct = self.get_pct_memory(self.peak_active_memory)
        peak_allocated_pct = self.get_pct_memory(self.peak_allocated_memory)
        peak_reserved_pct = self.get_pct_memory(self.peak_reserved_memory)
        display_str += f"Peak Memory: Reserved {peak_reserved_pct}%, Alloc {peak_allocated_pct}%, Active: {peak_active_pct}%\n"

        display_str += f"num retries: {self.num_retries}, num ooms: {self.num_ooms}"
        if self.num_retries > 0:
            display_str += f"\nWARNING: {self.num_retries} retries -- recommend lowering batch size for max performance\n"

        if not return_data:
            return display_str

        # return named tuple
        curr_mem_stats = GPUMemStats(
            self.device_alloc_memory_pct,
            peak_active_pct,
            self.device_reserved_memory_pct,
            peak_reserved_pct,
            self.device_active_memory_pct,
            peak_active_pct,
            self.num_retries,
        )
        return curr_mem_stats

    def start_monitoring(self):
        """reset all monitoring stats"""
        self.reset_peak_stats()

    def get_peak_stats(self, cuda_info=None):
        """capture current peak memory stats"""
        if not cuda_info:
            cuda_info = torch.cuda.memory_stats()

        self.peak_active_memory = cuda_info.get("active_bytes.all.peak", 0)
        self.peak_allocated_memory = cuda_info.get("allocated_bytes.all.peak", 0)
        self.peak_reserved_memory = cuda_info.get("reserved_bytes.all.peak", 0)

        self.num_retries = cuda_info.get("num_alloc_retries", 0)
        self.num_ooms = cuda_info.get("num_ooms", 0)

    def reset_peak_stats(self):
        """reset peak memory stats"""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        self.num_retries = 0
        self.num_ooms = 0
        self.active_peak_memory_utilization_str = ""
        self.peak_memory_utilization_str = ""
        self.peak_reserved_memory_utilization_str = ""

    def __str__(self):
        _ = self.get_current_stats()
        display_str = f"{self.device_name} ({self.device_index}): {self.device_capacity_gib} GiB capacity, "
        display_str += f"{self.device_alloc_memory_gib} GiB in-use, {self.device_alloc_memory_pct}% in-use"
        return f"{display_str}"


def get_num_params(model: nn.Module, only_trainable: bool = False) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    param_list = list(model.parameters())
    if only_trainable:
        param_list = [p for p in param_list if p.requires_grad]
    unique_params = {p.data_ptr(): p for p in param_list}.values()
    return sum(p.numel() for p in unique_params)


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


def build_metric_logger(config: JobConfig, tag: Optional[str] = None):
    dump_dir = config.job.dump_folder
    save_tb_folder = config.metrics.save_tb_folder
    # since we don't have run id yet, use current minute as identifier
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(dump_dir, save_tb_folder, datetime_str)

    enable_tb = config.metrics.enable_tensorboard
    if enable_tb:
        rank0_log(
            f"Metrics logging active. Tensorboard logs will be saved at {log_dir}."
        )

    rank_str = f"rank_{torch.distributed.get_rank()}"
    return MetricLogger(os.path.join(log_dir, rank_str), tag, enable_tb)
