# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved

import torch
import torch.nn as nn

_gb_in_bytes = 1024 * 1024 * 1024
_mb_in_bytes = 1024 * 1024

def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / _gb_in_bytes
    metric_num = round(metric_num, ndigits=precision)
    return metric_num

def convert_to_gpu_pct(value, total_gpu_memory):
        return round(100 * (value / total_gpu_memory), 2)


class GPU_Memory_Monitor:
    """
    Class to monitor GPU memory usage
    """
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device) # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(self.device).total_memory
        self.device_capacity_gb = format_to_gb(self.device_capacity)
        self.num_retries = 0
        self.num_ooms = 0
        self.peak_active_memory=0
        self.peak_allocated_memory=0
        self.peak_reserved_memory=0
        self.curr_reserved_memory = 0

        # current stats
        self.device_memory_usage = torch.cuda.memory_allocated(self.device)
        self.device_memory_usage_gb = format_to_gb(self.device_memory_usage)
        self.device_memory_utilization = convert_to_gpu_pct(self.device_memory_usage, self.device_capacity)
        self.device_memory_utilization_str = f"{self.device_memory_utilization:.2f}%"


        # reset stats, clear cache
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def get_pct_memory(self, memory_num):
        pct_memory = memory_num / self.device_capacity
        pct_memory = round(100 * (pct_memory), 2)
        return pct_memory

    def get_gb_memory(self, memory_num):
        gb_memory = memory_num / _gb_in_bytes
        gb_memory = round(gb_memory, 2)
        return gb_memory


    def get_current_stats(self):
        self.device_memory_usage = torch.cuda.memory_allocated(self.device)
        self.device_memory_usage_gb = format_to_gb(self.device_memory_usage)
        self.device_memory_utilization = convert_to_gpu_pct(self.device_memory_usage, self.device_capacity)
        self.device_memory_utilization_str = f"{self.device_memory_utilization:.2f}%"

    def start_monitoring(self):
        """ reset all monitoring stats """
        self.reset_peak_stats()


    def stop_monitoring(self):
        """ capture peak memory stats """
        cuda_info = torch.cuda.memory_stats()

        self.peak_active_memory = cuda_info.get("active_bytes.all.peak", 0)
        self.peak_allocated_memory = torch.cuda.max_memory_allocated(self.device)
        self.peak_reserved_memory = torch.cuda.max_memory_reserved(self.device)

        self.num_retries = cuda_info.get("num_alloc_retries", 0)
        self.num_ooms = cuda_info.get("num_ooms", 0)

    def get_peak_stats_str(self)-> str:
        """ return string to show overall peak memory stats, warn user re: cudacache retries """
        display_str = ""
        display_str += f"{self.device_name} ({self.device_index}): {self.device_capacity_gb} GB capacity. Peak memory usage:\n"
        peak_active_gb = self.get_gb_memory(self.peak_active_memory)
        peak_allocated_gb = self.get_gb_memory(self.peak_allocated_memory)
        peak_reserved_gb = self.get_gb_memory(self.peak_reserved_memory)

        peak_active_pct = self.get_pct_memory(self.peak_active_memory)
        peak_allocated_pct = self.get_pct_memory(self.peak_allocated_memory)
        peak_reserved_pct = self.get_pct_memory(self.peak_reserved_memory)

        # print stats
        display_str+= f"Peak Reserved Memory: {peak_reserved_gb:.2f} GB ({peak_reserved_pct:.2f}%)\n"
        display_str += f"Peak Allocated Memory: {peak_allocated_gb:.2f} GB ({peak_allocated_pct:.2f}%)\n"
        display_str += f"Peak Active Memory: {peak_active_gb:.2f} GB ({peak_active_pct:.2f}%)\n"

        display_str+=f"num retries: {self.num_retries}, num ooms: {self.num_ooms}"
        if self.num_retries > 0:
            display_str += f"\nWARNING: {self.num_retries} retries -- recommend lowering batch size for max performance\n"
        return display_str

    def reset_peak_stats(self):
        """ reset peak memory stats """
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        self.num_retries = 0
        self.num_ooms = 0
        self.active_peak_memory_utilization_str=""
        self.peak_memory_utilization_str=""
        self.peak_reserved_memory_utilization_str=""

    def __str__(self):
        self.get_current_stats()
        return f"{self.device_name} ({self.device_index}): {self.device_capacity_gb} GB capacity, {self.device_memory_usage_gb} GB used, {self.device_memory_utilization_str} used"


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
