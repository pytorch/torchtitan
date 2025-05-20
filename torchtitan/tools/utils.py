# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch._utils import _get_available_device_type, _get_device_module


def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()
curr_device = f"{device_type}:0"
device_name = device_module.get_device_name(curr_device)

from torchtitan.tools.logging import logger


def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq=1000):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        gc.disable()
        self.collect("Initial GC collection.")

    def run(self, step_count):
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Peforming periodical GC collection.")

    @staticmethod
    def collect(reason: str):
        begin = time.monotonic()
        gc.collect(1)
        logger.info("[GC] %s %.2f seconds.", reason, time.monotonic() - begin)


def get_peak_flops(device_name: str) -> int:
    """returns peak flops for BF16 (non-sparse) dtype for:

    NVIDIA: A100, H100, H200, B200
    AMD: MI250, MI300X, MI325X
    Intel PVC GPU
    """

    # Dictionary mapping device names to their BF16 peak flops and source URL for where the data was obtained
    device_flops = {
        "A100": (312e12, "https://www.nvidia.com/en-us/data-center/a100/"),
        "H100 NVL": (835e12, "https://www.nvidia.com/en-us/data-center/h100/"),
        "H100 PCIe": (756e12, "https://www.nvidia.com/en-us/data-center/h100/"),
        "H100": (989e12, "https://www.nvidia.com/en-us/data-center/h100/"),
        "H200": (989e12, "https://www.nvidia.com/en-us/data-center/h200/"),
        "B200": (
            4.5e15,
            "https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703",
        ),
        "MI300X": (
            1300e12,
            "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html",
        ),
        "MI325X": (
            1300e12,
            "https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html",
        ),
        "MI250X": (
            191.5e12,
            "https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html",
        ),
    }

    # Attempt to determine the device name using lspci
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
        logger.info(f"Device name from lspci: {device_name}")
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")

    # Check for Intel PVC
    if "Data Center GPU Max 1550" in device_name:
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Return the peak flops for the known device or log a warning and assume A100
    for key, (flops, _) in device_flops.items():
        if key in device_name:
            return flops

    logger.warning(f"Peak flops undefined for: {device_name}, falling back to A100")
    return device_flops["A100"][0]


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""


def check_if_feature_in_pytorch(
    feature_name: str,
    pull_request: str,
    min_nightly_version: Optional[str] = None,
) -> None:
    if "git" in torch.__version__:  # pytorch is built from source
        # notify users to check if the pull request is included in their pytorch
        logger.warning(
            "detected that the pytorch is built from source. Please make sure the PR "
            f"({pull_request_link}) is included in pytorch for correct {feature_name}."
        )
    elif min_nightly_version is not None and torch.__version__ < min_nightly_version:
        logger.warning(
            f"detected that the pytorch version {torch.__version__} is older than "
            f"{min_nightly_version}. Please upgrade a newer version to include the "
            f"change in ({pull_request_link}) for correct {feature_name}."
        )
