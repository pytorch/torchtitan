# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import importlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch._utils import _get_available_device_type, _get_device_module

from torchtitan.tools.logging import logger


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


def get_num_params(model: nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )
    return num_params


def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    l, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


# hardcoded BF16 type peak flops for NVIDIA A100, H100, H200 GPU and AMD MI250, MI300X and AMD MI325X
def get_peak_flops(device_name: str) -> int:
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
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250.html
        return 362e12
    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


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


def import_module_from_path(path: str):
    path = os.path.expanduser(path)

    # 1. Check if path is an existing file or directory path.
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ImportError(f"Path '{path}' is not a directory.")
        init_file = os.path.join(path, "__init__.py")
        if os.path.isfile(init_file):
            return _import_module_from_init(path)

        raise ImportError(
            f"Directory '{path}' is not a Python package because it does not "
            "contain an __init__.py file."
        )

    # 2. If not a valid path, assume it's a dotted module name.
    return importlib.import_module(path)


def _import_module_from_init(path: str):
    init_file = os.path.join(path, "__init__.py")
    module_name = os.path.basename(path)
    spec = importlib.util.spec_from_file_location(module_name, init_file)
    if spec is None:
        raise ImportError(f"Could not create spec from '{init_file}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
