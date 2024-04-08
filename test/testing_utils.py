# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import pytest
import torch
import torch.distributed as dist
from torch import nn, Tensor


def perf_timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        output = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start
        print(elapsed_time)
        return output, elapsed_time

    return wrapper


def assert_expected(
    actual: Any,
    expected: Any,
    rtol: Optional[float] = 0,
    atol: Optional[float] = 1e-4,
    check_device=True,
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=check_device,
        msg=f"actual: {actual}, expected: {expected}",
    )


def set_rng_seed(seed):
    """Sets the seed for pytorch random number generators"""
    torch.manual_seed(seed)


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """
    message = f"Not enough GPUs to run the test: required {gpu_count}"
    return pytest.mark.skipif(torch.cuda.device_count() < gpu_count, reason=message)
