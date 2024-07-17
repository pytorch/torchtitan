# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [Note] Getting the 'float8_experimental' package:
# This script requires the 'float8_experimental' package to function correctly.
# Please ensure you have this package installed from the appropriate repository.
# You can obtain it from https://github.com/pytorch-labs/float8_experimental.
# Either clone and run `pip install .` or run `pip install git+https://github.com/pytorch-labs/float8_experimental.git`

# Note: Performance
# Float8 experimental is intended to be ran under `torch.compile`` for competitive performance
import contextlib
import functools
from typing import Optional

import torch
import torch.nn as nn
from torch._logging import warning_once

from torchtitan.config_manager import JobConfig
from torchtitan.logging_utils import logger


@contextlib.contextmanager
def set_enable_fsdp_float8_all_gather(enable_fsdp_fp8_all_gather: bool):
    import float8_experimental.config as config

    prev = config.enable_fsdp_fp8_all_gather
    torch.distributed.barrier()
    config.enable_fsdp_fp8_all_gather = enable_fsdp_fp8_all_gather
    try:
        yield
    finally:
        torch.distributed.barrier()
        config.enable_fsdp_fp8_all_gather = prev


@functools.lru_cache(None)
def is_sm90_or_later():
    # Float8 is only supported on H100+ GPUs
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


def maybe_build_fp8_linear(
    model: nn.Module, job_config: JobConfig, dp_enabled: Optional[bool] = False
):
    """
    This function converts the linear layers to `Float8Linear`. Note that today,
    only dynamic tensor scaling (the default) is supported.

    This will mutate the model inplace.
    """
    enable_float8_linear = job_config.training.enable_float8_linear
    if not enable_float8_linear:
        return
    if not is_sm90_or_later():
        warning_once(
            logger,
            "Failed to swap to Float8Linear because SM90 or later is not available",
        )
        return
    try:
        from float8_experimental.float8_linear import TensorScalingType
        from float8_experimental.float8_linear_utils import (
            swap_linear_with_float8_linear,
        )

        # Mutates the model inplace replacing instances of torch.nn.Linear with Float8Linear
        enable_fsdp_float8_all_gather = (
            job_config.training.enable_fsdp_float8_all_gather and dp_enabled
        )
        with set_enable_fsdp_float8_all_gather(enable_fsdp_float8_all_gather):
            swap_linear_with_float8_linear(
                model, scaling_type_w=TensorScalingType.DYNAMIC
            )
        logger.info(
            f"Swapped to Float8Linear layers with {enable_fsdp_float8_all_gather=}"
        )
    except ImportError as exc:
        raise ImportError(
            "float8_experimental is not installed. Please install it to use fp8 linear layers."
        ) from exc


def maybe_precompute_fp8_dynamic_scale_for_fsdp(
    model: nn.Module, job_config: JobConfig
):
    if not (
        job_config.training.enable_float8_linear
        and job_config.training.enable_fsdp_float8_all_gather
        and job_config.training.precompute_float8_dynamic_scale_for_fsdp
    ):
        return
    if not is_sm90_or_later():
        warning_once(
            logger,
            "Skipped precomputing fp8 scales because SM90 or later is not available",
        )
        return
    from float8_experimental.fsdp_utils import precompute_float8_dynamic_scale_for_fsdp

    precompute_float8_dynamic_scale_for_fsdp(model)
