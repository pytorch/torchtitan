# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [Note] Getting the 'torchao' package:
# This script requires the 'torchao' package to function correctly.
# Please ensure you have this package installed from the appropriate repository.
# You can obtain it from https://github.com/pytorch/ao by following the
# installation instructions.

# Note: Performance
# Float8 experimental is intended to be ran under `torch.compile`` for competitive performance
import functools
from typing import Optional

import torch
import torch.nn as nn
from torch._logging import warning_once

from torchtitan.config_manager import JobConfig
from torchtitan.logging_utils import logger


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
        from torchao.float8 import (
            CastConfig,
            convert_to_float8_training,
            Float8LinearConfig,
            ScalingType,
        )

        # Mutates the model inplace replacing instances of torch.nn.Linear with Float8Linear
        enable_fsdp_float8_all_gather = (
            job_config.training.enable_fsdp_float8_all_gather and dp_enabled
        )
        scaling_type_input = ScalingType(job_config.training.float8_scaling_type_input)
        scaling_type_weight = ScalingType(
            job_config.training.float8_scaling_type_weight
        )
        scaling_type_grad_output = ScalingType(
            job_config.training.float8_scaling_type_grad_output
        )
        float8_config = Float8LinearConfig(
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            cast_config_input=CastConfig(scaling_type=scaling_type_input),
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
            cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
            enable_pre_and_post_forward=False,
        )
        convert_to_float8_training(
            model,
            config=float8_config,
            module_filter_fn=lambda mod, fqn: fqn != "output",
        )
        logger.info(
            f"Swapped to Float8Linear layers with {enable_fsdp_float8_all_gather=}"
        )
    except ImportError as exc:
        raise ImportError(
            "torchao is not installed. Please install it to use fp8 linear layers."
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
    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

    precompute_float8_dynamic_scale_for_fsdp(model)


_sync_float8_amax_and_scale_history = None


def maybe_sync_float8_amax_and_scale_history(model: nn.Module, job_config: JobConfig):
    if not (
        job_config.training.enable_float8_linear
        and (
            job_config.training.float8_scaling_type_input == "delayed"
            or job_config.training.float8_scaling_type_weight == "delayed"
            or job_config.training.float8_scaling_type_grad_output == "delayed"
        )
    ):
        return

    from torchao.float8 import sync_float8_amax_and_scale_history

    # TODO(future): see if precalculating the modules to sync over is going to
    # meaningfully help performance

    global _sync_float8_amax_and_scale_history
    if _sync_float8_amax_and_scale_history is None:
        if job_config.training.compile:
            _sync_float8_amax_and_scale_history = torch.compile(
                sync_float8_amax_and_scale_history
            )
        else:
            _sync_float8_amax_and_scale_history = sync_float8_amax_and_scale_history

    sync_float8_amax_and_scale_history(model)
