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

import float8_experimental.config as config
import torch.nn as nn
from float8_experimental.float8_linear import TensorScalingType

from torchtitan.config_manager import JobConfig
from torchtitan.logging_utils import logger


@contextlib.contextmanager
def set_enable_fsdp_fp8_all_gather(enable_fsdp_fp8_all_gather: bool):
    prev = config.enable_fsdp_fp8_all_gather
    torch.distributed.barrier()
    config.enable_fsdp_fp8_all_gather = enable_fsdp_fp8_all_gather
    try:
        yield
    finally:
        torch.distributed.barrier()
        config.enable_fsdp_fp8_all_gather = prev


def build_fp8_linear(model: nn.Module, job_config: JobConfig):
    """
    This function converts the linear layers to `Float8Linear`. Note that today,
    only dynamic tensor scaling (the default) is supported.

    This will mutate the model inplace.
    """
    use_fp8_linear = job_config.training.fp8_linear
    enable_fsdp_fp8_all_gather = job_config.training.enable_fsdp_fp8_all_gather
    try:
        from float8_experimental.float8_linear_utils import (
            swap_linear_with_float8_linear,
        )
    except ImportError as exc:
        raise ImportError(
            "float8_experimental is not installed. Please install it to use fp8 linear layers."
        ) from exc
    if use_fp8_linear:
        # Mutates the model inplace replacing instances of torch.nn.Linear with Float8Linear
        with set_enable_fsdp_fp8_all_gather(enable_fsdp_fp8_all_gather):
            swap_linear_with_float8_linear(
                model, scaling_type_w=TensorScalingType.DYNAMIC
            )
        logger.info(
            f"Swapped to Float8Linear layers with {enable_fsdp_fp8_all_gather=}"
        )
