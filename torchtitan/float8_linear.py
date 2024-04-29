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

import torch.nn as nn

from torchtitan.config_manager import JobConfig
from torchtitan.logging_utils import logger


def build_fp8_linear(model: nn.Module, job_config: JobConfig):
    """
    This function converts the linear layers to one of the fp8 types:
    - Float8DynamicLinear: Dynamic quantization of the weights and the activations
    - [Not Yet Supported] Float8Linear: Uses a history of amaxs to quantize the weights and activations

    This will mutate the model inplace.
    """
    linear_type = job_config.training.fp8_linear.lower()
    try:
        from float8_experimental.float8_dynamic_linear import Float8DynamicLinear

        # from float8_experimental.float8_linear import Float8Linear
        from float8_experimental.float8_linear_utils import (
            swap_linear_with_float8_linear,
        )
    except ImportError as exc:
        raise ImportError(
            "float8_experimental is not installed. Please install it to use fp8 linear layers."
        ) from exc
    if linear_type:
        linear_type_map = {
            # "delayed": Float8Linear, # TODO: add "delayed" option back in when supported
            "dynamic": Float8DynamicLinear,
        }
        assert (
            linear_type in linear_type_map
        ), f"Invalid fp8 linear type: {linear_type}, supported types: {', '.join(linear_type_map.keys())}."
        float8_linear_type = linear_type_map[linear_type.lower()]

        # Mutates the model inplace replacing instances of torch.nn.Linear with float8_linear_type
        swap_linear_with_float8_linear(model, float8_linear_type)
        logger.info(f"Swapped to {linear_type} float8 linear layers")
