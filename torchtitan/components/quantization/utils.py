# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torchtitan.config import JobConfig


def module_filter_fn(mod: nn.Module, fqn: str, filter_fqns: list[str]) -> bool:
    """
    Filter function to determine which modules should be converted.
    For both Float8 and MXFP8, we only convert Linear modules
    with dimensions divisible by 16 and not matching any filtered FQNs.
    """
    if not isinstance(mod, nn.Linear):
        return False

    # All dims must be divisible by 16 due to float8 tensorcore hardware requirements.
    dims_multiples_of_16 = (
        mod.weight.shape[0] % 16 == 0 and mod.weight.shape[1] % 16 == 0
    )

    # If the fqn matches any filtered fqn, then we should not convert this module.
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)

    return dims_multiples_of_16 and not is_filtered_fqn


def validate_quantization_converters(job_config: JobConfig):
    """
    Validates that the job config uses the same quantization type for dense and MoE layers.
    """
    # TODO: Explore supporting applying different quantization methods to dense and MoE layers.
    existing_quantization_converter = None
    for converter in job_config.model.converters:
        if is_quantization_converter(converter):
            if existing_quantization_converter is None:
                existing_quantization_converter = converter
            else:
                assert quantization_type(converter) == quantization_type(
                    existing_quantization_converter
                ), (
                    "Cannot combine model converters with different quantization types: "
                    f"'{quantization_type(converter)}' and '{quantization_type(existing_quantization_converter)}'"
                )


def is_quantization_converter(converter: str):
    return "quantize" in converter


def quantization_type(converter: str):
    # quantization converter format:
    # `quantize.[dense|moe].[mx|float8]`
    return converter.split(".")[-1]
