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
# The quantization modules are intended to be ran under `torch.compile`` for competitive performance

# Module level global constants
FP8_GROUP_ALIGNMENT_SIZE = 16
MXFP8_GROUP_ALIGNMENT_SIZE = 32

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims

from torchtitan.protocols.model_converter import ModelConverter


class QuantizationConverter(ModelConverter):
    """
    Base class for quantization converters, which implements generic validation reusable across all quantization converters.
    """

    enabled: bool = False

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self._validate(job_config)

    @staticmethod
    def _validate(job_config: JobConfig):
        """
        Validates that the job config uses the same quantization type for dense and MoE layers.
        """
        # TODO: Explore supporting applying different quantization methods to dense and MoE layers.
        # quantization converter format:
        # `quantize.[linear | grouped_mm].[float8 | mx]`
        quantization_type = lambda converter: converter.split(".")[-1]
        existing_quantization_converter = None
        for converter in job_config.model.converters:
            if "quantize" in converter:
                if existing_quantization_converter is None:
                    existing_quantization_converter = converter
                else:
                    assert quantization_type(converter) == quantization_type(
                        existing_quantization_converter
                    ), (
                        "Cannot combine model converters with different quantization types: "
                        f"'{quantization_type(converter)}' and '{quantization_type(existing_quantization_converter)}'"
                    )


# Import to register quantization modules as ModelConverter
# (imports down here to avoid circular imports with QuantizationConverter)
import torchtitan.components.quantization.float8  # noqa: F401
import torchtitan.components.quantization.mx  # noqa: F401
