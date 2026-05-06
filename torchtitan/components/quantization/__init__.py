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
# The quantization modules are intended to be ran under `torch.compile` for competitive performance

from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear


@dataclass(kw_only=True, slots=True)
class QuantizedLinearConfig(Linear.Config):
    """Base config for all quantized Linear variants."""

    pass


class _QuantizedGroupedExpertsConfig:
    """Marker base for dynamically created quantized GroupedExperts configs."""

    pass


class QuantizationConverter(Configurable):
    """Base class for quantization converters.

    Subclasses define a nested Config and implement ``convert()``
    to transform the model config tree.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        model_compile_enabled: bool = False
        """Whether torch.compile is enabled for the model."""

    def convert(self, model_config) -> None:
        raise NotImplementedError


# Re-export all public symbols so callers can import from the package directly.
from .float8 import (  # noqa: F401, E402
    Float8GroupedExperts,
    Float8GroupedExpertsConverter,
    Float8Linear,
    Float8LinearConverter,
)
from .mx import (  # noqa: F401, E402
    MXFP8GroupedExperts,
    MXFP8GroupedExpertsConverter,
    MXFP8Linear,
    MXFP8LinearConverter,
)

__all__ = [
    "Float8GroupedExperts",
    "Float8GroupedExpertsConverter",
    "Float8Linear",
    "Float8LinearConverter",
    "MXFP8GroupedExperts",
    "MXFP8GroupedExpertsConverter",
    "MXFP8Linear",
    "MXFP8LinearConverter",
    "QuantizationConverter",
    "QuantizedLinearConfig",
    "_QuantizedGroupedExpertsConfig",
]
