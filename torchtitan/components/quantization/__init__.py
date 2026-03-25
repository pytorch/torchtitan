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

from dataclasses import dataclass
from typing import ClassVar

from torchtitan.config import Configurable


class QuantizationConverter(Configurable):
    """Base class for quantization converters (FP8, MX, etc.).

    All quantization converter classes should inherit from this so they can be
    identified via isinstance checks.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        _quantization_type: ClassVar[str]


# Module level global constants
FP8_GROUP_ALIGNMENT_SIZE = 16
MXFP8_GROUP_ALIGNMENT_SIZE = 32
