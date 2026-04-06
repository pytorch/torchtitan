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


# Mapping from quantization type to the pad_multiple needed for grouped GEMMs.
# FP8: 16 byte alignment / 1 byte per elem = 16 elements.
# MXFP8: scaling block size is (1 x 32), so contracting dim must be divisible by 32.
PAD_MULTIPLE_MAP: dict[str, int] = {
    "float8": 16,
    "mxfp8": 32,
}


def find_pad_multiple(converters: list) -> int | None:
    """Return pad_multiple needed for quantized grouped GEMMs, or None.

    Inspects the list of converter configs to determine if any require
    token group padding (Float8GroupedMMConverter or MXFP8Converter).
    """
    from torchtitan.components.quantization.float8 import Float8GroupedMMConverter
    from torchtitan.components.quantization.mx import MXFP8Converter

    for c in converters:
        if isinstance(c, Float8GroupedMMConverter.Config):
            return PAD_MULTIPLE_MAP["float8"]
        if isinstance(c, MXFP8Converter.Config):
            return PAD_MULTIPLE_MAP["mxfp8"]
    return None
