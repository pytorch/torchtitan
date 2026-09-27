# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config.transform import (
    Float8LinearConverter,
    LMHeadFp32OutputConverter,
    validate_converter_compatibility,
)


def test_validate_converter_compatibility():
    """Quantization and an fp32-output lm_head cannot be combined."""
    float8 = Float8LinearConverter.Config(emulate=True)
    lm_head_fp32 = LMHeadFp32OutputConverter.Config()

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_converter_compatibility([lm_head_fp32, float8])
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_converter_compatibility([float8, lm_head_fp32])
