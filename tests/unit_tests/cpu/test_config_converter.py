# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config.transform import (
    Float8LinearConverter,
    LMHeadCastConverter,
    validate_converter_compatibility,
)


def test_validate_converter_compatibility():
    """Quantization and lm-head casting cannot be combined."""
    float8 = Float8LinearConverter.Config(emulate=True)
    lm_head_cast = LMHeadCastConverter.Config()

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_converter_compatibility([lm_head_cast, float8])
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_converter_compatibility([float8, lm_head_cast])
