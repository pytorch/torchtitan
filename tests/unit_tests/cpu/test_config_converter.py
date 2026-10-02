# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config.transform import (
    LMHeadCastConverter,
    MXFP8LinearConverter,
    validate_converter_compatibility,
)


def test_validate_converter_compatibility():
    """Quantization and lm-head casting cannot be combined."""
    mxfp8 = MXFP8LinearConverter.Config()
    lm_head_cast = LMHeadCastConverter.Config()

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_converter_compatibility([lm_head_cast, mxfp8])
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_converter_compatibility([mxfp8, lm_head_cast])
