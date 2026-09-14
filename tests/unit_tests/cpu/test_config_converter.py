# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config.transform import (
    Float8LinearConverter,
    LMHeadCastConverter,
    LoRAConverter,
    validate_converter_order,
)


def test_validate_converter_order():
    """Quantization and lm-head casting cannot mix and must precede LoRA."""
    float8 = Float8LinearConverter.Config(emulate=True)
    lm_head_cast = LMHeadCastConverter.Config()
    lora = LoRAConverter.Config(rank=8, alpha=16.0)

    validate_converter_order([float8, lora])
    validate_converter_order([lm_head_cast, lora])

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_converter_order([lm_head_cast, float8])
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_converter_order([float8, lm_head_cast])
    with pytest.raises(ValueError, match="before LoRAConverter"):
        validate_converter_order([lora, float8])
    with pytest.raises(ValueError, match="before LoRAConverter"):
        validate_converter_order([lora, lm_head_cast])
