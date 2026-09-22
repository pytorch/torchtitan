# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config.transform import BatchInvariantFlexConverter
from torchtitan.models.llama3 import model_registry


def test_batch_invariant_flex_converter_pins_kernel_options():
    model = model_registry("debugmodel", attn_backend="flex")

    converted = BatchInvariantFlexConverter.Config().build().convert(model)

    assert converted is model
    for layer in model.layers:
        assert layer.attention.inner_attention.kernel_options == {
            "BACKEND": "TRITON",
            "BLOCK_M": 16,
            "BLOCK_N": 16,
        }
