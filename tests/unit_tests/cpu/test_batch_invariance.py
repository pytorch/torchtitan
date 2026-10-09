# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config.transform import BatchInvariantFlexConverter
from torchtitan.models.common.attention import FlexInnerAttention, MLAAttention
from torchtitan.models.deepseek_v3 import build_model_config as build_deepseek_config
from torchtitan.models.llama3 import build_model_config


def test_batch_invariant_flex_converter_pins_kernel_options():
    model = build_model_config("debugmodel", attn_backend="flex")

    converted = BatchInvariantFlexConverter.Config().build().convert(model)

    assert converted is model
    for layer in model.layers:
        assert layer.attention.inner_attention.kernel_options == {
            "BACKEND": "TRITON",
            "BLOCK_M": 16,
            "BLOCK_N": 16,
        }


def test_batch_invariant_flex_converter_pins_composed_mla_kernel_options():
    model = build_deepseek_config("debugmodel", attn_backend="flex", seq_len=128)

    converted = BatchInvariantFlexConverter.Config().build().convert(model)

    assert converted is model
    for layer in model.layers:
        mla_attention = layer.attention.mla_attention
        assert isinstance(mla_attention, MLAAttention.Config)
        assert isinstance(mla_attention.inner_attention, FlexInnerAttention.Config)
        assert mla_attention.inner_attention.kernel_options == {
            "BACKEND": "TRITON",
            "BLOCK_M": 16,
            "BLOCK_N": 16,
        }
