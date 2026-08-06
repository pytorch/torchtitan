# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

from torchtitan.models.llama3 import model_registry as llama3_model_registry
from torchtitan.models.qwen3 import model_registry as qwen3_model_registry


def test_decoder_config_first_layer_component_properties() -> None:
    config = llama3_model_registry("debugmodel").model
    first_layer = config.layers[0]

    assert config.first_attention is first_layer.attention
    assert config.first_inner_attention is first_layer.attention.inner_attention
    assert config.first_feed_forward is first_layer.feed_forward
    assert config.first_moe is None

    replacement = dataclasses.replace(config, layers=[])
    assert replacement.first_attention is None
    assert replacement.first_inner_attention is None
    assert replacement.first_feed_forward is None
    assert replacement.first_moe is None


def test_decoder_config_finds_first_moe() -> None:
    config = qwen3_model_registry("debugmodel_moe").model
    expected = next(layer.moe for layer in config.layers if layer.moe is not None)

    assert config.first_moe is expected
