# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchtitan.components.quantization import Float8LinearConfig, has_quantization
from torchtitan.config import ConfigManager
from torchtitan.models.common.linear import Linear


def test_no_float8_by_default():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    model_config = config.model_spec.model
    assert not has_quantization(model_config)
    # All Linear.Config instances should remain Linear.Config
    for _fqn, lc, _parent, _attr in model_config.walk(Linear.Config):
        assert not isinstance(lc, Float8LinearConfig)


def test_float8_applied_by_model_registry():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_float8"]
    )
    model_config = config.model_spec.model
    assert has_quantization(model_config)
    # Some Linear.Config instances should be swapped to Float8LinearConfig
    converted = [
        fqn
        for fqn, lc, _parent, _attr in model_config.walk(Linear.Config)
        if isinstance(lc, Float8LinearConfig)
    ]
    assert len(converted) > 0
