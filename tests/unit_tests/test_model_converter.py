# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchtitan.config import ConfigManager
from torchtitan.models.common.linear import Linear


def test_no_float8_by_default():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    model_config = config.model_spec.model
    # No _convert_fn set on Linear.Config instances
    for _fqn, lc in model_config.walk(Linear.Config):
        assert lc._convert_fn is None


def test_float8_applied_by_model_registry():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_float8"]
    )
    model_config = config.model_spec.model
    # _convert_fn should be set on Linear.Config instances (dims divisible by 16)
    converted = [
        fqn
        for fqn, lc in model_config.walk(Linear.Config)
        if lc._convert_fn is not None
    ]
    assert len(converted) > 0
