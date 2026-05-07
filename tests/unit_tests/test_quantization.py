# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchtitan.components.quantization import Float8Linear
from torchtitan.components.quantization.float8 import _get_float8_grouped_experts_cls
from torchtitan.components.quantization.mx import _get_mxfp8_grouped_experts_cls
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import ConfigManager
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts


def test_no_float8_by_default():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    model_config = config.model_spec.model
    assert not has_quantization(model_config)
    # All Linear.Config instances should remain Linear.Config
    if Float8Linear is not None:
        for _fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
            assert not isinstance(lc, Float8Linear.Config)


def test_float8_applied_by_model_registry():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_float8_emulate"]
    )
    model_config = config.model_spec.model
    assert has_quantization(model_config)
    # Some Linear.Config instances should be swapped to Float8Linear
    converted = [
        fqn
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if isinstance(lc, Float8Linear.Config)
    ]
    assert len(converted) > 0


def test_quantized_grouped_experts_owner():
    """Dynamic quantized classes get correct _owner, not GroupedExperts."""
    mxfp8_cls = _get_mxfp8_grouped_experts_cls(GroupedExperts)
    float8_cls = _get_float8_grouped_experts_cls(GroupedExperts)
    assert mxfp8_cls.Config._owner is mxfp8_cls
    assert float8_cls.Config._owner is float8_cls
    assert issubclass(mxfp8_cls, GroupedExperts)
    assert issubclass(float8_cls, GroupedExperts)
    assert issubclass(mxfp8_cls.Config, GroupedExperts.Config)
    assert issubclass(float8_cls.Config, GroupedExperts.Config)
