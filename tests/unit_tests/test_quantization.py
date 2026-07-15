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
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts


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
        ["--module", "llama3", "--config", "llama3_debugmodel_float8_emulate_lora"]
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


def test_quantized_grouped_experts():
    """Quantized GroupedExperts: _owner, subclass handling, extra config fields."""
    # Base case
    MXFP8GroupedExperts = _get_mxfp8_grouped_experts_cls(GroupedExperts)
    Float8GroupedExperts = _get_float8_grouped_experts_cls(GroupedExperts)

    assert MXFP8GroupedExperts.Config._owner is MXFP8GroupedExperts
    assert Float8GroupedExperts.Config._owner is Float8GroupedExperts

    # Subclass case (GptOssGroupedExperts has extra swiglu_limit field)
    mxfp8_cls = _get_mxfp8_grouped_experts_cls(GptOssGroupedExperts)
    float8_cls = _get_float8_grouped_experts_cls(GptOssGroupedExperts)

    assert mxfp8_cls.Config._owner is mxfp8_cls
    assert float8_cls.Config._owner is float8_cls
    assert issubclass(mxfp8_cls, GptOssGroupedExperts)
    assert issubclass(float8_cls, GptOssGroupedExperts)
    assert hasattr(mxfp8_cls.Config, "swiglu_limit")
    assert hasattr(float8_cls.Config, "swiglu_limit")


def test_fp8_blockwise_applied_by_model_registry():
    """Blockwise FP8 converters swap both linears and grouped experts."""
    pytest.importorskip("torchao.prototype.blockwise_fp8_training.linear")
    from torchtitan.components.quantization import Float8BlockwiseLinear
    from torchtitan.tools.utils import has_cuda_capability

    if Float8BlockwiseLinear is None:
        pytest.skip("torchao blockwise FP8 training linear is unavailable")
    if not has_cuda_capability(9, 0):
        pytest.skip("blockwise FP8 training requires SM90 or later")

    config_manager = ConfigManager()
    config = config_manager.parse_args(
        [
            "--module",
            "deepseek_v3",
            "--config",
            "deepseek_v3_debugmodel_fp8_blockwise",
        ]
    )
    model_config = config.model_spec.model
    assert has_quantization(model_config)

    converted = [
        (fqn, lc)
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if isinstance(lc, Float8BlockwiseLinear.Config)
    ]
    assert len(converted) > 0
    for _fqn, lc in converted:
        assert lc.in_features % 128 == 0
        assert lc.out_features % 128 == 0
        assert not lc.bias
    for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        if fqn == "lm_head" or "router.gate" in fqn:
            assert not isinstance(lc, Float8BlockwiseLinear.Config)

    experts_configs = [
        config
        for _fqn, config, _parent, _attr in model_config.traverse(GroupedExperts.Config)
    ]
    assert len(experts_configs) > 0
    assert all(
        getattr(config, "recipe_name", None) == "fp8_blockwise"
        for config in experts_configs
    )
