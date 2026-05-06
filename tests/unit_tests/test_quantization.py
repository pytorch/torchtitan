# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from torchtitan.components.quantization import Float8BlockwiseLinear, Float8Linear
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import ConfigManager
from torchtitan.models.common.linear import Linear
from torchtitan.tools.utils import has_cuda_capability


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


@pytest.mark.parametrize(
    ("config_name", "local_batch_size"),
    [
        ("deepseek_v3_debugmodel_fp8_blockwise_bs1", 1),
        ("deepseek_v3_debugmodel_fp8_blockwise_bs4", 4),
        ("deepseek_v3_debugmodel_fp8_blockwise_bs8", 8),
    ],
)
def test_deepseek_v3_blockwise_float8_applied_by_model_registry(
    config_name,
    local_batch_size,
):
    pytest.importorskip("torchao.prototype.blockwise_fp8_training.linear")
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
            config_name,
        ]
    )
    model_config = config.model_spec.model
    assert has_quantization(model_config)
    assert config.training.local_batch_size == local_batch_size

    converted = [
        lc
        for _fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if isinstance(lc, Float8BlockwiseLinear.Config)
    ]
    assert len(converted) > 0
    assert all(lc.in_features % 128 == 0 for lc in converted)
    assert all(lc.out_features % 128 == 0 for lc in converted)
    assert all(not lc.bias for lc in converted)
    lm_heads = [
        lc
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if fqn == "lm_head"
    ]
    assert len(lm_heads) == 1
    assert not isinstance(lm_heads[0], Float8BlockwiseLinear.Config)

    with torch.device("meta"):
        layer = converted[0].build()
    assert isinstance(layer, Float8BlockwiseLinear)
