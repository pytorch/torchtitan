# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.quantization.qat import (
    convert_qat_to_linear,
    QATConverter,
    QATLinearConfig,
)


@pytest.mark.parametrize(
    "scheme, group_size, expected_linear_cls",
    [
        ("int4_weight_only", 64, "FakeQuantizedLinear"),
        ("intx_weight_only", 64, "FakeQuantizedLinear"),
        ("int8_dynamic_act_intx_weight", 64, "FakeQuantizedLinear"),
        ("float8_dynamic_act_float8_weight", None, "FakeQuantizedLinear"),
        ("float8_dynamic_act_int4_weight", None, "FakeQuantizedLinear"),
        ("nvfp4", None, "NVFP4FakeQuantizedLinear"),
        ("mx", None, "MXFakeQuantizedLinear"),
    ],
)
def test_qat_all_schemes(scheme, group_size, expected_linear_cls):
    """Each QAT scheme should replace Linear with the correct fake-quantized class."""
    pytest.importorskip("torchao")

    config_kwargs = {"_scheme": scheme}
    if group_size is not None:
        config_kwargs["_group_size"] = group_size
    config = QATLinearConfig(in_features=64, out_features=64, **config_kwargs)
    mod = config.build()

    assert (
        type(mod).__name__ == expected_linear_cls
    ), f"scheme={scheme}: expected {expected_linear_cls}, got {type(mod).__name__}"


def test_qat_unknown_scheme_raises():
    """QATConverter should raise ValueError for unknown schemes."""
    with pytest.raises(ValueError, match="Unknown QAT scheme"):
        QATConverter(QATConverter.Config(scheme="not_a_real_scheme"))


def test_qat_group_size_warning_for_unsupported_scheme(caplog):
    """QATConverter should warn when group_size is set for a scheme that ignores it."""
    pytest.importorskip("torchao")
    import logging

    with caplog.at_level(logging.WARNING):
        QATConverter(
            QATConverter.Config(
                scheme="float8_dynamic_act_float8_weight", group_size=64
            )
        )
    assert "does not use group_size" in caplog.text


def test_qat_forward():
    """QAT model forward should produce correct output shape."""
    pytest.importorskip("torchao")

    config = QATLinearConfig(
        in_features=64, out_features=32, _scheme="intx_weight_only", _group_size=64
    )
    mod = config.build()

    x = torch.randn(4, 64)
    out = mod(x)
    assert out.shape == (4, 32)


def test_qat_converter_config_tree():
    """QATConverter should swap Linear.Config -> QATLinear.Config in config tree."""
    pytest.importorskip("torchao")
    from torchtitan.models.llama3 import model_registry

    spec = model_registry(
        "debugmodel",
        quantization=[
            QATConverter.Config(scheme="int4_weight_only", group_size=64),
        ],
    )

    with torch.device("meta"):
        model = spec.model.build()

    fake_quant_count = 0
    for name, mod in model.named_modules():
        if "FakeQuantized" in type(mod).__name__:
            fake_quant_count += 1

    assert fake_quant_count > 0, "No FakeQuantizedLinear modules found after QAT"


def test_convert_qat_to_linear():
    """convert_qat_to_linear should strip fake quant back to nn.Linear."""
    pytest.importorskip("torchao")
    import torch.nn as nn

    config = QATLinearConfig(
        in_features=64, out_features=64, _scheme="int4_weight_only", _group_size=64
    )
    mod = config.build()
    assert "FakeQuantized" in type(mod).__name__

    parent = nn.Sequential(mod)
    convert_qat_to_linear(parent)

    assert type(parent[0]) is nn.Linear
    assert parent[0].weight.shape == (64, 64)
