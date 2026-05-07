# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from torchtitan.components.quantization.qat import QATConverter


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
    """Each QAT scheme should replace nn.Linear with the correct fake-quantized class."""
    pytest.importorskip("torchao")

    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(64, 64)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(64, 64)),
            ]
        )
    )
    original_dtypes = {name: param.dtype for name, param in model.named_parameters()}

    config_kwargs = {"scheme": scheme}
    if group_size is not None:
        config_kwargs["group_size"] = group_size
    converter = QATConverter(QATConverter.Config(**config_kwargs))
    converter.convert(model)

    assert (
        type(model.fc1).__name__ == expected_linear_cls
    ), f"scheme={scheme}: expected {expected_linear_cls}, got {type(model.fc1).__name__}"
    assert (
        type(model.fc2).__name__ == expected_linear_cls
    ), f"scheme={scheme}: expected {expected_linear_cls}, got {type(model.fc2).__name__}"

    for name, param in model.named_parameters():
        assert (
            param.dtype == original_dtypes[name]
        ), f"'{name}' dtype changed from {original_dtypes[name]} to {param.dtype}"


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

    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(64, 64)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(64, 32)),
            ]
        )
    )
    converter = QATConverter(
        QATConverter.Config(scheme="intx_weight_only", group_size=64)
    )
    converter.convert(model)

    x = torch.randn(4, 64)
    out = model(x)
    assert out.shape == (4, 32)
