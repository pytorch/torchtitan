# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.models.common.linear import GroupedLinear
from torchtitan.quantization.nvfp4.experts import _get_nvfp4_grouped_linear_cls


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="NVFP4 requires SM100 or later",
    ),
]


def test_nvfp4_grouped_w13_and_w2_forward_backward():
    quantized_cls = _get_nvfp4_grouped_linear_cls(GroupedLinear)
    w13 = (
        quantized_cls.Config(
            group_size=2,
            in_features=128,
            out_features=128,
            num_linears=2,
            param_init={"weight": torch.nn.init.normal_},
        )
        .build()
        .cuda()
        .bfloat16()
    )
    w2 = (
        quantized_cls.Config(
            group_size=2,
            in_features=128,
            out_features=128,
            param_init={"weight": torch.nn.init.normal_},
        )
        .build()
        .cuda()
        .bfloat16()
    )
    w13.init_states()
    w2.init_states()
    offsets_E = torch.tensor([128, 256], device="cuda", dtype=torch.int32)
    input_RD = torch.randn(
        256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    gate_up_R2F = w13(input_RD, offsets_E)
    hidden_RF = gate_up_R2F[:, 0] * gate_up_R2F[:, 1]
    output_RD = w2(hidden_RF, offsets_E)
    output_RD.sum().backward()

    assert gate_up_R2F.shape == (256, 2, 128)
    assert output_RD.shape == (256, 128)
    for tensor in (output_RD, input_RD.grad, w13.weight.grad, w2.weight.grad):
        assert torch.isfinite(tensor).all()
