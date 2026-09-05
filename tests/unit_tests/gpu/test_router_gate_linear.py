# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.utils import enable_fp32_matmul_emulation_with_bf16x9
from torchtitan.models.common.linear import RouterGateLinear


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available()
        and (
            torch.version.hip is not None
            or torch.cuda.get_device_capability() < (10, 0)
        ),
        reason="BFX9 requires NVIDIA compute capability 10.0 or later",
    ),
]


@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_router_gate_linear_compiles_with_global_bfx9(input_dtype, weight_dtype):
    previous_precision = torch.backends.cuda.matmul.fp32_precision
    layer = RouterGateLinear.Config(
        in_features=128,
        out_features=16,
        bias=True,
    ).build()
    layer = layer.to(device="cuda", dtype=weight_dtype)
    compiled = torch.compile(layer, fullgraph=True)
    input_TD = torch.randn(
        32, 128, device="cuda", dtype=input_dtype, requires_grad=True
    )

    try:
        enable_fp32_matmul_emulation_with_bf16x9()
        assert torch.backends.cuda.matmul.fp32_precision == "bfx9"
        output_TE = compiled(input_TD)
        output_TE.sum().backward()
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous_precision

    assert output_TE.dtype is torch.float32
    assert input_TD.grad is not None
    assert input_TD.grad.dtype is input_dtype
    assert layer.weight.grad is not None
    assert layer.weight.grad.dtype is weight_dtype
