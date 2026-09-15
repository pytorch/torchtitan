# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.utils import enable_fp32_matmul_emulation_with_bf16x9
from torchtitan.models.common.linear import RouterGateLinear


_accelerator = torch.accelerator.current_accelerator()
_device_type = _accelerator.type if _accelerator is not None else None

# BFX9 is a CUDA-only FP32 matmul emulation mode. On other accelerators (e.g.
# XPU) ``enable_fp32_matmul_emulation_with_bf16x9`` is a no-op, and the test
# only covers the RouterGateLinear compile and dtype contract.
_IS_CUDA = _device_type == "cuda"
_BFX9_SUPPORTED = (
    _IS_CUDA
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (10, 0)
)

pytestmark = [
    pytest.mark.skipif(
        not torch.accelerator.is_available(), reason="an accelerator is required"
    ),
    pytest.mark.skipif(
        _IS_CUDA and not _BFX9_SUPPORTED,
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
    previous_precision = (
        torch.backends.cuda.matmul.fp32_precision if _BFX9_SUPPORTED else None
    )
    layer = RouterGateLinear.Config(
        in_features=128,
        out_features=16,
        bias=True,
    ).build()
    layer = layer.to(device=_accelerator, dtype=weight_dtype)
    compiled = torch.compile(layer, fullgraph=True)
    input_TD = torch.randn(
        32, 128, device=_accelerator, dtype=input_dtype, requires_grad=True
    )

    try:
        enable_fp32_matmul_emulation_with_bf16x9()
        if _BFX9_SUPPORTED:
            assert torch.backends.cuda.matmul.fp32_precision == "bfx9"
        output_TE = compiled(input_TD)
        output_TE.sum().backward()
    finally:
        if _BFX9_SUPPORTED:
            torch.backends.cuda.matmul.fp32_precision = previous_precision

    assert output_TE.dtype is torch.float32
    assert input_TD.grad is not None
    assert input_TD.grad.dtype is input_dtype
    assert layer.weight.grad is not None
    assert layer.weight.grad.dtype is weight_dtype
