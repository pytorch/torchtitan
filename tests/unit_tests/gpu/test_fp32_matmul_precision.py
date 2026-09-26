# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests for ``set_fp32_matmul_precision`` on the MoE router gate.

The router gate is the model's only FP32 GEMM under BF16 mixed precision, so
these check that the knob reaches it and bound the accuracy it costs.
"""

import pytest
import torch
import torch.nn.functional as F

from torchtitan.distributed.utils import set_fp32_matmul_precision
from torchtitan.models.common.linear import RouterGateLinear


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available()
        and (
            torch.version.hip is not None or torch.cuda.get_device_capability() < (8, 0)
        ),
        reason="TF32 requires NVIDIA compute capability 8.0 or later",
    ),
]

_NUM_TOKENS, _DIM, _NUM_EXPERTS = 4096, 1024, 128


@pytest.fixture
def restore_fp32_precision():
    previous = torch.backends.cuda.matmul.fp32_precision
    yield
    torch.backends.cuda.matmul.fp32_precision = previous


def _router_backward(grad_output_TE: torch.Tensor):
    """Router gate gradients for a fixed BF16 layer and input."""
    torch.manual_seed(0)
    layer = (
        RouterGateLinear.Config(in_features=_DIM, out_features=_NUM_EXPERTS)
        .build()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    input_TD = torch.randn(
        _NUM_TOKENS, _DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    return (
        torch.autograd.grad(
            layer(input_TD), [input_TD, layer.weight], grad_outputs=grad_output_TE
        ),
        input_TD,
        layer.weight,
    )


def test_tf32_changes_router_gradients(restore_fp32_precision):
    """The knob reaches the router's FP32 backward GEMMs."""
    grad_output_TE = torch.randn(
        _NUM_TOKENS, _NUM_EXPERTS, device="cuda", dtype=torch.float32
    )

    set_fp32_matmul_precision("ieee")
    (ieee_input, ieee_weight), _, _ = _router_backward(grad_output_TE)
    set_fp32_matmul_precision("tf32")
    (tf32_input, tf32_weight), _, _ = _router_backward(grad_output_TE)

    # An all-ones grad_output would be exactly representable in TF32 and hide
    # the difference, hence the random upstream gradient above.
    assert not torch.equal(ieee_input, tf32_input)
    assert not torch.equal(ieee_weight, tf32_weight)


def test_tf32_error_stays_under_the_bf16_rounding_floor(restore_fp32_precision):
    """TF32 costs less accuracy than returning BF16 gradients already does."""
    grad_output_TE = torch.randn(
        _NUM_TOKENS, _NUM_EXPERTS, device="cuda", dtype=torch.float32
    )
    set_fp32_matmul_precision("tf32")
    (grad_input_TD, grad_weight_ED), input_TD, weight_ED = _router_backward(
        grad_output_TE
    )

    input_exact = input_TD.detach().double().requires_grad_()
    weight_exact = weight_ED.detach().double().requires_grad_()
    exact = torch.autograd.grad(
        F.linear(input_exact, weight_exact),
        [input_exact, weight_exact],
        grad_outputs=grad_output_TE.double(),
    )

    def relative_error(actual, exact):
        return ((actual.double() - exact).norm() / exact.norm()).item()

    # Gradients return in BF16 either way, so rounding the exact gradients to
    # BF16 is the floor. TF32's truncated mantissa adds only a little on top.
    for grad, grad_exact in zip((grad_input_TD, grad_weight_ED), exact):
        floor = relative_error(grad_exact.bfloat16(), grad_exact)
        assert relative_error(grad, grad_exact) < 1.05 * floor
