# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchtitan.models.common import linear as linear_module
from torchtitan.models.common.linear import FP32OutputLinear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("exact_grad_output_split", [False, True])
@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_fp32_output_linear_compiles(
    input_dtype, weight_dtype, exact_grad_output_split
):
    layer = FP32OutputLinear.Config(
        in_features=128,
        out_features=16,
        bias=True,
        exact_grad_output_split=exact_grad_output_split,
    ).build()
    layer = layer.to(device="cuda", dtype=weight_dtype)
    compiled = torch.compile(layer, fullgraph=True)
    input_TD = torch.randn(
        32, 128, device="cuda", dtype=input_dtype, requires_grad=True
    )

    output_TE = compiled(input_TD)
    output_TE.sum().backward()

    assert output_TE.dtype is torch.float32
    assert input_TD.grad is not None
    assert input_TD.grad.dtype is input_dtype
    assert layer.weight.grad is not None
    assert layer.weight.grad.dtype is weight_dtype


def _lm_head(in_features: int = 256, out_features: int = 1024) -> FP32OutputLinear:
    lm_head = FP32OutputLinear.Config(
        in_features=in_features, out_features=out_features
    ).build()
    torch.nn.init.normal_(lm_head.weight, std=0.02)
    return lm_head.to(device="cuda", dtype=torch.bfloat16)


def test_forward_matches_fp64_reference():
    lm_head = _lm_head()
    x = torch.randn(3, 5, 256, device="cuda", dtype=torch.bfloat16)
    reference = F.linear(x.double(), lm_head.weight.double())

    out = lm_head(x)

    assert out.dtype == torch.float32
    assert out.shape == reference.shape
    torch.testing.assert_close(out.double(), reference, rtol=1e-4, atol=1e-4)


def _relative_error(actual, exact):
    return ((actual.double() - exact).norm() / exact.norm()).item()


def _backward_errors_vs_bf16_floor(function, num_tokens, in_features, out_features):
    """Backward of ``function`` vs exact fp64 gradients, as multiples of the bf16 floor."""
    x = torch.randn(num_tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(out_features, in_features, device="cuda") * 0.02).bfloat16()
    grad_output = torch.randn(num_tokens, out_features, device="cuda")

    x_exact = x.double().requires_grad_()
    weight_exact = weight.double().requires_grad_()
    F.linear(x_exact, weight_exact).backward(grad_output.double())

    x_leaf = x.clone().requires_grad_()
    weight_leaf = weight.clone().requires_grad_()
    function(x_leaf, weight_leaf).backward(grad_output)

    # Gradients return in bf16, so rounding the exact gradients to bf16 is the floor.
    ratios = []
    for grad, exact in (
        (x_leaf.grad, x_exact.grad),
        (weight_leaf.grad, weight_exact.grad),
    ):
        assert grad.dtype == torch.bfloat16
        ratios.append(
            _relative_error(grad, exact) / _relative_error(exact.bfloat16(), exact)
        )
    return ratios


# out_features > num_tokens takes the LM-head layout, out_features < num_tokens the router one.
@pytest.mark.parametrize("exact_grad_output_split", [False, True])
@pytest.mark.parametrize(
    "num_tokens,in_features,out_features", [(64, 256, 1024), (512, 256, 16)]
)
def test_backward_error_stays_at_bf16_rounding_floor(
    num_tokens, in_features, out_features, exact_grad_output_split
):
    ratios = _backward_errors_vs_bf16_floor(
        lambda input, weight: linear_module._FP32OutputLinearFunction.apply(
            input, weight, exact_grad_output_split
        ),
        num_tokens,
        in_features,
        out_features,
    )
    assert max(ratios) < 1.05, ratios


def test_compiled_backward_keeps_lo_half():
    # torch.compile folds a bf16 round trip away inside fused kernels; if the split used one,
    # lo would compile to zero and the error would rise to that of a bf16 grad_output.
    # Compile a wrapper: compiling any ``Function.apply`` directly breaks later compiles of other
    # autograd Functions in the same process (test_qwen3_5_deltanet fails after it).
    def linear(input, weight):
        return linear_module._FP32OutputLinearFunction.apply(input, weight, False)

    ratios = _backward_errors_vs_bf16_floor(torch.compile(linear), 64, 256, 1024)
    assert max(ratios) < 1.05, ratios


@pytest.mark.parametrize("compile", [False, True])
# 1 token takes the LM-head layout, 4 tokens the router one.
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("exact_grad_output_split", [False, True])
def test_third_piece_keeps_what_two_pieces_drop(
    exact_grad_output_split, num_tokens, compile
):
    # (1 + 2^-8 + 2^-20) - (1 + 2^-8) = 2^-20: 2 pieces round the 2^-20 away, 3 keep it. Compiled,
    # this also catches a split that inductor folds (the third piece would compile to zero).
    grad_output = torch.tensor([[1 + 2**-8 + 2**-20, 1 + 2**-8]], device="cuda")
    grad_output = grad_output.repeat(num_tokens, 1)
    weight = torch.tensor([[1.0] * 8, [-1.0] * 8], device="cuda", dtype=torch.bfloat16)
    x = torch.zeros(
        num_tokens, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    def linear(input, weight):
        return linear_module._FP32OutputLinearFunction.apply(
            input, weight, exact_grad_output_split
        )

    (torch.compile(linear) if compile else linear)(x, weight).backward(grad_output)

    expected = 2**-20 if exact_grad_output_split else 0.0
    assert torch.equal(x.grad, torch.full_like(x.grad, expected))


def test_batch_invariant_mode_computes_in_fp32(monkeypatch):
    lm_head = _lm_head()
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    monkeypatch.setattr(linear_module, "is_in_batch_invariant_mode", lambda: True)

    out = lm_head(x)
    out.sum().backward()

    assert torch.equal(out, F.linear(x.detach().float(), lm_head.weight.float()))
    assert x.grad.dtype is torch.bfloat16


def test_backward_runs_under_autocast_with_fp32_params():
    # Autocast makes the fp32 fallback's output bf16; the backward must still run.
    layer = FP32OutputLinear.Config(in_features=64, out_features=32).build().cuda()
    x = torch.randn(16, 64, device="cuda", requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x)
    out.float().sum().backward()

    assert x.grad.dtype is torch.float32
    assert layer.weight.grad.dtype is torch.float32
