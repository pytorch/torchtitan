# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchtitan.models.common.activation import ClampedSwiGLU


@pytest.mark.parametrize(
    ("alpha", "offset"),
    [(1.702, 1.0), (1.0, 0.0)],
    ids=["gpt_oss", "deepseek_v4"],
)
def test_clamped_swiglu_matches_formula_and_gradients(alpha, offset):
    gate = torch.tensor(
        [[-20.0, -2.0, 0.0, 2.0, 20.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    up = torch.tensor(
        [[20.0, -20.0, 1.0, -2.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_gate = gate.detach().clone().requires_grad_()
    reference_up = up.detach().clone().requires_grad_()
    clamped_gate = reference_gate.clamp(max=2.0)
    clamped_up = reference_up.clamp(min=-2.0, max=2.0)
    expected = (
        clamped_gate * torch.sigmoid(alpha * clamped_gate) * (clamped_up + offset)
    )
    actual = ClampedSwiGLU.Config(
        swiglu_alpha=alpha, swiglu_limit=2.0, swiglu_offset=offset
    ).build()(torch.stack((gate, up), dim=-2))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    grad_output = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0]])
    actual_grads = torch.autograd.grad(actual, (gate, up), grad_output)
    expected_grads = torch.autograd.grad(
        expected, (reference_gate, reference_up), grad_output
    )
    torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_deepseek_v4_clamped_swiglu_preserves_silu_forward_and_backward(dtype):
    gate_up = torch.tensor(
        [[[-20.0, 3.0, 25.0], [30.0, -40.0, 2.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    reference_gate_up = gate_up.detach().clone().requires_grad_()
    gate, up = reference_gate_up.unbind(-2)
    expected = F.silu(gate.clamp(max=10.0)) * up.clamp(min=-10.0, max=10.0)
    actual = ClampedSwiGLU.Config(
        swiglu_alpha=1.0, swiglu_limit=10.0, swiglu_offset=0.0
    ).build()(gate_up)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad_output = torch.tensor([[1.0, -2.0, 3.0]], dtype=dtype)
    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(gate_up.grad, reference_gate_up.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("alpha", [1.702, 1.0])
def test_gpt_oss_clamped_swiglu_preserves_forward_and_backward(dtype, alpha):
    torch.manual_seed(0)
    gate = torch.randn(4, 4, dtype=dtype, requires_grad=True)
    up = torch.randn(4, 4, dtype=dtype, requires_grad=True)
    reference_gate = gate.detach().clone().requires_grad_()
    reference_up = up.detach().clone().requires_grad_()
    clamped_gate = reference_gate.clamp(max=2.0)
    clamped_up = reference_up.clamp(min=-2.0, max=2.0)
    silu = clamped_gate * torch.sigmoid(alpha * clamped_gate)
    expected = torch.addcmul(silu, silu, clamped_up)
    actual = ClampedSwiGLU.Config(swiglu_alpha=alpha, swiglu_limit=2.0).build()(
        torch.stack((gate, up), dim=-2)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (gate, up), grad_output)
    expected_grads = torch.autograd.grad(
        expected, (reference_gate, reference_up), grad_output
    )
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("swiglu_alpha", 0.0),
        ("swiglu_alpha", float("inf")),
        ("swiglu_limit", 0.0),
        ("swiglu_limit", float("nan")),
    ],
)
def test_clamped_swiglu_rejects_invalid_policy(field, value):
    with pytest.raises(ValueError, match=f"{field} must be finite and positive"):
        ClampedSwiGLU.Config(**{field: value})


def test_clamped_swiglu_rejects_unsupported_offset():
    with pytest.raises(ValueError, match=r"swiglu_offset must be 0\.0 or 1\.0"):
        ClampedSwiGLU.Config(swiglu_offset=0.5)


def test_clamped_swiglu_rejects_scaled_zero_offset():
    with pytest.raises(
        ValueError, match=r"swiglu_offset=0\.0 requires swiglu_alpha=1\.0"
    ):
        ClampedSwiGLU.Config(swiglu_alpha=1.702, swiglu_offset=0.0)
