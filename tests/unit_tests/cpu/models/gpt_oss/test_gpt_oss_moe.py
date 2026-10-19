# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.models.common.activation import ClampedSwiGLU


def _reference_swiglu(gate, up, limit: float = 7.0):
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    silu = gate * torch.sigmoid(1.702 * gate)
    return torch.addcmul(silu, silu, up)


def test_swiglu_matches_reference_formula_and_gradients():
    """The configured GPT-OSS activation matches its forward and backward formula."""
    torch.manual_seed(0)
    gate = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    up = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_()
    up_ref = up.detach().clone().requires_grad_()

    out = ClampedSwiGLU.Config(swiglu_limit=2.0).build()(
        torch.stack([gate, up], dim=-2)
    )
    out_ref = _reference_swiglu(gate_ref, up_ref, limit=2.0)

    assert torch.equal(out, out_ref)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    assert torch.equal(gate.grad, gate_ref.grad)
    assert torch.equal(up.grad, up_ref.grad)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("swiglu_alpha", 0.0),
        ("swiglu_alpha", float("inf")),
        ("swiglu_limit", 0.0),
        ("swiglu_limit", float("nan")),
    ],
)
def test_clamped_swiglu_rejects_invalid_policy(field: str, value: float) -> None:
    """Invalid kernel policy is rejected by the source activation config."""
    with pytest.raises(ValueError, match=f"{field} must be finite and positive"):
        ClampedSwiGLU.Config(**{field: value})
