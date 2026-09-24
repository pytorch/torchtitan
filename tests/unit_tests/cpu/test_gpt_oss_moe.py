# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.models.gpt_oss.moe import GptOssSwiGLU


def _reference_swiglu(gate, up, limit: float = 7.0):
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    silu = gate * torch.sigmoid(1.702 * gate)
    return silu * (up + 1)


def test_swiglu_matches_reference_formula_and_gradients():
    """The configured GPT-OSS activation matches its forward and backward formula."""
    torch.manual_seed(0)
    gate = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    up = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_()
    up_ref = up.detach().clone().requires_grad_()

    out = GptOssSwiGLU.Config(swiglu_limit=2.0).build()(gate, up)
    out_ref = _reference_swiglu(gate_ref, up_ref, limit=2.0)

    torch.testing.assert_close(out, out_ref)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    torch.testing.assert_close(gate.grad, gate_ref.grad)
    torch.testing.assert_close(up.grad, up_ref.grad)
