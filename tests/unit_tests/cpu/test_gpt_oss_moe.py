# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.models.gpt_oss.moe import swiglu


def _reference_swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


def test_swiglu_matches_reference_formula_and_gradients():
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    out = swiglu(x, alpha=1.5, limit=2.0)
    out_ref = _reference_swiglu(x_ref, alpha=1.5, limit=2.0)

    torch.testing.assert_close(out, out_ref)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    torch.testing.assert_close(x.grad, x_ref.grad)
