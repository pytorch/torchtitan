# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchtitan.ops.grid_sample import deterministic_grid_sample


def _inputs():
    torch.manual_seed(0)
    input_NCHW = torch.randn(2, 5, 7, 9, dtype=torch.float64, requires_grad=True)
    # Some points fall outside [-1, 1] to cover zeros padding.
    grid = torch.rand(2, 4, 6, 2, dtype=torch.float64) * 2.4 - 1.2
    return input_NCHW, grid


def test_matches_grid_sample():
    input_NCHW, grid = _inputs()
    output = deterministic_grid_sample(input_NCHW, grid)
    expected = F.grid_sample(
        input_NCHW, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    grad_output = torch.randn_like(output)
    (grad,) = torch.autograd.grad(output, input_NCHW, grad_output)
    (expected_grad,) = torch.autograd.grad(expected, input_NCHW, grad_output)
    torch.testing.assert_close(grad, expected_grad, rtol=0, atol=1e-12)


def test_gradcheck_and_opcheck():
    input_NCHW, grid = _inputs()
    assert torch.autograd.gradcheck(
        lambda x: deterministic_grid_sample(x, grid), (input_NCHW,)
    )
    torch.library.opcheck(deterministic_grid_sample, (input_NCHW, grid))


def test_rejects_grid_grad():
    input_NCHW, grid = _inputs()
    with pytest.raises(ValueError, match="grid gradient"):
        deterministic_grid_sample(input_NCHW, grid.requires_grad_())
