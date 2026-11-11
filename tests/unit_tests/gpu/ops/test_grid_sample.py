# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.ops.grid_sample import deterministic_grid_sample


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_input_grad_is_bitwise_repeatable(dtype):
    # A NaViT-sized position table sampled at a 37x37 grid; the CUDA
    # grid_sample backward is not repeatable at this size.
    torch.manual_seed(0)
    table = torch.randn(1, 1536, 32, 32, device="cuda", dtype=dtype, requires_grad=True)
    coords = torch.linspace(-1 + 1 / 37, 1 - 1 / 37, 37, device="cuda", dtype=dtype)
    grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), -1)
    grid = grid.reshape(1, 1, 37 * 37, 2)
    grad_output = torch.randn(1, 1536, 1, 37 * 37, device="cuda", dtype=dtype)
    compiled = torch.compile(deterministic_grid_sample, fullgraph=True)

    grads = []
    for fn in [deterministic_grid_sample] * 10 + [compiled] * 2:
        table.grad = None
        fn(table, grid).backward(grad_output)
        grads.append(table.grad.clone())
    # Each path must repeat itself. Eager and compiled may round differently.
    for grad in grads[1:10]:
        assert torch.equal(grad, grads[0])
    assert torch.equal(grads[11], grads[10])
