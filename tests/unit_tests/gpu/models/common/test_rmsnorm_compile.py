# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common import RMSNorm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("shape", [(64, 2048), (64, 8, 128)])
def test_rmsnorm_region_matches_eager(shape):
    torch.manual_seed(0)
    norm = RMSNorm.Config(normalized_shape=shape[-1], eps=1e-6).build().cuda()
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    norm = norm.to(torch.bfloat16)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def run(regions):
        apply_local_compile(regions)
        norm.weight.grad = None
        leaf = x.clone().requires_grad_()
        out = norm(leaf)
        out.backward(torch.ones_like(out))
        return out, leaf.grad, norm.weight.grad

    try:
        expected = run([])
        actual = run(["rmsnorm"])
    finally:
        apply_local_compile([])
        torch._dynamo.reset()

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-2, rtol=2e-2)
