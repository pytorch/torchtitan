# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.deepseek_v4.mhc import HcHead, HcPost, HcPre

TOKENS, HC_MULT, DIM = 64, 4, 256


def _inputs(*shapes, dtype=torch.bfloat16):
    return [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for shape in shapes
    ]


def _random_parameters(module):
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param) * 0.1)
    return module


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMhcLocalCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def _run(self, module, inputs):
        out = module(*inputs)
        out = out if isinstance(out, tuple) else (out,)
        generator = torch.Generator(device="cuda").manual_seed(1)
        grad_out = [
            torch.randn(o.shape, dtype=o.dtype, device="cuda", generator=generator)
            for o in out
        ]
        torch.autograd.backward(out, grad_out)
        grads = [x.grad for x in inputs] + [p.grad for p in module.parameters()]
        values = [o.detach().clone() for o in out] + [g.clone() for g in grads]
        for x in inputs:
            x.grad = None
        module.zero_grad(set_to_none=True)
        return values

    def _assert_compiled_matches_eager(self, module, inputs):
        apply_local_compile([])
        eager = self._run(module, inputs)
        apply_local_compile(["mhc"])
        compiled = self._run(module, inputs)
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_hc_pre_matches_eager(self):
        torch.manual_seed(0)
        module = _random_parameters(
            HcPre.Config(hc_mult=HC_MULT, dim=DIM).build().cuda()
        )
        self._assert_compiled_matches_eager(module, _inputs((TOKENS, HC_MULT, DIM)))

    def test_hc_post_matches_eager(self):
        torch.manual_seed(0)
        module = HcPost.Config().build().cuda()
        x, residual = _inputs((TOKENS, DIM), (TOKENS, HC_MULT, DIM))
        post, comb = _inputs(
            (TOKENS, HC_MULT), (TOKENS, HC_MULT, HC_MULT), dtype=torch.float32
        )
        self._assert_compiled_matches_eager(module, [x, residual, post, comb])

    def test_hc_head_matches_eager(self):
        torch.manual_seed(0)
        module = _random_parameters(
            HcHead.Config(hc_mult=HC_MULT, dim=DIM).build().cuda()
        )
        self._assert_compiled_matches_eager(module, _inputs((TOKENS, HC_MULT, DIM)))


if __name__ == "__main__":
    unittest.main()
