# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.qwen3.model import _attention_residual_norm


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AttentionResidualNormLocalCompileTest(unittest.TestCase):
    def setUp(self):
        LocalCompileConfig(regions=["attention_residual_norm"]).apply()

    def tearDown(self):
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    def test_forward_and_backward_match_rms_norm(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        norm = RMSNorm.Config(normalized_shape=128, eps=1e-6).build().cuda()
        norm = norm.to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.normal_(generator=generator)
        reference_norm = deepcopy(norm)

        def randn() -> torch.Tensor:
            return torch.randn(
                64,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )

        h_TD = randn().requires_grad_()
        attn_out_TD = randn().requires_grad_()
        grad_h_TD = randn()
        grad_norm_TD = randn()
        assert norm.eps is not None
        output = _attention_residual_norm(
            h_TD,
            attn_out_TD,
            norm.weight,
            norm.eps,
        )
        grads = torch.autograd.grad(
            output,
            (h_TD, attn_out_TD, norm.weight),
            (grad_h_TD, grad_norm_TD),
        )

        LocalCompileConfig().apply()
        reference_h_TD = h_TD.detach().clone().requires_grad_()
        reference_attn_out_TD = attn_out_TD.detach().clone().requires_grad_()
        reference_residual_h_TD = reference_h_TD + reference_attn_out_TD
        reference_output = (
            reference_residual_h_TD,
            reference_norm(reference_residual_h_TD),
        )
        reference_grads = torch.autograd.grad(
            reference_output,
            (
                reference_h_TD,
                reference_attn_out_TD,
                reference_norm.weight,
            ),
            (grad_h_TD, grad_norm_TD),
        )

        for actual, expected in zip(output, reference_output, strict=True):
            torch.testing.assert_close(actual, expected)
        # Fused RMSNorm backward can differ from eager by a few bf16 ulps.
        for actual, expected in zip(grads[:2], reference_grads[:2], strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(grads[2], reference_grads[2])


if __name__ == "__main__":
    unittest.main()
