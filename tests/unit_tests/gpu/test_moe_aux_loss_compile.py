# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.moe import _microbatch_load_balance_local_stats


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMoEAuxLossLocalCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_forward_and_backward_match_eager(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        scores_TE = torch.rand(
            128,
            64,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
            requires_grad=True,
        )
        expert_ids_TK = torch.topk(
            scores_TE.detach(), k=8, dim=-1, sorted=False
        ).indices
        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1, expert_ids_TK, True
        )
        padding_mask_T = torch.zeros(128, device="cuda", dtype=torch.bool)
        padding_mask_T[::11] = True
        grad_prob_sums_E = torch.randn(
            64,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )

        apply_local_compile([])
        eager_counts_E, eager_prob_sums_E = _microbatch_load_balance_local_stats(
            scores_TE,
            routing_map_TE,
            padding_mask_T,
        )
        eager_grad_TE = torch.autograd.grad(
            eager_prob_sums_E, scores_TE, grad_prob_sums_E
        )[0]

        apply_local_compile(["moe_aux_loss"])
        compiled_counts_E, compiled_prob_sums_E = _microbatch_load_balance_local_stats(
            scores_TE,
            routing_map_TE,
            padding_mask_T,
        )
        compiled_grad_TE = torch.autograd.grad(
            compiled_prob_sums_E, scores_TE, grad_prob_sums_E
        )[0]

        torch.testing.assert_close(compiled_counts_E, eager_counts_E, rtol=0, atol=0)
        torch.testing.assert_close(
            compiled_prob_sums_E, eager_prob_sums_E, rtol=1e-6, atol=1e-5
        )
        torch.testing.assert_close(
            compiled_grad_TE,
            eager_grad_TE,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
