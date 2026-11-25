# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import ANY, call, patch

import spmd_types as spmd
import torch

from torchtitan.models.common.linear import Linear, RowParallelLinear
from torchtitan.models.common.moe.token_dispatcher import (
    LocalDispatchMetadata,
    LocalTokenDispatcher,
)
from torchtitan.models.qwen3_5.moe import SigmoidGatedFeedForward


class TestSigmoidGatedFeedForward(unittest.TestCase):
    def test_shared_input_is_gathered_once_before_the_projection_regions(self):
        x_TD = torch.randn(4, 4)
        tp_group = object()

        for sp_enabled, expected_names, expected_redistributions in (
            (
                False,
                ["linear", "linear", "linear"],
                [
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.I,
                        dst=spmd.R,
                        backward_options={"op_dtype": x_TD.dtype},
                    )
                ],
            ),
            (
                True,
                ["linear", "linear", "linear", "tp_reduce"],
                [
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.S(0),
                        dst=spmd.R,
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                    call(
                        ANY,
                        tp_group,
                        src=spmd.R,
                        dst=spmd.S(0),
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                    call(
                        ANY,
                        tp_group,
                        src=spmd.P,
                        dst=spmd.S(0),
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                ],
            ),
        ):
            w2_config = (
                RowParallelLinear.Config(in_features=8, out_features=4)
                if sp_enabled
                else Linear.Config(in_features=8, out_features=4)
            )
            shared_expert = SigmoidGatedFeedForward.Config(
                w13=Linear.Config(in_features=4, out_features=8, num_linears=2),
                w2=w2_config,
                gate=Linear.Config(in_features=4, out_features=1),
            ).build()
            with (
                self.subTest(sp_enabled=sp_enabled),
                patch(
                    "torchtitan.models.qwen3_5.moe.spmd_sparse_mesh",
                    return_value=object(),
                ),
                patch(
                    "torchtitan.models.qwen3_5.moe.spmd_dense_sp_enabled",
                    return_value=sp_enabled,
                ),
                patch(
                    "torchtitan.models.qwen3_5.moe.spmd_mesh_group",
                    return_value=tp_group,
                ),
                patch(
                    "torchtitan.models.common.linear.linear.spmd_dense_sp_enabled",
                    return_value=sp_enabled,
                ),
                patch(
                    "torchtitan.models.common.linear.linear.spmd_mesh_group",
                    return_value=tp_group,
                ),
                patch(
                    "torchtitan.models.qwen3_5.moe.spmd.redistribute",
                    side_effect=lambda tensor, *_args, **_kwargs: tensor,
                ) as redistribute,
                patch(
                    "torch_remat.region",
                    side_effect=lambda function, *_args, **_kwargs: function,
                ) as region,
            ):
                output_TD = shared_expert(x_TD)

            self.assertEqual(output_TD.shape, x_TD.shape)
            self.assertEqual(
                [region_call.args[1] for region_call in region.call_args_list],
                expected_names,
            )
            self.assertEqual(redistribute.call_args_list, expected_redistributions)


class TestWeightedGatherSumCombine(unittest.TestCase):
    def test_weighted_gather_sum_combine_matches_fp32_reference_fwd_and_bwd(self):
        from torchtitan.distributed.local_compile import LocalCompileConfig

        torch.manual_seed(0)
        num_tokens, dim, num_experts, top_k = 32, 64, 16, 4
        dispatcher = LocalTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k
        ).build()

        x = torch.randn(num_tokens, dim, dtype=torch.bfloat16)
        logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
        topk_scores, topk_expert_ids = torch.topk(
            torch.softmax(logits, dim=-1), k=top_k, dim=-1
        )
        routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool).scatter_(
            1, topk_expert_ids, True
        )

        routed_in, _, meta = dispatcher.dispatch(
            x, topk_scores, topk_expert_ids, routing_map
        )
        assert meta.perm_N is not None

        routed_out = torch.randn_like(routed_in).detach().requires_grad_(True)
        scores_sorted = meta.topk_scores_experts_sorted_N.detach().requires_grad_(True)
        meta_grad = LocalDispatchMetadata(
            token_indices_experts_sorted_N=meta.token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=scores_sorted,
            perm_N=meta.perm_N,
        )
        grad_out = torch.randn(num_tokens, dim, dtype=torch.bfloat16)

        LocalCompileConfig(regions=["moe_weighted_gather_sum"]).apply()
        try:
            out = dispatcher.combine(routed_out, meta_grad, x)
            out.backward(grad_out)

            # Exact fp32 reference without scatter atomics or bf16 intermediate truncation.
            routed_ref = routed_out.detach().float().requires_grad_(True)
            scores_ref = scores_sorted.detach().float().requires_grad_(True)
            out_ref = torch.zeros(num_tokens, dim, dtype=torch.float32)
            out_ref.index_add_(
                0,
                meta.token_indices_experts_sorted_N,
                routed_ref * scores_ref.unsqueeze(-1),
            )
            out_ref.backward(grad_out.float())

            torch.testing.assert_close(out.float(), out_ref, rtol=1e-2, atol=2e-2)
            assert routed_out.grad is not None and routed_ref.grad is not None
            assert scores_sorted.grad is not None and scores_ref.grad is not None
            torch.testing.assert_close(
                routed_out.grad.float(), routed_ref.grad, rtol=1e-2, atol=2e-2
            )
            torch.testing.assert_close(
                scores_sorted.grad.float(), scores_ref.grad, rtol=1e-2, atol=2e-2
            )

            # Verify needs_input_grad guards when only one input requires grad.
            routed_only = routed_out.detach().requires_grad_(True)
            meta_no_score_grad = LocalDispatchMetadata(
                token_indices_experts_sorted_N=meta.token_indices_experts_sorted_N,
                topk_scores_experts_sorted_N=scores_sorted.detach(),
                perm_N=meta.perm_N,
            )
            out_routed_only = dispatcher.combine(routed_only, meta_no_score_grad, x)
            out_routed_only.backward(grad_out)
            assert routed_only.grad is not None
            torch.testing.assert_close(
                routed_only.grad, routed_out.grad, rtol=0, atol=0
            )
        finally:
            LocalCompileConfig(regions=[]).apply()

        # When moe_weighted_gather_sum is disabled, combine preserves the scatter fallback.
        with patch(
            "torchtitan.models.common.moe.token_dispatcher._WeightedGatherSumTopK.apply",
            side_effect=AssertionError("fused combine must stay gated off"),
        ):
            fallback_out = dispatcher.combine(
                routed_out.detach(),
                LocalDispatchMetadata(
                    token_indices_experts_sorted_N=meta.token_indices_experts_sorted_N,
                    topk_scores_experts_sorted_N=scores_sorted.detach(),
                    perm_N=meta.perm_N,
                ),
                x,
            )
            self.assertEqual(fallback_out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
