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

    def test_gather_topk_rows_and_score_scale_match_eager(self) -> None:
        from torchtitan.models.common.moe.token_dispatcher import (
            LocalTokenDispatcher,
            scale_by_router_scores,
        )

        torch.manual_seed(0)
        num_tokens, dim, num_experts, top_k = 12, 16, 8, 4
        dispatcher = LocalTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k
        ).build()
        x_TD = torch.randn(num_tokens, dim, dtype=torch.bfloat16, requires_grad=True)
        x_ref = x_TD.detach().clone().requires_grad_()
        scores_TK = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)
        expert_ids_TK = torch.randint(0, num_experts, (num_tokens, top_k))

        routed_ND, tok_idx_N, sorted_scores_N = dispatcher._local_reorder(
            x_TD, scores_TK, expert_ids_TK
        )
        perm_ref = torch.argsort(expert_ids_TK.view(-1).to(torch.uint8), stable=True)
        routed_ref = x_ref[perm_ref // top_k]
        torch.testing.assert_close(routed_ND, routed_ref, rtol=0, atol=0)

        grad_ND = torch.randn_like(routed_ND)
        routed_ND.backward(grad_ND)
        routed_ref.backward(grad_ND)
        torch.testing.assert_close(x_TD.grad, x_ref.grad, rtol=1e-2, atol=1e-2)

        # Verify score scaling matches fp32 multiply-and-cast bitwise.
        y_ND = torch.randn(
            num_tokens * top_k, dim, dtype=torch.bfloat16, requires_grad=True
        )
        y_ref = y_ND.detach().clone().requires_grad_()
        s_N = sorted_scores_N.detach().clone().requires_grad_()
        s_ref = s_N.detach().clone().requires_grad_()
        out = scale_by_router_scores(y_ND, s_N)
        out_ref = (y_ref * s_ref.reshape(-1, 1).to(torch.float32)).to(y_ref.dtype)
        torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
        g_out = torch.randn_like(out)
        out.backward(g_out)
        out_ref.backward(g_out)
        torch.testing.assert_close(y_ND.grad, y_ref.grad, rtol=0, atol=0)
        torch.testing.assert_close(s_N.grad, s_ref.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
