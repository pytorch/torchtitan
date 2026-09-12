# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.decoder_sharding import (
    token_id_placement,
    token_id_sequence_parallel_placement,
)
from torchtitan.models.common.moe_sharding import _moe_sharding_config


class _PassthroughRoutedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tokens_per_expert_E = None

    def forward(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        self.num_tokens_per_expert_E = num_local_tokens_per_expert_E
        return x_TD


class _CapturingAuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.routing_map_TE = None

    def forward(self, scores_TE, routing_map_TE, *, carrier, padding_mask=None):
        del scores_TE, padding_mask
        self.routing_map_TE = routing_map_TE
        return carrier


class TestMoE(unittest.TestCase):
    def _build_moe(self):
        num_experts = 2
        dim = 4
        top_k = 1
        moe = make_moe_config(
            num_experts=num_experts,
            router=make_router_config(
                dim=dim,
                num_experts=num_experts,
                gate_param_init={"weight": nn.init.zeros_},
                top_k=top_k,
            ),
            routed_experts=make_routed_experts_config(
                dim=dim,
                hidden_dim=8,
                num_experts=num_experts,
                top_k=top_k,
                param_init={},
                comm_backend="standard",
            ),
        ).build()
        with torch.no_grad():
            moe.router.gate.weight.zero_()
        moe.routed_experts = _PassthroughRoutedExperts()
        return moe

    def test_eval_forward_does_not_accumulate_tokens_per_expert(self):
        dim = 4
        top_k = 1
        moe = self._build_moe()

        x_TD = torch.randn(6, dim)
        moe.train()
        moe(x_TD)
        self.assertEqual(moe.router.tokens_per_expert_E.sum().item(), 2 * 3 * top_k)
        self.assertIs(moe.tokens_per_expert_E, moe.router.tokens_per_expert_E)
        training_counts = moe.tokens_per_expert_E.clone()

        moe.eval()
        with torch.no_grad():
            moe(x_TD)

        torch.testing.assert_close(
            moe.tokens_per_expert_E,
            training_counts,
        )

    def test_padding_is_excluded_from_counts_but_still_dispatched(self):
        moe = self._build_moe()
        x_TD = torch.randn(6, 4)
        padding_mask = torch.tensor([False, False, False, True, True, True])

        moe.train()
        moe(x_TD, padding_mask=padding_mask)

        self.assertEqual(moe.router.tokens_per_expert_E.sum().item(), 3)
        self.assertEqual(moe.routed_experts.num_tokens_per_expert_E.sum().item(), 6)

    def test_router_masks_padding_only_for_aux_loss(self):
        router = make_router_config(
            dim=4,
            num_experts=2,
            gate_param_init={"weight": nn.init.zeros_},
            top_k=1,
        ).build()
        router.init_states()
        aux_loss = _CapturingAuxLoss()
        router.aux_loss = aux_loss
        router.train()

        padding_mask = torch.tensor([False, False, True, True])
        _, _, routing_map_TE = router(
            torch.randn(4, 4),
            padding_mask=padding_mask,
        )

        self.assertTrue(routing_map_TE[padding_mask].any())
        self.assertIsNotNone(aux_loss.routing_map_TE)
        self.assertFalse(aux_loss.routing_map_TE[padding_mask].any())
        torch.testing.assert_close(
            aux_loss.routing_map_TE[~padding_mask],
            routing_map_TE[~padding_mask],
        )
        torch.testing.assert_close(
            router.tokens_per_expert_E,
            aux_loss.routing_map_TE.sum(dim=0).to(torch.float32),
        )

    def test_padding_mask_sharding_matches_router_token_layout(self):
        config = _moe_sharding_config(enable_ep=True, enable_sp=False)
        assert config.in_src_shardings is not None
        assert config.in_dst_shardings is not None
        padding_mask_src = config.in_src_shardings["padding_mask"]
        padding_mask_dst = config.in_dst_shardings["padding_mask"]

        self.assertEqual(
            _per_axis_types(padding_mask_src),
            _per_axis_types(token_id_placement()),
        )
        self.assertEqual(
            _per_axis_types(padding_mask_dst),
            _per_axis_types(token_id_sequence_parallel_placement()),
        )


if __name__ == "__main__":
    unittest.main()
