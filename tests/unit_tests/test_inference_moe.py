# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fused_grouped_experts + deepep_override compose as disjoint sibling nodes
under moe.routed_experts (no ancestor/descendant conflict)."""

import unittest
from functools import partial

import torch
from torch.nn import init

from torchtitan.config.override import _REGISTRY, apply_overrides, OverrideConfig
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.moe import (
    _build_token_valid_mask,
    GroupedExperts,
    MoE,
    RoutedExperts,
)
from torchtitan.models.common.token_dispatcher import (
    DeepEPTokenDispatcher,
    LocalTokenDispatcher,
)
from torchtitan.overrides.fused_swiglu import fused_grouped_experts, FusedGroupedExperts
from torchtitan.overrides.moe_token_dispatcher import deepep_override

_DIM = 16
_HIDDEN = 32
_E = 4

# fused_swiglu registers both the dense-FFN and routed-experts (fused_grouped_experts)
# overrides; each is activated by its own module.function target.
_FUSED_SWIGLU = (
    "torchtitan.overrides.fused_swiglu.fused_swiglu",
    "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
)
_DEEPEP_OVERRIDE = (
    "torchtitan.overrides.moe_token_dispatcher.deepep_override",
    {"cudagraphable": True},
)

# The @override decorators register once, at the imports above. Capture the
# entries this test needs so a sibling test that calls clear_overrides() (e.g.
# test_override.py) can't leave the registry empty; setUp restores them.
_OVERRIDES = {
    key: _REGISTRY[key]
    for key in (
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
        "torchtitan.overrides.moe_token_dispatcher.deepep_override",
    )
    if key in _REGISTRY
}


def _moe_config(comm_backend: str):
    param_init = {
        "w1_EFD": partial(init.trunc_normal_, std=0.02),
        "w2_EDF": partial(init.trunc_normal_, std=0.02),
        "w3_EFD": partial(init.trunc_normal_, std=0.02),
    }
    routed_experts = make_routed_experts_config(
        dim=_DIM,
        hidden_dim=_HIDDEN,
        num_experts=_E,
        top_k=1,
        param_init=param_init,
        comm_backend=comm_backend,
    )
    router = make_router_config(
        dim=_DIM,
        num_experts=_E,
        gate_param_init={"weight": partial(init.trunc_normal_, std=0.02)},
        top_k=1,
    )
    return make_moe_config(num_experts=_E, router=router, routed_experts=routed_experts)


class TestInferenceMoEOverrides(unittest.TestCase):
    def setUp(self):
        # Restore the overrides if a previously run test cleared the registry.
        for name, ov in _OVERRIDES.items():
            _REGISTRY.setdefault(name, ov)

    def test_grouped_experts_and_dispatcher_are_siblings(self):
        cfg = _moe_config("deepep")
        self.assertIsInstance(cfg.routed_experts.inner_experts, GroupedExperts.Config)
        self.assertIsInstance(
            cfg.routed_experts.token_dispatcher, DeepEPTokenDispatcher.Config
        )

    def test_deepep_both_overrides_apply_without_conflict(self):
        cfg = _moe_config("deepep")

        replacements = apply_overrides(
            OverrideConfig(imports=[*_FUSED_SWIGLU, _DEEPEP_OVERRIDE]),
            cfg,
        )

        self.assertEqual(len(replacements), 2)
        self.assertIsInstance(
            cfg.routed_experts.inner_experts, FusedGroupedExperts.Config
        )
        self.assertIsInstance(
            cfg.routed_experts.token_dispatcher, DeepEPTokenDispatcher.Config
        )
        self.assertTrue(cfg.routed_experts.token_dispatcher.cudagraphable)

    def test_non_deepep_dispatcher_flip_is_noop(self):
        cfg = _moe_config("standard")

        # deepep_override targets DeepEP only; on a standard dispatcher just fusion applies.
        replacements = apply_overrides(
            OverrideConfig(imports=[*_FUSED_SWIGLU, _DEEPEP_OVERRIDE]),
            cfg,
        )

        self.assertEqual(len(replacements), 1)
        self.assertIsInstance(
            cfg.routed_experts.inner_experts, FusedGroupedExperts.Config
        )

    def test_composition_is_order_independent(self):
        # Disjoint sibling nodes -> either application order yields the same result.
        def summarize(ge):
            return (
                type(ge.inner_experts).__qualname__,
                type(ge.token_dispatcher).__qualname__,
                ge.token_dispatcher.cudagraphable,
            )

        a = _moe_config("deepep").routed_experts
        a.inner_experts = fused_grouped_experts(a.inner_experts)
        a.token_dispatcher = deepep_override(a.token_dispatcher, cudagraphable=True)

        b = _moe_config("deepep").routed_experts
        b.token_dispatcher = deepep_override(b.token_dispatcher, cudagraphable=True)
        b.inner_experts = fused_grouped_experts(b.inner_experts)

        self.assertEqual(summarize(a), summarize(b))
        self.assertIsInstance(a.inner_experts, FusedGroupedExperts.Config)
        self.assertTrue(a.token_dispatcher.cudagraphable)

    def test_trainer_uses_only_experts_fusion(self):
        cfg = _moe_config("deepep")

        # Trainer imports only fused_swiglu: experts fused, dispatcher left compact.
        apply_overrides(OverrideConfig(imports=[*_FUSED_SWIGLU]), cfg)

        self.assertIsInstance(
            cfg.routed_experts.inner_experts, FusedGroupedExperts.Config
        )
        self.assertFalse(cfg.routed_experts.token_dispatcher.cudagraphable)


class _IdentityExperts(torch.nn.Module):
    def forward(self, x_RD, num_tokens_per_expert_E):
        del num_tokens_per_expert_E
        return x_RD


class _FixedRouter(torch.nn.Module):
    def forward(self, x_BLD, expert_bias_E):
        del expert_bias_E
        B, L, _ = x_BLD.shape
        topk_scores_BLK = torch.ones(B, L, 1, device=x_BLD.device)
        topk_expert_ids_BLK = torch.zeros(
            B, L, 1, dtype=torch.int64, device=x_BLD.device
        )
        scores_BLE = torch.zeros(B, L, 2, device=x_BLD.device)
        return topk_scores_BLK, topk_expert_ids_BLK, scores_BLE


class _CapturingRoutedExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(num_experts=2, top_k=1)
        )
        self.topk_scores_BLK = None
        self.num_local_tokens_per_expert_E = None

    def forward(
        self,
        x_BLD,
        topk_scores_BLK,
        topk_expert_ids_BLK,
        num_local_tokens_per_expert_E,
        *,
        num_actual_tokens=None,
    ):
        del topk_expert_ids_BLK, num_actual_tokens
        self.topk_scores_BLK = topk_scores_BLK
        self.num_local_tokens_per_expert_E = num_local_tokens_per_expert_E
        return x_BLD * topk_scores_BLK


def _new_test_moe(*, tp_rank=0, tp_size=1):
    moe = MoE.__new__(MoE)
    torch.nn.Module.__init__(moe)
    moe.input_sequence_parallel = tp_size > 1
    moe.tp_rank = tp_rank
    moe.tp_size = tp_size
    moe.router = _FixedRouter()
    moe.routed_experts = _CapturingRoutedExperts()
    moe.shared_experts = torch.nn.Identity()
    moe.expert_bias_E = None
    moe.register_buffer("tokens_per_expert_E", torch.zeros(2, dtype=torch.int64))
    return moe


class TestMoEActualTokens(unittest.TestCase):
    def test_sequence_sharded_valid_mask(self):
        num_actual_tokens = torch.tensor([5], dtype=torch.int64)
        expected = (
            [[True, True]],
            [[True, True]],
            [[True, False]],
            [[False, False]],
        )

        for rank, expected_rank in enumerate(expected):
            valid_BL = _build_token_valid_mask(
                torch.empty(1, 2),
                num_actual_tokens,
                tp_rank=rank,
                tp_size=4,
            )
            self.assertEqual(valid_BL.tolist(), expected_rank)

    def test_valid_mask_accounts_for_batch_offsets(self):
        valid_BL = _build_token_valid_mask(
            torch.empty(2, 2),
            torch.tensor([5], dtype=torch.int64),
            tp_rank=0,
            tp_size=2,
        )

        self.assertEqual(valid_BL.tolist(), [[True, True], [True, False]])

    def test_dynamic_routing_restores_padded_extent(self):
        routed_experts = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(routed_experts)
        routed_experts.inner_experts = _IdentityExperts()
        routed_experts.token_dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(num_experts=2, top_k=1)
        )
        routed_experts.tp_rank = 0
        routed_experts.tp_size = 1

        x_BLD = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
        out_BLD = routed_experts(
            x_BLD,
            torch.ones(1, 4, 1),
            torch.zeros(1, 4, 1, dtype=torch.int64),
            torch.tensor([3, 0]),
            num_actual_tokens=torch.tensor([3], dtype=torch.int64),
        )

        torch.testing.assert_close(out_BLD[:, :3], x_BLD[:, :3])
        torch.testing.assert_close(out_BLD[:, 3:], torch.zeros_like(x_BLD[:, 3:]))

    def test_moe_excludes_padding_from_routing_and_shared_output(self):
        moe = _new_test_moe()
        x_BLD = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
        out_BLD = moe(
            x_BLD,
            num_actual_tokens=torch.tensor([3], dtype=torch.int64),
        )

        torch.testing.assert_close(out_BLD[:, :3], 2 * x_BLD[:, :3])
        torch.testing.assert_close(out_BLD[:, 3:], torch.zeros_like(x_BLD[:, 3:]))
        self.assertEqual(moe.tokens_per_expert_E.tolist(), [3, 0])
        self.assertEqual(
            moe.routed_experts.num_local_tokens_per_expert_E.tolist(), [3, 0]
        )
        self.assertEqual(
            moe.routed_experts.topk_scores_BLK.squeeze(-1).tolist(),
            [[1.0, 1.0, 1.0, 0.0]],
        )

    def test_moe_derives_local_mask_when_dense_input_is_tp_sharded(self):
        moe = _new_test_moe(tp_rank=2, tp_size=4)
        x_BLD = torch.arange(4, dtype=torch.float32).view(1, 2, 2)

        out_BLD = moe(
            x_BLD,
            num_actual_tokens=torch.tensor([5], dtype=torch.int64),
        )

        torch.testing.assert_close(out_BLD[:, :1], 2 * x_BLD[:, :1])
        torch.testing.assert_close(out_BLD[:, 1:], torch.zeros_like(x_BLD[:, 1:]))
        self.assertEqual(moe.tokens_per_expert_E.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
