# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fused_grouped_experts + deepep_override compose as disjoint sibling nodes
under moe.routed_experts (no ancestor/descendant conflict)."""

import unittest
from functools import partial

from torch.nn import init

from torchtitan.config.override import _REGISTRY, apply_overrides, OverrideConfig
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher
from torchtitan.overrides.fused_swiglu import fused_grouped_experts, FusedGroupedExperts
from torchtitan.overrides.moe_dispatch_override import deepep_override

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
    "torchtitan.overrides.moe_dispatch_override.deepep_override",
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
        "torchtitan.overrides.moe_dispatch_override.deepep_override",
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


if __name__ == "__main__":
    unittest.main()
