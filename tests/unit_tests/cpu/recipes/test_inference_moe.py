# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused activation override and the DeepEP dispatcher transform compose."""

import unittest
from functools import partial

from torch.nn import init

from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.config.transform import (
    TokenDispatcherTransform,
    transform_model_config_,
)
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher
from torchtitan_recipes.overrides.fused_swiglu import FusedSwiGLU

_DIM = 16
_HIDDEN = 32
_E = 4

_FUSED_SWIGLU = "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu"


def _moe_config():
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
        enable_ep=True,
    )
    router = make_router_config(
        dim=_DIM,
        num_experts=_E,
        gate_param_init={"weight": partial(init.trunc_normal_, std=0.02)},
        score_func=Sigmoid.Config(),
        top_k=1,
    )
    return make_moe_config(
        num_experts=_E, router=router, routed_experts=routed_experts, enable_sp=True
    )


def _deepep(cfg, **kwargs):
    transform = TokenDispatcherTransform(
        dispatcher=DeepEPTokenDispatcher, num_max_tokens_per_rank=16, kwargs=kwargs
    )
    return transform_model_config_(cfg, [transform])


class TestInferenceMoEComposition(unittest.TestCase):
    def test_grouped_linears_and_dispatcher_are_siblings(self):
        """Expert projections and the dispatcher are first-class siblings."""
        cfg = _deepep(_moe_config())
        self.assertIsInstance(cfg.routed_experts.w13, GroupedLinear.Config)
        self.assertIsInstance(cfg.routed_experts.w2, GroupedLinear.Config)
        self.assertIsInstance(
            cfg.routed_experts.token_dispatcher, DeepEPTokenDispatcher.Config
        )

    def test_generator_transform_and_fused_override_compose(self):
        cfg = _deepep(_moe_config(), cuda_graph_compatible=True)

        replacements = apply_overrides(OverrideConfig(imports=[_FUSED_SWIGLU]), cfg)

        self.assertEqual(len(replacements), 1)
        self.assertIsInstance(cfg.routed_experts.activation_fn, FusedSwiGLU.Config)
        self.assertTrue(cfg.routed_experts.token_dispatcher.cuda_graph_compatible)

    def test_composition_is_order_independent(self):
        def summarize(cfg):
            experts = cfg.routed_experts
            return (
                type(experts.activation_fn).__qualname__,
                type(experts.token_dispatcher).__qualname__,
                experts.token_dispatcher.cuda_graph_compatible,
            )

        a = _deepep(_moe_config(), cuda_graph_compatible=True)
        apply_overrides(OverrideConfig(imports=[_FUSED_SWIGLU]), a)

        b = _moe_config()
        apply_overrides(OverrideConfig(imports=[_FUSED_SWIGLU]), b)
        b = _deepep(b, cuda_graph_compatible=True)

        self.assertEqual(summarize(a), summarize(b))

    def test_trainer_keeps_compact_dispatch(self):
        cfg = _deepep(_moe_config())

        apply_overrides(OverrideConfig(imports=[_FUSED_SWIGLU]), cfg)

        self.assertIsInstance(cfg.routed_experts.activation_fn, FusedSwiGLU.Config)
        self.assertFalse(cfg.routed_experts.token_dispatcher.cuda_graph_compatible)


if __name__ == "__main__":
    unittest.main()
