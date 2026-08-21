# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for ModelSpec integration.

Covers:
* ``KimiK3Spec.build()`` dispatches to baseline vs AttnRes variant.
* ``model_registry(flavor)`` returns a valid :class:`ModelSpec` for each
  of the 15 scaling-law flavors.
* ``Trainer.Config`` factory resolves for at least one flavor.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3 import (
    flavor_names,
    KimiK3AttnResModel,
    KimiK3Model,
    KimiK3Spec,
    model_registry,
)
from torchtitan.models.kimi_k3.config_registry import (
    build_kimi_linear_config,
    kimi_linear_194m_baseline,
    kimi_linear_528m_block_attn_res,
    SCALING_LAW_TABLE,
)
from torchtitan.protocols.model_spec import ModelSpec


class TestKimiK3Spec(unittest.TestCase):
    def test_baseline_build(self):
        kcfg = build_kimi_linear_config("194m")
        spec = KimiK3Spec(kimi_config=kcfg, num_blocks=None)
        model = spec.build()
        self.assertIsInstance(model, KimiK3Model)

    def test_attn_res_build(self):
        kcfg = build_kimi_linear_config("194m")
        spec = KimiK3Spec(kimi_config=kcfg, num_blocks=12)
        model = spec.build()
        self.assertIsInstance(model, KimiK3AttnResModel)
        self.assertEqual(model.num_blocks, 12)

    def test_nparams_and_flops(self):
        kcfg = build_kimi_linear_config("194m")
        spec = KimiK3Spec(kimi_config=kcfg, num_blocks=None)
        model = spec.build()
        n_params, flops = spec.get_nparams_and_flops(model, seq_len=8192)
        self.assertGreater(n_params, 100_000_000)  # ~580M total MoE params
        # flops is per-TOKEN (not per-step), and it must follow the ACTIVATED
        # parameter count: 32 experts with top_k 8 means most of the 579M total
        # does not participate in a given token. The band here is deliberately
        # narrow -- the version of this assertion that spanned 1e6 to 1e11
        # admitted a 26x over-count without complaint.
        self.assertGreater(flops, 1.0e9)
        self.assertLess(flops, 1.5e9)
        # And specifically not the all-experts-activated answer, which is what
        # counting every routed parameter as dense produces.
        self.assertLess(flops, 6 * n_params // 2)


class TestModelRegistry(unittest.TestCase):
    def test_all_flavors_build(self):
        flavors = flavor_names()
        # flavor_names() = every scaling-law size × 3 AttnRes variants
        # (baseline / block_attn_res / full_attn_res). Derive the expected
        # count from the table so adding a size row can't silently drift it.
        self.assertEqual(len(flavors), len(SCALING_LAW_TABLE) * 3)
        for flavor in flavors:
            spec = model_registry(flavor)
            self.assertIsInstance(spec, ModelSpec)
            self.assertEqual(spec.name, "kimi_linear")
            self.assertEqual(spec.flavor, flavor)
            # pipelining_fn is wired (runtime-dispatches
            # to cache adapter when AttnRes+Interleaved1F1B, else PP passthrough).
            self.assertIsNotNone(spec.pipelining_fn)
            self.assertIsNotNone(spec.parallelize_fn)
            self.assertIsNotNone(spec.parallelize_fn)

    def test_reject_unknown_flavor(self):
        with self.assertRaises(ValueError):
            model_registry("kimi_linear_999q_baseline")

    def test_reject_malformed_flavor(self):
        with self.assertRaises(ValueError):
            model_registry("not_kimi_linear_194m_baseline")


class TestTrainerConfigFactory(unittest.TestCase):
    def test_194m_baseline_builds(self):
        cfg = kimi_linear_194m_baseline()
        self.assertIsNotNone(cfg.model_spec)
        self.assertEqual(cfg.model_spec.flavor, "kimi_linear_194m_baseline")
        # LR from paper Table 2
        self.assertAlmostEqual(
            cfg.optimizer.param_groups[0].optimizer_kwargs["lr"], 2.99e-3, places=5
        )

    def test_528m_block_attn_res_builds(self):
        cfg = kimi_linear_528m_block_attn_res()
        self.assertEqual(cfg.model_spec.flavor, "kimi_linear_528m_block_attn_res")
        self.assertAlmostEqual(
            cfg.optimizer.param_groups[0].optimizer_kwargs["lr"], 2.02e-3, places=5
        )


class TestFlopsFollowActivatedParams(unittest.TestCase):
    """The MoE parameter buckets, checked against the released figures.

    K3 activates 104.2B of 2.78T. Every bucket pattern in
    ``get_nparams_and_flops`` once failed to match a single parameter name, which
    sent all routed experts into the dense bucket -- so the reported number was
    the model's TOTAL, not its activated cost.
    """

    def test_k3_shaped_flavor_matches_the_released_activated_count(self):
        from torchtitan.models.kimi_k3 import model_registry

        spec = model_registry("kimi_linear_2p8t_block_attn_res").model
        with torch.device("meta"):
            model = spec.build()
        n_params, flops = spec.get_nparams_and_flops(model, seq_len=4096)
        self.assertAlmostEqual(n_params / 1e12, 2.78, places=1)
        # 6 * 104.2e9 = 625e9 for the linear term; MLA, KDA and the AttnRes
        # reads add a few percent. Counting all 896 experts gives ~16.7e12.
        self.assertGreater(flops, 6 * 100e9)
        self.assertLess(flops, 6 * 115e9)


if __name__ == "__main__":
    unittest.main()
