# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import Counter

import torch
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b_muon,
    deepseek_v3_16b_muon_with_ffn,
    deepseek_v3_debugmodel_muon,
)

_DEBUG_MUON_COHORTS = (
    (".attention.wq.weight", 6, (3072, 256)),
    (".attention.wkv_a.weight", 6, (576, 256)),
    (".attention.wkv_b.weight", 6, (4096, 512)),
    (".moe.routed_experts.inner_experts.w1_EFD", 5, (8, 256, 256)),
    (".moe.routed_experts.inner_experts.w2_EDF", 5, (8, 256, 256)),
    (".moe.routed_experts.inner_experts.w3_EFD", 5, (8, 256, 256)),
)

_16B_BASE_MUON_COHORTS = (
    (".attention.wq.weight", 27, (3072, 2048)),
    (".attention.wkv_a.weight", 27, (576, 2048)),
    (".attention.wkv_b.weight", 27, (4096, 512)),
    (
        ".moe.routed_experts.inner_experts.w1_EFD",
        26,
        (64, 1408, 2048),
    ),
    (
        ".moe.routed_experts.inner_experts.w2_EDF",
        26,
        (64, 2048, 1408),
    ),
    (
        ".moe.routed_experts.inner_experts.w3_EFD",
        26,
        (64, 1408, 2048),
    ),
)

_16B_FFN_MUON_COHORTS = (
    ("layers.0.feed_forward.w1.weight", 1, (10944, 2048)),
    ("layers.0.feed_forward.w2.weight", 1, (2048, 10944)),
    ("layers.0.feed_forward.w3.weight", 1, (10944, 2048)),
    (".moe.shared_experts.w1.weight", 26, (2816, 2048)),
    (".moe.shared_experts.w2.weight", 26, (2048, 2816)),
    (".moe.shared_experts.w3.weight", 26, (2816, 2048)),
)


class TestDeepSeekV3MuonConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.debug_config = deepseek_v3_debugmodel_muon()
        cls.base_config = deepseek_v3_16b_muon()
        cls.ffn_config = deepseek_v3_16b_muon_with_ffn()
        with torch.device("meta"):
            cls.debug_model = cls.debug_config.model_spec.model.build()
            cls.model = cls.base_config.model_spec.model.build()

    @classmethod
    def tearDownClass(cls):
        del cls.debug_model
        del cls.model

    def _assert_routing(
        self,
        config,
        model,
        cohorts,
        *,
        expected_total_params,
        expected_wo_count,
        expected_wo_shape,
        wq_matrix_shape,
        wkv_b_matrix_shape,
    ):
        optimizer_config = config.optimizer
        self.assertEqual(optimizer_config.implementation, "fused")
        impl_kwargs = OptimizersContainer._build_impl_kwargs(optimizer_config)
        groups_by_optimizer, _ = OptimizersContainer._build_param_groups(
            model,
            optimizer_config.param_groups,
            impl_kwargs,
        )

        model_params = dict(model.named_parameters())
        all_names = set(model_params)
        self.assertEqual(len(all_names), expected_total_params)

        expected_muon_names = set()
        for suffix, expected_count, expected_shape in cohorts:
            names = {name for name in all_names if name.endswith(suffix)}
            self.assertEqual(len(names), expected_count, suffix)
            for name in names:
                self.assertEqual(tuple(model_params[name].shape), expected_shape, name)
            expected_muon_names.update(names)

        muon_groups = groups_by_optimizer["Muon"]
        muon_names = {name for group in muon_groups for name in group["param_names"]}
        self.assertEqual(muon_names, expected_muon_names)
        self.assertEqual(len(muon_groups), len(expected_muon_names))

        for group in muon_groups:
            self.assertEqual(len(group["params"]), 1)
            self.assertEqual(len(group["param_names"]), 1)
            self.assertFalse(group["fused"])
            self.assertFalse(group["foreach"])
            name = group["param_names"][0]
            if name.endswith(".attention.wq.weight"):
                self.assertEqual(group["matrix_shape"], wq_matrix_shape)
            elif name.endswith(".attention.wkv_b.weight"):
                self.assertEqual(group["matrix_shape"], wkv_b_matrix_shape)
            else:
                self.assertNotIn("matrix_shape", group)

        adamw_groups = groups_by_optimizer["AdamW"]
        self.assertEqual(len(adamw_groups), 2)
        wo_names = set(adamw_groups[0]["param_names"])
        self.assertEqual(len(wo_names), expected_wo_count)
        for name in wo_names:
            self.assertTrue(name.endswith(".attention.wo.weight"), name)
            self.assertEqual(tuple(model_params[name].shape), expected_wo_shape)

        fallback_names = set(adamw_groups[1]["param_names"])
        self.assertEqual(fallback_names, all_names - expected_muon_names - wo_names)
        for group in adamw_groups:
            self.assertTrue(group["fused"])
            self.assertFalse(group["foreach"])

        assigned_names = [
            name
            for groups in groups_by_optimizer.values()
            for group in groups
            for name in group["param_names"]
        ]
        self.assertEqual(set(assigned_names), all_names)
        self.assertTrue(all(count == 1 for count in Counter(assigned_names).values()))

    def test_debugmodel_attention_and_routed_expert_cohorts(self):
        self._assert_routing(
            self.debug_config,
            self.debug_model,
            _DEBUG_MUON_COHORTS,
            expected_total_params=83,
            expected_wo_count=6,
            expected_wo_shape=(256, 2048),
            wq_matrix_shape=(192, 256),
            wkv_b_matrix_shape=(256, 512),
        )

    def test_16b_attention_and_routed_expert_cohorts(self):
        self._assert_routing(
            self.base_config,
            self.model,
            _16B_BASE_MUON_COHORTS,
            expected_total_params=377,
            expected_wo_count=27,
            expected_wo_shape=(2048, 2048),
            wq_matrix_shape=(192, 2048),
            wkv_b_matrix_shape=(256, 512),
        )

    def test_ffn_cohort_is_opt_in(self):
        self._assert_routing(
            self.ffn_config,
            self.model,
            _16B_BASE_MUON_COHORTS + _16B_FFN_MUON_COHORTS,
            expected_total_params=377,
            expected_wo_count=27,
            expected_wo_shape=(2048, 2048),
            wq_matrix_shape=(192, 2048),
            wkv_b_matrix_shape=(256, 512),
        )


if __name__ == "__main__":
    unittest.main()
