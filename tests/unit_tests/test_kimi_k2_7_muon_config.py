# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.tensor import Shard
from torchtitan.components.distributed_optimizers.muon import (
    BatchedMatrixComputeView,
    MuonComputeSharding,
    Owned,
)
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.models.kimi_k2_7.config_registry import kimi_k2_5, moonlight_16b_a3b


class _KimiMuonConfigTests:
    config_factory = None
    num_layers = 0
    num_heads = 0
    attention_projections: tuple[str, ...] = ()
    owned_attention_projections: frozenset[str] = frozenset()

    @classmethod
    def setUpClass(cls):
        assert cls.config_factory is not None
        cls.config = cls.config_factory()
        assert cls.config.model_spec is not None
        with torch.device("meta"):
            cls.model = cls.config.model_spec.model.build()

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def test_parameter_routing(self):
        optimizer_config = self.config.optimizer
        impl_kwargs = OptimizersContainer._build_impl_kwargs(optimizer_config)
        groups_by_optimizer, _ = OptimizersContainer._build_param_groups(
            self.model,
            optimizer_config.param_groups,
            impl_kwargs,
        )
        model_names = set(dict(self.model.named_parameters()))
        expected_muon_names = set()
        suffix_counts = (
            *(
                (f".attention.{projection}.weight", self.num_layers)
                for projection in self.attention_projections
            ),
            (".feed_forward.w1.weight", 1),
            (".feed_forward.w2.weight", 1),
            (".feed_forward.w3.weight", 1),
            (
                ".moe.routed_experts.inner_experts.w1_EFD",
                self.num_layers - 1,
            ),
            (
                ".moe.routed_experts.inner_experts.w2_EDF",
                self.num_layers - 1,
            ),
            (
                ".moe.routed_experts.inner_experts.w3_EFD",
                self.num_layers - 1,
            ),
            (".moe.router.gate.weight", self.num_layers - 1),
            (".moe.shared_experts.w1.weight", self.num_layers - 1),
            (".moe.shared_experts.w2.weight", self.num_layers - 1),
            (".moe.shared_experts.w3.weight", self.num_layers - 1),
        )
        for suffix, count in suffix_counts:
            names = {name for name in model_names if name.endswith(suffix)}
            self.assertEqual(len(names), count, suffix)
            expected_muon_names.update(names)

        muon_groups = groups_by_optimizer["DistributedMuon"]
        muon_names = {name for group in muon_groups for name in group["param_names"]}
        adamw_names = {
            name
            for group in groups_by_optimizer["AdamW"]
            for name in group["param_names"]
        }
        self.assertEqual(muon_names, expected_muon_names)
        self.assertEqual(adamw_names, model_names - expected_muon_names)
        self.assertEqual(
            {group["adjust_lr_fn"] for group in muon_groups},
            {"match_rms_adamw"},
        )

        representative_suffixes = (
            *(
                f".attention.{projection}.weight"
                for projection in self.attention_projections
            ),
            ".feed_forward.w1.weight",
            ".moe.router.gate.weight",
            ".moe.shared_experts.w1.weight",
            ".moe.routed_experts.inner_experts.w1_EFD",
        )
        group_by_suffix = {
            suffix: next(
                group
                for group in muon_groups
                if group["param_names"][0].endswith(suffix)
            )
            for suffix in representative_suffixes
        }
        per_head = MuonComputeSharding(
            view_before_placement=BatchedMatrixComputeView(
                num_matrices=self.num_heads,
            ),
            placement=Shard(0),
        )
        expected_sharding = {
            f".attention.{projection}.weight": (
                MuonComputeSharding(placement=Owned())
                if projection in self.owned_attention_projections
                else per_head
            )
            for projection in self.attention_projections
        }
        expected_sharding.update(
            {
                ".feed_forward.w1.weight": MuonComputeSharding(placement=Owned()),
                ".moe.router.gate.weight": MuonComputeSharding(placement=Owned()),
                ".moe.shared_experts.w1.weight": MuonComputeSharding(placement=Owned()),
                ".moe.routed_experts.inner_experts.w1_EFD": MuonComputeSharding(
                    placement=Shard(0)
                ),
            }
        )
        for suffix, group in group_by_suffix.items():
            self.assertEqual(group["compute_sharding"], expected_sharding[suffix])


class TestKimiK25MuonConfig(_KimiMuonConfigTests, unittest.TestCase):
    config_factory = staticmethod(kimi_k2_5)
    num_layers = 61
    num_heads = 64
    attention_projections = ("wq_a", "wq_b", "wkv_a", "wkv_b", "wo")
    owned_attention_projections = frozenset(("wq_a", "wkv_a", "wo"))


class TestMoonlightMuonConfig(_KimiMuonConfigTests, unittest.TestCase):
    config_factory = staticmethod(moonlight_16b_a3b)
    num_layers = 27
    num_heads = 16
    attention_projections = ("wq", "wkv_a", "wkv_b", "wo")
    owned_attention_projections = frozenset(("wkv_a", "wo"))


if __name__ == "__main__":
    unittest.main()
