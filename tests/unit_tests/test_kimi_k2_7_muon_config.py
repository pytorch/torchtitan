# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.tensor import Shard
from torchtitan.components.distributed_optimizers.muon import Owned
from torchtitan.components.distributed_optimizers.muon_parameter_prep import (
    BatchedMatrixComputeView,
    MuonComputeSharding,
)
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.models.kimi_k2_7.config_registry import (
    kimi_k2_5_muon,
)


class TestKimiK25MuonConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = kimi_k2_5_muon()
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
        for suffix, count in (
            (".attention.wq_a.weight", 61),
            (".attention.wq_b.weight", 61),
            (".attention.wkv_a.weight", 61),
            (".attention.wkv_b.weight", 61),
            (".moe.routed_experts.inner_experts.w1_EFD", 60),
            (".moe.routed_experts.inner_experts.w2_EDF", 60),
            (".moe.routed_experts.inner_experts.w3_EFD", 60),
        ):
            names = {name for name in model_names if name.endswith(suffix)}
            self.assertEqual(len(names), count, suffix)
            expected_muon_names.update(names)

        muon_groups = groups_by_optimizer["DistributedMuon"]
        muon_names = {
            name for group in muon_groups for name in group["param_names"]
        }
        adamw_names = {
            name
            for group in groups_by_optimizer["AdamW"]
            for name in group["param_names"]
        }
        self.assertEqual(len(muon_names), 424)
        self.assertEqual(muon_names, expected_muon_names)
        self.assertEqual(adamw_names, model_names - expected_muon_names)
        self.assertFalse(muon_names & adamw_names)
        self.assertTrue(
            {
                name
                for name in model_names
                if name.endswith(".attention.wo.weight")
            }
            <= adamw_names
        )

        group_by_suffix = {
            suffix: next(
                group
                for group in muon_groups
                if group["param_names"][0].endswith(suffix)
            )
            for suffix in (
                ".attention.wq_a.weight",
                ".attention.wq_b.weight",
                ".attention.wkv_a.weight",
                ".attention.wkv_b.weight",
                ".moe.routed_experts.inner_experts.w1_EFD",
            )
        }
        per_head = MuonComputeSharding(
            view_before_placement=BatchedMatrixComputeView(
                num_matrices=64,
                matrices_flattened_into_dim=0,
            ),
            placement=Shard(0),
        )
        expected_sharding = {
            ".attention.wq_a.weight": MuonComputeSharding(placement=Owned()),
            ".attention.wq_b.weight": per_head,
            ".attention.wkv_a.weight": MuonComputeSharding(placement=Owned()),
            ".attention.wkv_b.weight": per_head,
            ".moe.routed_experts.inner_experts.w1_EFD": MuonComputeSharding(
                placement=Shard(0)
            ),
        }
        for suffix, group in group_by_suffix.items():
            self.assertEqual(group["compute_sharding"], expected_sharding[suffix])

    def test_bucket_and_parallelism_config(self):
        optimizer_config = self.config.optimizer
        bucket_configs = optimizer_config.optimizer_init_kwargs["DistributedMuon"][
            "bucket_configs"
        ]
        self.assertEqual(len(bucket_configs), 61)
        for layer_id, bucket in enumerate(bucket_configs):
            prefix = f"layers.{layer_id}"
            expected = tuple(
                f"{prefix}.attention.{projection}.weight"
                for projection in ("wq_a", "wq_b", "wkv_a", "wkv_b")
            )
            if layer_id:
                expected += tuple(
                    f"{prefix}.moe.routed_experts.inner_experts.{projection}"
                    for projection in ("w1_EFD", "w2_EDF", "w3_EFD")
                )
            self.assertEqual(bucket.name, prefix)
            self.assertEqual(bucket.patterns, expected)
            self.assertEqual(bucket.mesh_axes, ("dp_shard",))
            self.assertEqual(
                set(bucket.owner_rank_by_fqn),
                {
                    f"{prefix}.attention.wq_a.weight",
                    f"{prefix}.attention.wkv_a.weight",
                },
            )
            self.assertTrue(
                all(rank in range(64) for rank in bucket.owner_rank_by_fqn.values())
            )

        parallelism = self.config.parallelism
        self.assertEqual(parallelism.data_parallel_replicate_degree, 1)
        self.assertEqual(parallelism.data_parallel_shard_degree, 64)
        self.assertEqual(parallelism.expert_parallel_degree, 8)
        self.assertEqual(parallelism.tensor_parallel_degree, 1)
        self.assertEqual(parallelism.context_parallel_degree, 1)
        self.assertEqual(parallelism.pipeline_parallel_degree, 1)
        self.assertFalse(parallelism.enable_sequence_parallel)
        self.assertEqual(parallelism.spmd_backend, "spmd_types")


if __name__ == "__main__":
    unittest.main()
