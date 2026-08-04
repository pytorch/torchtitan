# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.tensor import Shard
from torchtitan.components.distributed_optimizers.bucketed_redistribution import (
    BucketConfig,
    assign_balanced_owners,
)
from torchtitan.components.distributed_optimizers.muon import Owned
from torchtitan.components.distributed_optimizers.muon_parameter_prep import (
    BatchedMatrixComputeView,
    MuonComputeSharding,
)
from torchtitan.components.optimizer import (
    OptimizersContainer,
    register_moe_load_balancing_hook,
)
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b_distributed_muon,
)


class TestDeepSeekV3DistributedMuonConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = deepseek_v3_16b_distributed_muon()
        assert cls.config.model_spec is not None
        with torch.device("meta"):
            cls.model = cls.config.model_spec.model.build()

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def test_balanced_owner_assignment(self):
        self.assertEqual(
            assign_balanced_owners(
                [("a", "b"), ("c",)],
                {"a": 8, "b": 4, "c": 4},
                num_ranks=2,
                initial_memory_by_rank=(0, 4),
            ),
            ({"a": 0, "b": 1}, {"c": 0}),
        )

        owners = {"a": 0}
        spec = BucketConfig(
            patterns=("a",),
            owner_rank_by_fqn=owners,
            mesh_axes=("dp_shard",),
        )
        owners["a"] = 1
        self.assertEqual(spec.owner_rank_by_fqn, {"a": 0})

    def test_parameter_routing(self):
        optimizer_config = self.config.optimizer
        impl_kwargs = OptimizersContainer._build_impl_kwargs(optimizer_config)
        groups_by_optimizer, _ = OptimizersContainer._build_param_groups(
            self.model,
            optimizer_config.param_groups,
            impl_kwargs,
        )

        model_names = set(dict(self.model.named_parameters()))
        self.assertEqual(len(model_names), 377)

        expected_muon_names = set()
        for suffix, count in (
            (".attention.wq.weight", 27),
            (".attention.wkv_a.weight", 27),
            (".attention.wkv_b.weight", 27),
            (".moe.routed_experts.inner_experts.w1_EFD", 26),
            (".moe.routed_experts.inner_experts.w2_EDF", 26),
            (".moe.routed_experts.inner_experts.w3_EFD", 26),
        ):
            names = {name for name in model_names if name.endswith(suffix)}
            self.assertEqual(len(names), count, suffix)
            expected_muon_names.update(names)

        muon_groups = groups_by_optimizer["DistributedMuon"]
        muon_names = {
            name for group in muon_groups for name in group["param_names"]
        }
        self.assertEqual(len(muon_names), 159)
        self.assertEqual(muon_names, expected_muon_names)

        adamw_names = {
            name
            for group in groups_by_optimizer["AdamW"]
            for name in group["param_names"]
        }
        self.assertEqual(len(adamw_names), 218)
        self.assertEqual(adamw_names, model_names - expected_muon_names)
        self.assertEqual(len(muon_names | adamw_names), 377)
        self.assertFalse(muon_names & adamw_names)
        wo_names = {
            name for name in model_names if name.endswith(".attention.wo.weight")
        }
        self.assertEqual(len(wo_names), 27)
        self.assertTrue(wo_names <= adamw_names)

        groups_by_suffix = {
            suffix: next(
                group
                for group in muon_groups
                if group["param_names"][0].endswith(suffix)
            )
            for suffix in (
                ".attention.wq.weight",
                ".attention.wkv_a.weight",
                ".attention.wkv_b.weight",
                ".moe.routed_experts.inner_experts.w1_EFD",
            )
        }
        expected_compute_sharding = {
            ".attention.wq.weight": MuonComputeSharding(
                view_before_placement=BatchedMatrixComputeView(
                    num_matrices=16,
                    matrices_flattened_into_dim=0,
                ),
                placement=Shard(0),
            ),
            ".attention.wkv_a.weight": MuonComputeSharding(placement=Owned()),
            ".attention.wkv_b.weight": MuonComputeSharding(
                view_before_placement=BatchedMatrixComputeView(
                    num_matrices=16,
                    matrices_flattened_into_dim=0,
                ),
                placement=Shard(0),
            ),
            ".moe.routed_experts.inner_experts.w1_EFD": MuonComputeSharding(
                placement=Shard(0)
            ),
        }
        for suffix, group in groups_by_suffix.items():
            self.assertEqual(
                group["compute_sharding"],
                expected_compute_sharding[suffix],
            )

    def test_bucket_and_parallelism_config(self):
        optimizer_config = self.config.optimizer
        bucket_configs = optimizer_config.optimizer_init_kwargs["DistributedMuon"][
            "bucket_configs"
        ]
        self.assertEqual(
            set(optimizer_config.optimizer_init_kwargs["DistributedMuon"]),
            {"bucket_configs"},
        )
        self.assertEqual(
            [config.name for config in bucket_configs],
            [f"layers.{layer_id}" for layer_id in range(27)],
        )
        for layer_id, config in enumerate(bucket_configs):
            prefix = f"layers.{layer_id}"
            expected = tuple(
                f"{prefix}.attention.{projection}.weight"
                for projection in ("wq", "wkv_a", "wkv_b")
            )
            if layer_id:
                expected += tuple(
                    f"{prefix}.moe.routed_experts.inner_experts.{projection}"
                    for projection in ("w1_EFD", "w2_EDF", "w3_EFD")
                )
            self.assertEqual(config.patterns, expected)
            self.assertEqual(config.mesh_axes, ("dp_shard",))
            self.assertEqual(
                config.owner_rank_by_fqn,
                {f"{prefix}.attention.wkv_a.weight": layer_id % 8},
            )

        parallelism = self.config.parallelism
        self.assertEqual(parallelism.data_parallel_replicate_degree, 1)
        self.assertEqual(parallelism.data_parallel_shard_degree, 8)
        self.assertEqual(parallelism.expert_parallel_degree, 4)
        self.assertEqual(parallelism.tensor_parallel_degree, 1)
        self.assertEqual(parallelism.context_parallel_degree, 1)
        self.assertEqual(parallelism.pipeline_parallel_degree, 1)
        self.assertFalse(parallelism.enable_sequence_parallel)
        self.assertEqual(parallelism.spmd_backend, "spmd_types")
        self.assertIs(
            self.config.model_spec.post_optimizer_build_fn,
            register_moe_load_balancing_hook,
        )


if __name__ == "__main__":
    unittest.main()
