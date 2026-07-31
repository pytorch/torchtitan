# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchtitan.components.distributed_muon import (
    BucketSpec,
    assign_balanced_owners,
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
        spec = BucketSpec(patterns=("a",), owner_rank_by_fqn=owners)
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
            (".attention.wo.weight", 27),
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
        self.assertEqual(len(muon_names), 186)
        self.assertEqual(muon_names, expected_muon_names)

        adamw_names = {
            name
            for group in groups_by_optimizer["AdamW"]
            for name in group["param_names"]
        }
        self.assertEqual(len(adamw_names), 191)
        self.assertEqual(adamw_names, model_names - expected_muon_names)
        self.assertEqual(len(muon_names | adamw_names), 377)
        self.assertFalse(muon_names & adamw_names)

        groups_by_suffix = {
            suffix: next(
                group
                for group in muon_groups
                if group["param_names"][0].endswith(suffix)
            )
            for suffix in (
                ".attention.wq.weight",
                ".attention.wkv_b.weight",
                ".attention.wo.weight",
            )
        }
        self.assertEqual(
            groups_by_suffix[".attention.wq.weight"]["matrix_shape"],
            (192, 2048),
        )
        self.assertEqual(
            groups_by_suffix[".attention.wkv_b.weight"]["matrix_shape"],
            (256, 512),
        )
        self.assertEqual(
            groups_by_suffix[".attention.wo.weight"]["matrix_shape"],
            (2048, 128),
        )
        self.assertEqual(
            groups_by_suffix[".attention.wo.weight"]["matrix_block_dim"],
            1,
        )

    def test_bucket_and_parallelism_config(self):
        optimizer_config = self.config.optimizer
        bucket_specs = optimizer_config.optimizer_init_kwargs["DistributedMuon"][
            "bucket_spec"
        ]
        self.assertEqual(len(bucket_specs), 27)
        self.assertTrue(all(isinstance(spec, BucketSpec) for spec in bucket_specs))
        self.assertEqual(
            [spec.name for spec in bucket_specs],
            [f"layers.{layer_id}" for layer_id in range(27)],
        )
        for layer_id, spec in enumerate(bucket_specs):
            prefix = f"layers.{layer_id}"
            expected = tuple(
                f"{prefix}.attention.{projection}.weight"
                for projection in ("wq", "wkv_a", "wkv_b", "wo")
            )
            if layer_id:
                expected += tuple(
                    f"{prefix}.moe.routed_experts.inner_experts.{projection}"
                    for projection in ("w1_EFD", "w2_EDF", "w3_EFD")
                )
            self.assertEqual(spec.patterns, expected)
            self.assertEqual(
                spec.owner_rank_by_fqn,
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
