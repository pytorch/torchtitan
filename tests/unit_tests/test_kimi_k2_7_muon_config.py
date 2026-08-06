# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest

import torch
from torch.distributed.tensor import Shard
from torchtitan.components.distributed_optimizers.flex_optimizer_reshard import (
    assign_balanced_owners,
)
from torchtitan.components.distributed_optimizers.muon import Owned
from torchtitan.components.distributed_optimizers.muon_parameter_prep import (
    BatchedMatrixComputeView,
    MuonComputeSharding,
)
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.kimi_k2_7.config_registry import (
    kimi_k2_5_muon,
    moonlight_16b_a3b_muon,
)


class _KimiMuonConfigTests:
    config_factory = None
    num_layers = 0
    num_heads = 0
    num_owner_ranks = 0
    expert_parallel_degree = 0
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
                matrices_flattened_into_dim=0,
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

    def test_bucket_and_parallelism_config(self):
        optimizer_config = self.config.optimizer
        bucket_configs = optimizer_config.optimizer_init_kwargs["DistributedMuon"][
            "bucket_configs"
        ]
        bucket_layer_ids = ((0,),) + tuple(
            tuple(
                range(
                    first_layer_id,
                    min(first_layer_id + 2, self.num_layers),
                )
            )
            for first_layer_id in range(1, self.num_layers, 2)
        )
        self.assertEqual(len(bucket_configs), len(bucket_layer_ids))
        expected_bucket_patterns = []
        expected_owned_fqns = set()
        for layer_ids, bucket in zip(bucket_layer_ids, bucket_configs, strict=True):
            expected = ()
            expected_owners = set()
            for layer_id in layer_ids:
                prefix = f"layers.{layer_id}"
                attention_fqns = tuple(
                    f"{prefix}.attention.{projection}.weight"
                    for projection in self.attention_projections
                )
                expected += attention_fqns
                expected_owners.update(
                    f"{prefix}.attention.{projection}.weight"
                    for projection in self.owned_attention_projections
                )
                if not layer_id:
                    dense_fqns = tuple(
                        f"{prefix}.feed_forward.{projection}.weight"
                        for projection in ("w1", "w2", "w3")
                    )
                    expected += dense_fqns
                    expected_owners.update(dense_fqns)
                else:
                    expert_fqns = tuple(
                        f"{prefix}.moe.routed_experts.inner_experts.{projection}"
                        for projection in ("w1_EFD", "w2_EDF", "w3_EFD")
                    )
                    router_fqn = f"{prefix}.moe.router.gate.weight"
                    shared_fqns = tuple(
                        f"{prefix}.moe.shared_experts.{projection}.weight"
                        for projection in ("w1", "w2", "w3")
                    )
                    expected += expert_fqns + (router_fqn,) + shared_fqns
                    expected_owners.update((router_fqn, *shared_fqns))
            self.assertEqual(
                bucket.name,
                "layers." + "-".join(map(str, layer_ids)),
            )
            self.assertEqual(bucket.patterns, expected)
            self.assertEqual(bucket.mesh_axes, ("dp_shard",))
            self.assertEqual(
                set(bucket.owner_rank_by_fqn),
                expected_owners,
            )
            expected_bucket_patterns.append(expected)
            expected_owned_fqns.update(expected_owners)
            self.assertTrue(
                all(
                    rank in range(self.num_owner_ranks)
                    for rank in bucket.owner_rank_by_fqn.values()
                )
            )

        parameter_numel_by_fqn = {
            fqn: parameter.numel()
            for fqn, parameter in self.model.named_parameters()
            if fqn in expected_owned_fqns
        }
        self.assertEqual(set(parameter_numel_by_fqn), expected_owned_fqns)
        self.assertEqual(
            tuple(dict(bucket.owner_rank_by_fqn) for bucket in bucket_configs),
            assign_balanced_owners(
                expected_bucket_patterns,
                parameter_numel_by_fqn,
                num_ranks=self.num_owner_ranks,
            ),
        )

        parallelism = self.config.parallelism
        self.assertEqual(parallelism.data_parallel_replicate_degree, 1)
        self.assertEqual(
            parallelism.data_parallel_shard_degree,
            self.num_owner_ranks,
        )
        self.assertEqual(
            parallelism.expert_parallel_degree,
            self.expert_parallel_degree,
        )
        self.assertEqual(parallelism.tensor_parallel_degree, 1)
        self.assertEqual(parallelism.context_parallel_degree, 1)
        self.assertEqual(parallelism.pipeline_parallel_degree, 1)
        self.assertFalse(parallelism.enable_sequence_parallel)
        self.assertEqual(parallelism.spmd_backend, "spmd_types")
        self.assertIsInstance(self.config.activation_checkpoint, FullAC.Config)

    def test_config_is_json_serializable(self):
        json.dumps(self.config.to_dict())


class TestKimiK25MuonConfig(_KimiMuonConfigTests, unittest.TestCase):
    config_factory = staticmethod(kimi_k2_5_muon)
    num_layers = 61
    num_heads = 64
    num_owner_ranks = 64
    expert_parallel_degree = 8
    attention_projections = ("wq_a", "wq_b", "wkv_a", "wkv_b", "wo")
    owned_attention_projections = frozenset(("wq_a", "wkv_a", "wo"))


class TestMoonlightMuonConfig(_KimiMuonConfigTests, unittest.TestCase):
    config_factory = staticmethod(moonlight_16b_a3b_muon)
    num_layers = 27
    num_heads = 16
    num_owner_ranks = 8
    expert_parallel_degree = 4
    attention_projections = ("wq", "wkv_a", "wkv_b", "wo")
    owned_attention_projections = frozenset(("wkv_a", "wo"))


if __name__ == "__main__":
    unittest.main()
