# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.config import CompileConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.flex_shard.deepseek_v3.config_registry import (
    flex_shard_deepseek_v3_16b,
)
from torchtitan.experiments.flex_shard.deepseek_v3.parallelize import (
    _validate_supported_parallelisms,
)
from torchtitan.experiments.flex_shard.deepseek_v3.placement_policy import (
    DeepSeekV3FlexShardPolicy,
)
from torchtitan.experiments.flex_shard.example.owned import GroupedOwned
from torchtitan.experiments.flex_shard.example.shard import Shard
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    MixedPrecisionPolicy,
)
from torchtitan.experiments.flex_shard.grad_norm import (
    install_flex_shard_grad_norm_clipping,
)


class _FakeMesh:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class TestFlexShardDeepSeekV3Config(TestCase):
    def _parallel_dims(self, **kwargs):
        values = dict(
            pp_enabled=False,
            tp_enabled=False,
            cp_enabled=False,
            dp_replicate_enabled=False,
        )
        values.update(kwargs)
        return SimpleNamespace(**values)

    def _training(self, **kwargs):
        values = dict(enable_cpu_offload=False)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def _build_buckets(self, policy=None):
        dp_mesh = _FakeMesh(4)
        efsdp_mesh = _FakeMesh(2)
        model = SimpleNamespace(layers={"0": object()})
        buckets = (policy or DeepSeekV3FlexShardPolicy()).build_buckets(
            model,
            dp_mesh=dp_mesh,
            efsdp_mesh=efsdp_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
            reshard_after_forward=True,
            reshard_last=False,
        )
        return buckets, dp_mesh, efsdp_mesh

    def _bucket_with_pattern(self, buckets, pattern):
        matches = [bucket for bucket in buckets if pattern in bucket.patterns]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def _expert_params(self):
        return [
            (
                "layers.0.moe.experts.w1",
                torch.nn.Parameter(torch.empty(4, 2, 3)),
            ),
            (
                "layers.0.moe.experts.w2",
                torch.nn.Parameter(torch.empty(4, 3, 2)),
            ),
            (
                "layers.0.moe.experts.w3",
                torch.nn.Parameter(torch.empty(4, 2, 3)),
            ),
        ]

    def test_16b_config_preserves_original_compile_ep_ac_knobs(self):
        config = flex_shard_deepseek_v3_16b()

        self.assertTrue(config.compile.enable)
        self.assertEqual(config.compile.components, ["loss"])
        self.assertEqual(config.parallelism.expert_parallel_degree, 8)
        self.assertEqual(config.activation_checkpoint.mode, "selective")

    def test_default_policy_uses_grouped_owned_for_routed_experts(self):
        buckets, _, efsdp_mesh = self._build_buckets()
        expert_bucket = self._bucket_with_pattern(
            buckets, "layers.0.*moe.experts.*"
        )

        placements = expert_bucket.placement_fn(self._expert_params(), efsdp_mesh)

        grouped_owned = placements["layers.0.moe.experts.w1"][0]
        self.assertIsInstance(grouped_owned, GroupedOwned)
        for fqn, _ in self._expert_params():
            self.assertIs(placements[fqn][0], grouped_owned)
        self.assertIs(expert_bucket.mesh, efsdp_mesh)

    def test_routed_experts_can_use_shard_baseline(self):
        buckets, _, efsdp_mesh = self._build_buckets(
            DeepSeekV3FlexShardPolicy(routed_experts="shard")
        )
        expert_bucket = self._bucket_with_pattern(
            buckets, "layers.0.*moe.experts.*"
        )

        placements = expert_bucket.placement_fn(self._expert_params(), efsdp_mesh)

        for fqn, _ in self._expert_params():
            self.assertEqual(placements[fqn], (Shard(0),))
        self.assertIs(expert_bucket.mesh, efsdp_mesh)

    def test_common_and_output_params_stay_sharded(self):
        buckets, dp_mesh, _ = self._build_buckets()
        common_bucket = self._bucket_with_pattern(
            buckets, "layers.0.*attention.*"
        )
        output_bucket = self._bucket_with_pattern(buckets, "lm_head.*")

        common_params = [
            (
                "layers.0.attention.wq.weight",
                torch.nn.Parameter(torch.empty(2, 2)),
            ),
            (
                "layers.0.moe.router.weight",
                torch.nn.Parameter(torch.empty(2, 2)),
            ),
            (
                "layers.0.moe.shared_experts.w1.weight",
                torch.nn.Parameter(torch.empty(2, 2)),
            ),
        ]
        output_params = [
            ("lm_head.weight", torch.nn.Parameter(torch.empty(2, 2))),
        ]

        common_placements = common_bucket.placement_fn(common_params, dp_mesh)
        output_placements = output_bucket.placement_fn(output_params, dp_mesh)

        for fqn, _ in common_params:
            self.assertEqual(common_placements[fqn], (Shard(0),))
        self.assertEqual(output_placements["lm_head.weight"], (Shard(0),))
        self.assertIs(common_bucket.mesh, dp_mesh)
        self.assertIs(output_bucket.mesh, dp_mesh)

    def test_validation_allows_loss_compile(self):
        _validate_supported_parallelisms(
            parallel_dims=self._parallel_dims(),
            training=self._training(),
            compile_config=CompileConfig(enable=True, components=["loss"]),
        )

    def test_validation_rejects_unsupported_main_path_features(self):
        unsupported_parallel_dims = [
            ("PP", dict(pp_enabled=True)),
            ("TP", dict(tp_enabled=True)),
            ("CP", dict(cp_enabled=True)),
            ("HSDP", dict(dp_replicate_enabled=True)),
        ]
        for feature_name, kwargs in unsupported_parallel_dims:
            with self.subTest(feature_name=feature_name):
                with self.assertRaises(NotImplementedError):
                    _validate_supported_parallelisms(
                        parallel_dims=self._parallel_dims(**kwargs),
                        training=self._training(),
                        compile_config=CompileConfig(enable=True, components=["loss"]),
                    )

        with self.assertRaises(NotImplementedError):
            _validate_supported_parallelisms(
                parallel_dims=self._parallel_dims(),
                training=self._training(enable_cpu_offload=True),
                compile_config=CompileConfig(enable=True, components=["loss"]),
            )

        with self.assertRaises(NotImplementedError):
            _validate_supported_parallelisms(
                parallel_dims=self._parallel_dims(),
                training=self._training(),
                compile_config=CompileConfig(enable=True, components=["model"]),
            )

    def test_flex_shard_grad_norm_adapter_handles_local_shards(self):
        original_clip_grad_norm = dist_utils.clip_grad_norm_
        try:
            install_flex_shard_grad_norm_clipping()
            param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
            param.grad = torch.tensor([3.0, 4.0])
            param._placements = ("test",)
            param._mesh = "test"

            total_norm = dist_utils.clip_grad_norm_(
                [param],
                max_norm=1.0,
                ep_enabled=True,
            )

            self.assertEqual(total_norm, torch.tensor(5.0))
            self.assertEqual(param.grad, torch.tensor([0.6, 0.8]))
        finally:
            dist_utils.clip_grad_norm_ = original_clip_grad_norm


if __name__ == "__main__":
    run_tests()
