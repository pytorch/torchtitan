# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
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
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    MixedPrecisionPolicy,
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

    def _build_buckets(self):
        dp_mesh = _FakeMesh(4)
        efsdp_mesh = _FakeMesh(2)
        model = SimpleNamespace(layers={"0": object()})
        buckets = DeepSeekV3FlexShardPolicy().build_buckets(
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
                "layers.0.moe.experts.w1_EFD",
                torch.nn.Parameter(torch.empty(4, 2, 3)),
            ),
            (
                "layers.0.moe.experts.w2_EDF",
                torch.nn.Parameter(torch.empty(4, 3, 2)),
            ),
            (
                "layers.0.moe.experts.w3_EFD",
                torch.nn.Parameter(torch.empty(4, 2, 3)),
            ),
        ]

    def test_16b_config_preserves_original_compile_ep_ac_knobs(self):
        config = flex_shard_deepseek_v3_16b()

        self.assertTrue(config.compile.enable)
        self.assertEqual(config.compile.components, ["loss"])
        self.assertEqual(config.parallelism.expert_parallel_degree, 8)
        self.assertIsInstance(config.activation_checkpoint, SelectiveAC.Config)

    def test_default_policy_uses_grouped_owned_for_routed_experts(self):
        buckets, _, efsdp_mesh = self._build_buckets()
        expert_bucket = self._bucket_with_pattern(
            buckets, "layers.0.*moe.experts.*"
        )

        placements = expert_bucket.placement_fn(self._expert_params(), efsdp_mesh)

        grouped_owned = placements["layers.0.moe.experts.w1_EFD"][0]
        self.assertIsInstance(grouped_owned, GroupedOwned)
        self.assertEqual(grouped_owned.view_kind, "expert_block")
        for fqn, _ in self._expert_params():
            self.assertIs(placements[fqn][0], grouped_owned)
        self.assertIs(expert_bucket.mesh, efsdp_mesh)

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


if __name__ == "__main__":
    run_tests()
