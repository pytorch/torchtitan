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
from torchtitan.experiments.flex_shard.grad_norm import (
    install_flex_shard_grad_norm_clipping,
)


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

    def test_16b_config_preserves_original_compile_ep_ac_knobs(self):
        config = flex_shard_deepseek_v3_16b()

        self.assertTrue(config.compile.enable)
        self.assertEqual(config.compile.components, ["loss"])
        self.assertEqual(config.parallelism.expert_parallel_degree, 8)
        self.assertEqual(config.activation_checkpoint.mode, "selective")

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
