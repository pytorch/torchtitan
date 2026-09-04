# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import spmd_types as spmd
import torch

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.config.override import _REGISTRY
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.model import OffsetRMSNorm
from torchtitan.overrides.offset_rmsnorm import (
    triton_offset_rmsnorm,
    TritonOffsetRMSNorm,
)
from torchtitan.protocols.sharding import ShardingConfig


_OVERRIDE_TARGET = "torchtitan.overrides.offset_rmsnorm.triton_offset_rmsnorm"
_OFFSET_RMSNORM_OVERRIDE = _REGISTRY[_OVERRIDE_TARGET]


class TestTritonOffsetRMSNormOverride(unittest.TestCase):
    def setUp(self):
        _REGISTRY.setdefault(_OVERRIDE_TARGET, _OFFSET_RMSNORM_OVERRIDE)

    def test_override_replaces_all_qwen35_offset_norms(self):
        config = model_registry("debugmodel", attn_backend="flex").model
        num_offset_norms = len(list(config.traverse(OffsetRMSNorm.Config)))

        replacements = apply_overrides(
            OverrideConfig(
                imports=["torchtitan.overrides.offset_rmsnorm." "triton_offset_rmsnorm"]
            ),
            config,
        )

        self.assertGreater(num_offset_norms, 0)
        self.assertEqual(len(replacements), num_offset_norms)
        self.assertEqual(
            len(list(config.traverse(TritonOffsetRMSNorm.Config))),
            num_offset_norms,
        )

    def test_config_is_replaced_without_changing_state_dict(self):
        stock_config = OffsetRMSNorm.Config(
            dim=32,
            eps=1e-5,
            param_init={"weight": torch.nn.init.zeros_},
        )

        replacement = triton_offset_rmsnorm(stock_config)

        self.assertIsInstance(replacement, TritonOffsetRMSNorm.Config)
        self.assertEqual(replacement.dim, stock_config.dim)
        self.assertEqual(replacement.eps, stock_config.eps)
        self.assertIs(replacement.param_init, stock_config.param_init)
        self.assertEqual(
            list(replacement.build().state_dict()),
            list(stock_config.build().state_dict()),
        )

    def test_override_adds_local_compute_region_for_sharded_norm(self):
        activation = spmd.SpmdType(
            {"dp": spmd.V, "tp": spmd.I},
            partition_spec=spmd.PartitionSpec("dp", None),
        )
        weight = dense_param_placement(tp=spmd.R)
        sharding = ShardingConfig(
            state_shardings={"weight": weight},
            in_src_shardings={"input": activation},
            out_src_shardings=activation,
        )
        stock_config = OffsetRMSNorm.Config(
            dim=32,
            sharding_config=sharding,
        )

        replacement = triton_offset_rmsnorm(stock_config)

        self.assertIsNotNone(replacement.sharding_config)
        assert replacement.sharding_config is not None
        self.assertIsNotNone(replacement.sharding_config.local_map)
        assert replacement.sharding_config.local_map is not None
        self.assertEqual(
            replacement.sharding_config.local_map.in_grad_placements,
            (activation,),
        )
        self.assertIsNotNone(replacement.weight_grad_sharding)
        assert replacement.weight_grad_sharding is not None
        self.assertEqual(
            replacement.weight_grad_sharding.local_type["tp"],
            spmd.P,
        )

    def test_cpu_fallback_matches_stock_module(self):
        config = OffsetRMSNorm.Config(dim=32, eps=1e-6)
        stock = config.build()
        fused = triton_offset_rmsnorm(config).build()
        with torch.no_grad():
            weight = torch.randn(32)
            stock.weight.copy_(weight)
            fused.weight.copy_(weight)

        input = torch.randn(4, 32)

        torch.testing.assert_close(fused(input), stock(input))


if __name__ == "__main__":
    unittest.main()
