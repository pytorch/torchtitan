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
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_param_placement,
)
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.kda import KimiGatedRMSNorm
from torchtitan.overrides.kimi_gated_rmsnorm import (
    triton_kimi_gated_rmsnorm,
    TritonKimiGatedRMSNorm,
)
from torchtitan.protocols.sharding import ShardingConfig


_OVERRIDE_TARGET = "torchtitan.overrides.kimi_gated_rmsnorm.triton_kimi_gated_rmsnorm"
_KIMI_GATED_RMSNORM_OVERRIDE = _REGISTRY[_OVERRIDE_TARGET]


class TestTritonKimiGatedRMSNormOverride(unittest.TestCase):
    def setUp(self):
        _REGISTRY.setdefault(_OVERRIDE_TARGET, _KIMI_GATED_RMSNORM_OVERRIDE)

    def test_override_replaces_all_kimi_gated_rmsnorm_modules(self):
        config = model_registry("debugmodel", attn_backend="flex")
        num_gated_norms = len(list(config.traverse(KimiGatedRMSNorm.Config)))

        replacements = apply_overrides(
            OverrideConfig(imports=[_OVERRIDE_TARGET]),
            config,
        )

        self.assertGreater(num_gated_norms, 0)
        self.assertEqual(len(replacements), num_gated_norms)
        self.assertEqual(
            len(list(config.traverse(TritonKimiGatedRMSNorm.Config))),
            num_gated_norms,
        )

    def test_config_is_replaced_without_changing_state_dict(self):
        stock_config = KimiGatedRMSNorm.Config(
            dim=128,
            eps=1e-5,
            param_init={"weight": torch.nn.init.ones_},
        )

        replacement = triton_kimi_gated_rmsnorm(stock_config)

        self.assertIsInstance(replacement, TritonKimiGatedRMSNorm.Config)
        self.assertEqual(replacement.dim, stock_config.dim)
        self.assertEqual(replacement.eps, stock_config.eps)
        self.assertIs(replacement.param_init, stock_config.param_init)
        self.assertEqual(
            list(replacement.build().state_dict()),
            list(stock_config.build().state_dict()),
        )

    def test_override_adds_local_compute_region_for_sharded_norm(self):
        weight = dense_param_placement(tp=spmd.R)
        sharding = ShardingConfig(state_shardings={"weight": weight})
        stock_config = KimiGatedRMSNorm.Config(
            dim=128,
            sharding_config=sharding,
        )

        replacement = triton_kimi_gated_rmsnorm(stock_config)

        self.assertIsNotNone(replacement.sharding_config)
        assert replacement.sharding_config is not None
        activation = attention_activation_placement()
        self.assertTrue(replacement.sharding_config.local_spmd)
        self.assertEqual(
            replacement.sharding_config.in_src_shardings,
            {"x_THV": activation, "gate_THV": activation},
        )
        self.assertEqual(replacement.sharding_config.out_src_shardings, activation)

    def test_cpu_fallback_matches_stock_module(self):
        config = KimiGatedRMSNorm.Config(dim=128, eps=1e-5)
        stock = config.build()
        fused = triton_kimi_gated_rmsnorm(config).build()
        with torch.no_grad():
            weight = torch.randn(128)
            stock.weight.copy_(weight)
            fused.weight.copy_(weight)

        input = torch.randn(4, 3, 128)
        gate = torch.randn_like(input)

        torch.testing.assert_close(fused(input, gate), stock(input, gate))


if __name__ == "__main__":
    unittest.main()
