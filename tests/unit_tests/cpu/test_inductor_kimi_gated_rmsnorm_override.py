# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import spmd_types as spmd
import torch
import torch.nn.functional as F

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.config.override import _REGISTRY
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_param_placement,
)
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.kda import KimiGatedRMSNorm
from torchtitan.models.kimi_k3.sharding import set_kimi_k3_sharding_config
from torchtitan.overrides.inductor_gated_rmsnorm import (
    inductor_kimi_gated_rmsnorm,
    InductorGatedRMSNorm,
)
from torchtitan.protocols.sharding import ShardingConfig


_OVERRIDE_TARGET = (
    "torchtitan.overrides.inductor_gated_rmsnorm." "inductor_kimi_gated_rmsnorm"
)
_KIMI_GATED_RMSNORM_OVERRIDE = _REGISTRY[_OVERRIDE_TARGET]


class TestInductorGatedRMSNormOverride(unittest.TestCase):
    def setUp(self):
        _REGISTRY.setdefault(_OVERRIDE_TARGET, _KIMI_GATED_RMSNORM_OVERRIDE)

    def test_override_replaces_all_kimi_gated_rmsnorm_modules(self):
        config = model_registry("debugmodel", attn_backend="flex")
        set_kimi_k3_sharding_config(config, enable_sp=True, enable_ep=False)
        num_gated_norms = len(list(config.traverse(KimiGatedRMSNorm.Config)))

        replacements = apply_overrides(
            OverrideConfig(imports=[_OVERRIDE_TARGET]),
            config,
        )

        self.assertGreater(num_gated_norms, 0)
        self.assertEqual(len(replacements), num_gated_norms)
        self.assertEqual(
            len(list(config.traverse(InductorGatedRMSNorm.Config))),
            num_gated_norms,
        )

    def test_config_is_replaced_without_changing_state_dict(self):
        stock_config = KimiGatedRMSNorm.Config(
            dim=128,
            eps=1e-5,
            param_init={"weight": torch.nn.init.ones_},
        )

        replacement = inductor_kimi_gated_rmsnorm(stock_config)

        self.assertIsInstance(replacement, InductorGatedRMSNorm.Config)
        self.assertEqual(replacement.dim, stock_config.dim)
        self.assertEqual(replacement.eps, stock_config.eps)
        self.assertIs(replacement.activation_fn, torch.sigmoid)
        self.assertIs(replacement.param_init, stock_config.param_init)
        self.assertEqual(
            list(replacement.build().state_dict()),
            list(stock_config.build().state_dict()),
        )

    def test_override_adds_local_compute_region_for_sharded_norm(self):
        weight = dense_param_placement(tp=spmd.R)
        activation = attention_activation_placement()
        input_shardings = {
            "x_THV": activation,
            "gate_THV": activation,
        }
        sharding = ShardingConfig(
            state_shardings={"weight": weight},
            in_src_shardings=input_shardings,
            in_dst_shardings=input_shardings,
            out_src_shardings=activation,
            out_dst_shardings=activation,
        )
        stock_config = KimiGatedRMSNorm.Config(
            dim=128,
            sharding_config=sharding,
        )

        replacement = inductor_kimi_gated_rmsnorm(stock_config)

        self.assertIsNotNone(replacement.sharding_config)
        assert replacement.sharding_config is not None
        self.assertTrue(replacement.sharding_config.local_spmd)
        self.assertEqual(
            replacement.sharding_config.in_src_shardings,
            {"x_THV": activation, "gate_THV": activation},
        )
        self.assertEqual(replacement.sharding_config.out_src_shardings, activation)

    def test_override_requires_activation_sharding_contracts(self):
        weight = dense_param_placement(tp=spmd.R)
        config = KimiGatedRMSNorm.Config(
            dim=128,
            sharding_config=ShardingConfig(
                state_shardings={"weight": weight},
            ),
        )

        with self.assertRaisesRegex(ValueError, "sharding contracts"):
            inductor_kimi_gated_rmsnorm(config)

    def test_decorated_forward_preserves_stock_implementation(self):
        config = KimiGatedRMSNorm.Config(dim=128, eps=1e-5)
        stock = config.build()
        compiled = inductor_kimi_gated_rmsnorm(config).build()
        eager_forward = type(compiled)._compiled_forward._torchdynamo_orig_callable

        with torch.no_grad():
            weight = torch.randn(128)
            stock.weight.copy_(weight)
            compiled.weight.copy_(weight)

        input = torch.randn(4, 3, 128)
        gate = torch.randn_like(input)
        torch.testing.assert_close(
            eager_forward(compiled, input, gate),
            stock(input, gate),
        )

    def test_configurable_unary_activation(self):
        config = KimiGatedRMSNorm.Config(dim=128, eps=1e-5)
        input = torch.randn(4, 3, 128)
        gate = torch.randn_like(input)

        for activation_fn in (F.silu, F.relu):
            with self.subTest(activation_fn=activation_fn.__name__):
                compiled = inductor_kimi_gated_rmsnorm(
                    config,
                    activation_fn=activation_fn,
                ).build()
                with torch.no_grad():
                    compiled.weight.normal_()
                eager_forward = type(
                    compiled
                )._compiled_forward._torchdynamo_orig_callable
                expected = F.rms_norm(
                    input.float(),
                    (input.shape[-1],),
                    compiled.weight.float(),
                    compiled.eps,
                ) * activation_fn(gate.float())
                torch.testing.assert_close(
                    eager_forward(compiled, input, gate),
                    expected.to(input.dtype),
                )


if __name__ == "__main__":
    unittest.main()
