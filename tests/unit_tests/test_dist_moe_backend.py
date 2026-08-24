# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the standalone DistMoE TorchTitan model backend."""

import importlib.util
import unittest

import spmd_types as spmd
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.models.common.moe import (
    GroupedExperts,
    RoutedExperts,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.protocols.sharding import ShardingConfig


_DIST_MOE_AVAILABLE = importlib.util.find_spec("dist_moe") is not None

if _DIST_MOE_AVAILABLE:
    from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
        graph_trainer_deepseek_v3_16b_dist_moe_bf16,
    )
    from torchtitan.models.common.dist_moe import (
        dist_moe_config,
        DistMoeBackendConfig,
        DistMoeRoutedExperts,
    )


def _stock_config() -> RoutedExperts.Config:
    layout = SpmdLayout({MeshAxisName.TP: spmd.S(1)})
    return RoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=64,
            hidden_dim=64,
            num_experts=4,
            param_init={
                "w1_EFD": lambda tensor: tensor.fill_(1.0),
                "w2_EDF": lambda tensor: tensor.zero_(),
                "w3_EFD": lambda tensor: tensor.fill_(3.0),
            },
            sharding_config=ShardingConfig(
                state_shardings={
                    "w1_EFD": layout,
                    "w2_EDF": SpmdLayout({MeshAxisName.TP: spmd.S(2)}),
                    "w3_EFD": layout,
                }
            ),
        ),
        token_dispatcher=LocalTokenDispatcher.Config(num_experts=4, top_k=2),
    )


@unittest.skipUnless(_DIST_MOE_AVAILABLE, "requires the optional dist-moe package")
class DistMoeBackendTest(TestCase):
    def test_fuses_initialization_and_sharding(self) -> None:
        config = dist_moe_config(_stock_config())
        self.assertIsInstance(config, DistMoeRoutedExperts.Config)
        module = config.build()
        with torch.no_grad():
            module.init_states()

        self.assertEqual(module.w13[:, 0], torch.ones_like(module.w13[:, 0]))
        self.assertEqual(module.w13[:, 1], 3 * torch.ones_like(module.w13[:, 1]))
        self.assertFalse(hasattr(module, "inner_experts"))

        self.assertIsNotNone(module._sharding_config)
        state = module._sharding_config.state_shardings
        self.assertEqual(set(state), {"w13", "w2_EDF"})
        w13_tp = state["w13"].axis_types[MeshAxisName.TP]
        self.assertIsInstance(w13_tp, spmd.Shard)
        self.assertEqual(w13_tp.dim, 2)

    def test_model_state_dict_uses_stock_keys(self) -> None:
        source = dist_moe_config(_stock_config()).build()
        with torch.no_grad():
            source.w13.copy_(torch.randn_like(source.w13))
            source.w2_EDF.copy_(torch.randn_like(source.w2_EDF))

        state = source.state_dict()
        self.assertEqual(
            set(state),
            {
                "inner_experts.w1_EFD",
                "inner_experts.w2_EDF",
                "inner_experts.w3_EFD",
            },
        )
        self.assertEqual(state["inner_experts.w1_EFD"], source.w13[:, 0])
        self.assertEqual(state["inner_experts.w3_EFD"], source.w13[:, 1])

        target = dist_moe_config(_stock_config()).build()
        target.load_state_dict(state)
        self.assertEqual(target.w13, source.w13)
        self.assertEqual(target.w2_EDF, source.w2_EDF)

    def test_backend_policy_validates_device_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "device_memory_budget_bytes"):
            DistMoeBackendConfig(device_memory_budget_bytes=0)

    def test_graph_trainer_config_replaces_every_moe_layer(self) -> None:
        config = graph_trainer_deepseek_v3_16b_dist_moe_bf16()
        experts = [
            expert_config
            for _fqn, expert_config, _parent, _attr in config.traverse(
                DistMoeRoutedExperts.Config
            )
        ]

        self.assertEqual(len(experts), 26)
        self.assertIsNotNone(config.model_spec.post_parallelize_fn)
        self.assertIsNotNone(config.model_spec.cleanup_fn)
        self.assertTrue(config.compile.require_cudagraph)
        self.assertEqual(config.compile.memory_policy, "full")
        self.assertEqual(
            config.compile.fsdp_contiguous_module_fqns,
            ["layers.*.moe.routed_experts"],
        )


if __name__ == "__main__":
    run_tests()
