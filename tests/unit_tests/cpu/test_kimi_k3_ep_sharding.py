# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The expert-parallel declaration, on CPU.

_set_sharding_config runs on the config tree, so what it declares can be read
back without a mesh or a device: a plain data-parallel run must leave the tree
untouched, and an expert-parallel run must shard the routed experts on the
expert axis.
"""

from __future__ import annotations

import unittest

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName


def _moe_configs(*, enable_ep: bool) -> list:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    model = kimi_k3_debugmodel().model_spec.model
    
    if enable_ep:
        from torchtitan.models.kimi_k3.sharding import (
            set_expert_parallel_sharding_config,
        )

        set_expert_parallel_sharding_config(model)
    return [layer.moe for layer in model.layers if layer.moe is not None]


class TestKimiK3ExpertParallelSharding(unittest.TestCase):
    def test_plain_data_parallel_declares_nothing(self):
        """A run without expert parallel must leave the tree untouched."""
        for moe in _moe_configs(enable_ep=False):
            self.assertIsNone(moe.sharding_config)

    def test_routed_experts_shard_on_the_expert_dim(self):
        for moe in _moe_configs(enable_ep=True):
            shardings = moe.routed_experts.inner_experts.sharding_config.state_shardings
            self.assertEqual(
                set(shardings), {"w1_EFD", "w2_EDF", "w3_EFD"}
            )
            for name, layout in shardings.items():
                self.assertEqual(
                    layout.axis_types[MeshAxisName.EP], spmd.S(0), msg=name
                )


if __name__ == "__main__":
    unittest.main()
