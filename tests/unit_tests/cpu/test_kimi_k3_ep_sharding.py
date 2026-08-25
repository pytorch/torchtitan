# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The expert-parallel declaration, on CPU.

_set_sharding_config runs on the config tree, so what it declares can be read
back without a mesh or a device. The one thing worth pinning is enable_sp: with
EP on, the TP axis becomes a token axis inside the MoE region, so declaring the
sequence-parallel layouts off enable_sp alone asks for a redistribution DTensor
rejects. Nothing in a shape check would catch that -- it fails at the first
step, on a mesh, with a placement error that does not name this call site.
"""

from __future__ import annotations

import unittest

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName


def _moe_configs(*, enable_ep: bool, enable_tp: bool) -> list:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_text

    model = kimi_k3_debugmodel_text().model_spec.model
    model._set_sharding_config(enable_ep=enable_ep, enable_tp=enable_tp)
    return [layer.moe for layer in model.layers if layer.moe is not None]


class TestKimiK3ExpertParallelSharding(unittest.TestCase):
    def test_plain_data_parallel_declares_nothing(self):
        """A run with neither axis must leave the tree untouched."""
        for moe in _moe_configs(enable_ep=False, enable_tp=False):
            self.assertIsNone(moe.sharding_config)

    def test_routed_experts_shard_on_the_expert_dim(self):
        for moe in _moe_configs(enable_ep=True, enable_tp=False):
            shardings = moe.routed_experts.inner_experts.sharding_config.state_shardings
            self.assertEqual(
                set(shardings), {"w1_EFD", "w2_EDF", "w3_EFD"}
            )
            for name, layout in shardings.items():
                self.assertEqual(
                    layout.axis_types[MeshAxisName.EP], spmd.S(0), msg=name
                )

    def test_sequence_parallel_is_derived_from_both_axes(self):
        """EP alone leaves the TP axis out; EP with TP declares it."""
        ep_only = _moe_configs(enable_ep=True, enable_tp=False)
        ep_and_tp = _moe_configs(enable_ep=True, enable_tp=True)
        self.assertEqual(len(ep_only), len(ep_and_tp))
        self.assertTrue(ep_only)
        for a, b in zip(ep_only, ep_and_tp):
            a_tp = a.sharding_config.in_dst_shardings["x_TD"].axis_types[
                MeshAxisName.TP
            ]
            b_tp = b.sharding_config.in_dst_shardings["x_TD"].axis_types[
                MeshAxisName.TP
            ]
            self.assertNotEqual(a_tp, b_tp)


if __name__ == "__main__":
    unittest.main()
