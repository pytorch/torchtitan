# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CPU-only tests verifying FlexShard FX metadata survives serialization.

The actual precompile path (precompile.py) serializes compiled artifacts via
Inductor's serialization, not by pickling GraphModules. These tests verify
that FlexShard's metadata values and op targets are individually picklable,
which is the prerequisite for precompile compatibility.

Usage:
    python -m pytest torchtitan/experiments/flex_shard/tests/test_precompile_roundtrip.py -v
"""

import pickle
import unittest

import torch
from torch.utils.checkpoint import CheckpointPolicy


class TestFlexShardPrecompileRoundtrip(unittest.TestCase):
    """Verify FlexShard FX metadata is serializable for precompile."""

    def test_flex_shard_placement_metadata_picklable(self):
        """node.meta values used by FlexShard survive pickle round-trip."""
        meta = {
            "flex_shard_placement": True,
            "recompute": CheckpointPolicy.MUST_RECOMPUTE,
            "ac_graph_id": 100000,
        }
        restored = pickle.loads(pickle.dumps(meta))
        self.assertTrue(restored["flex_shard_placement"])
        self.assertEqual(restored["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertEqual(restored["ac_graph_id"], 100000)

    def test_must_save_policy_picklable(self):
        """MUST_SAVE checkpoint policy (reshard_after_forward=False) is picklable."""
        meta = {
            "flex_shard_placement": True,
            "recompute": CheckpointPolicy.MUST_SAVE,
            "ac_graph_id": 100000,
        }
        restored = pickle.loads(pickle.dumps(meta))
        self.assertEqual(restored["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_string_pg_name_in_args_picklable(self):
        """Process group name (string) and world_size args are picklable."""
        # FlexShard uses string PG names (not ProcessGroup objects)
        args = ("param_placeholder", 2, "0")
        restored = pickle.loads(pickle.dumps(args))
        self.assertEqual(restored, args)

    def test_fx_graph_node_meta_roundtrip(self):
        """FX graph nodes retain meta after graph manipulation."""
        graph = torch.fx.Graph()
        p = graph.placeholder("param")
        ag = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            (p, 2, "0"),
        )
        ag.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        ag.meta["ac_graph_id"] = 100000

        wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default, (ag,)
        )
        wait.meta["flex_shard_placement"] = True
        wait.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        wait.meta["ac_graph_id"] = 100000
        graph.output(wait)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # Verify metadata is accessible through the GraphModule's graph
        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIn("recompute", node.meta)
                self.assertEqual(
                    node.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE
                )

        # Verify the graph's node metadata survives graph.lint() and recompile
        gm.graph.lint()
        gm.recompile()

        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIn("recompute", node.meta)


if __name__ == "__main__":
    unittest.main()
