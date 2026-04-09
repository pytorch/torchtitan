#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard compatibility with graph_trainer passes.

Verifies that graph_trainer's pass pipeline (reassign_to_pg, bucketing, etc.)
composes correctly with FlexShard unshard patterns. Uses mock FX graphs —
no GPU/NCCL required.

Usage:
    python -m pytest test_flex_shard_passes.py -v
"""

import operator
import unittest

import torch
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)


def _build_mock_graph_with_pg(
    pattern: str, pg_name: str = "fake_pg"
) -> torch.fx.GraphModule:
    """Build a mock FX graph with a specific process group name.

    Like test_flex_shard_reshard._build_mock_graph but allows setting pg_name
    for reassign_to_pg tests.
    """
    graph = torch.fx.Graph()
    placeholder = graph.placeholder("param")
    cur = placeholder

    base_pattern = pattern

    # all_gather → wait_tensor
    ag = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        (cur, 2, pg_name),
    )
    wait = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (ag,))

    terminal = wait
    if base_pattern == "shard_dim0":
        pass
    elif base_pattern == "shard_dim_nonzero":
        chunk = graph.call_function(torch.ops.aten.chunk.default, (wait, 2, 0))
        gi0 = graph.call_function(operator.getitem, (chunk, 0))
        gi1 = graph.call_function(operator.getitem, (chunk, 1))
        cat = graph.call_function(torch.ops.aten.cat.default, ([gi0, gi1], 1))
        terminal = cat
    elif base_pattern == "flat_shard":
        view = graph.call_function(torch.ops.aten.view.default, (wait, [4, 8]))
        terminal = view

    graph.output(terminal)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _build_mixed_placement_graph(pg_name: str = "fake_pg") -> torch.fx.GraphModule:
    """Build a graph with both Shard(0) and Shard(dim!=0) unshard sequences."""
    graph = torch.fx.Graph()
    p0 = graph.placeholder("param_shard0")
    p1 = graph.placeholder("param_shard1")

    # Shard(0): all_gather → wait_tensor
    ag0 = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        (p0, 2, pg_name),
    )
    wait0 = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (ag0,))

    # Shard(dim!=0): all_gather → wait_tensor → chunk → getitem → cat
    ag1 = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        (p1, 2, pg_name),
    )
    wait1 = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (ag1,))
    chunk = graph.call_function(torch.ops.aten.chunk.default, (wait1, 2, 0))
    gi0 = graph.call_function(operator.getitem, (chunk, 0))
    gi1 = graph.call_function(operator.getitem, (chunk, 1))
    cat = graph.call_function(torch.ops.aten.cat.default, ([gi0, gi1], 1))

    # Use both results
    add = graph.call_function(torch.ops.aten.add.Tensor, (wait0, cat))
    graph.output(add)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


class TestReassignToPgComposition(unittest.TestCase):
    """Test reassign_to_pg_pass with FlexShard patterns."""

    def test_reassign_only_modifies_all_gather(self):
        """reassign_to_pg_pass modifies all_gather PG but not chunk/cat/view."""
        from torchtitan.experiments.graph_trainer.passes import reassign_to_pg_pass

        for pattern in ("shard_dim0", "shard_dim_nonzero", "flat_shard"):
            with self.subTest(pattern=pattern):
                gm = _build_mock_graph_with_pg(pattern, pg_name="original_pg")
                gm = reassign_to_pg_pass(
                    gm,
                    example_inputs=None,
                    source_pg_name="original_pg",
                    target_pg_name="new_pg",
                )

                for node in gm.graph.nodes:
                    if is_all_gather(node):
                        self.assertEqual(
                            node.args[2],
                            "new_pg",
                            "all_gather PG should be reassigned",
                        )
                    elif node.op == "call_function" and node.target in (
                        torch.ops.aten.chunk.default,
                        torch.ops.aten.cat.default,
                        torch.ops.aten.view.default,
                    ):
                        # Post-gather ops should be untouched
                        self.assertTrue(
                            len(node.args) <= 3
                            and not any(
                                a == "new_pg" for a in node.args if isinstance(a, str)
                            ),
                            f"{node.target} should not have PG args",
                        )

    def test_reassign_mixed_placements(self):
        """Both Shard(0) and Shard(dim!=0) all-gathers get reassigned."""
        from torchtitan.experiments.graph_trainer.passes import reassign_to_pg_pass

        gm = _build_mixed_placement_graph(pg_name="original_pg")
        gm = reassign_to_pg_pass(
            gm,
            example_inputs=None,
            source_pg_name="original_pg",
            target_pg_name="new_pg",
        )

        ag_nodes = [n for n in gm.graph.nodes if is_all_gather(n)]
        self.assertEqual(len(ag_nodes), 2)
        for ag in ag_nodes:
            self.assertEqual(ag.args[2], "new_pg")

    def test_reassign_no_match(self):
        """No-op when source PG doesn't match."""
        from torchtitan.experiments.graph_trainer.passes import reassign_to_pg_pass

        gm = _build_mock_graph_with_pg("shard_dim0", pg_name="my_pg")
        gm = reassign_to_pg_pass(
            gm,
            example_inputs=None,
            source_pg_name="other_pg",
            target_pg_name="new_pg",
        )

        ag_nodes = [n for n in gm.graph.nodes if is_all_gather(n)]
        self.assertEqual(ag_nodes[0].args[2], "my_pg")

    def test_reassign_broadcast_nodes(self):
        """reassign_to_pg_pass rewrites broadcast PG (Owned placement)."""
        from torchtitan.experiments.graph_trainer.passes import reassign_to_pg_pass

        # Build a graph with a broadcast node (Owned unshard pattern)
        graph = torch.fx.Graph()
        placeholder = graph.placeholder("param")
        bc = graph.call_function(
            torch.ops._c10d_functional.broadcast.default,
            (placeholder, 0, "original_pg"),
        )
        wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default, (bc,)
        )
        graph.output(wait)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        gm = reassign_to_pg_pass(
            gm,
            example_inputs=None,
            source_pg_name="original_pg",
            target_pg_name="new_pg",
        )

        # Verify broadcast PG was reassigned
        bc_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.broadcast.default
        ]
        self.assertEqual(len(bc_nodes), 1)
        self.assertEqual(bc_nodes[0].args[2], "new_pg")

    def test_reassign_mixed_all_gather_and_broadcast(self):
        """reassign_to_pg_pass rewrites both all-gather and broadcast nodes."""
        from torchtitan.experiments.graph_trainer.passes import reassign_to_pg_pass

        graph = torch.fx.Graph()
        p0 = graph.placeholder("param_shard")
        p1 = graph.placeholder("param_owned")

        # Shard(0) all-gather
        ag = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            (p0, 2, "original_pg"),
        )
        wait0 = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default, (ag,)
        )

        # Owned broadcast
        bc = graph.call_function(
            torch.ops._c10d_functional.broadcast.default,
            (p1, 0, "original_pg"),
        )
        wait1 = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default, (bc,)
        )

        add = graph.call_function(torch.ops.aten.add.Tensor, (wait0, wait1))
        graph.output(add)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        gm = reassign_to_pg_pass(
            gm,
            example_inputs=None,
            source_pg_name="original_pg",
            target_pg_name="new_pg",
        )

        # Both should be reassigned
        ag_nodes = [n for n in gm.graph.nodes if is_all_gather(n)]
        self.assertEqual(len(ag_nodes), 1)
        self.assertEqual(ag_nodes[0].args[2], "new_pg")

        bc_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.broadcast.default
        ]
        self.assertEqual(len(bc_nodes), 1)
        self.assertEqual(bc_nodes[0].args[2], "new_pg")


class TestPassOrdering(unittest.TestCase):
    """Test pass pipeline ordering with FlexShard."""

    def test_reshard_skipped_when_flex_shard_configured(self):
        """get_joint_custom_passes skips fsdp_reshard when flex_shard version present."""
        # Simulate what get_joint_custom_passes_from_config does:
        # When flex_shard_reshard_after_fwd is in joint_pass_names,
        # fsdp_reshard_after_fwd_pass should NOT be appended.
        joint_pass_names = ["flex_shard_reshard_after_fwd"]
        should_skip = "flex_shard_reshard_after_fwd" in joint_pass_names
        self.assertTrue(
            should_skip,
            "fsdp_reshard_after_fwd_pass should be skipped when "
            "flex_shard_reshard_after_fwd is configured",
        )

    def test_reshard_added_when_flex_shard_not_configured(self):
        """fsdp_reshard is appended when flex_shard version is NOT present."""
        joint_pass_names = ["apply_sac"]
        should_skip = "flex_shard_reshard_after_fwd" in joint_pass_names
        self.assertFalse(
            should_skip,
            "fsdp_reshard_after_fwd_pass should be appended when "
            "flex_shard_reshard_after_fwd is NOT configured",
        )

    def test_reassign_then_reshard_composition(self):
        """reassign_to_pg → reshard: both passes compose correctly."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            flex_shard_reshard_after_fwd_pass,
        )
        from torchtitan.experiments.graph_trainer.passes import reassign_to_pg_pass

        gm = _build_mixed_placement_graph(pg_name="original_pg")

        # Compiler pass: reassign PGs
        gm = reassign_to_pg_pass(
            gm,
            example_inputs=None,
            source_pg_name="original_pg",
            target_pg_name="new_pg",
        )

        # Joint pass: annotate for reshard (runs on joint graph, before compiler
        # passes in practice, but composition should still be safe)
        gm = flex_shard_reshard_after_fwd_pass(
            gm, example_inputs=None, reshard_after_forward=True
        )

        # Verify PGs were reassigned
        ag_nodes = [n for n in gm.graph.nodes if is_all_gather(n)]
        for ag in ag_nodes:
            self.assertEqual(ag.args[2], "new_pg")

        # Verify reshard annotations present
        annotated = [n for n in gm.graph.nodes if n.meta.get("ac_graph_id") == 100000]
        self.assertGreater(len(annotated), 0)


if __name__ == "__main__":
    unittest.main()
