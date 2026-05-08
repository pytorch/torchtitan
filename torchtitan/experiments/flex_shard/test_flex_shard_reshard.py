#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard reshard_after_forward graph pass.

Uses mock FX graphs to test pattern matching without GPU/NCCL.

Usage:
    python -m pytest test_flex_shard_reshard.py -v
"""

import operator
import unittest

import torch
from torch.utils.checkpoint import CheckpointPolicy


def _build_mock_graph(pattern: str) -> torch.fx.GraphModule:
    """Build a mock FX graph matching FlexShard unshard patterns.

    Args:
        pattern: One of "shard_dim0", "shard_dim_nonzero", "flat_shard",
            "simple_fsdp", "shard_dim0_offload", "shard_dim0_mp",
            "flat_shard_mp", "shard_dim_nonzero_mp".
    """
    graph = torch.fx.Graph()

    # Placeholder (graph input = sharded param)
    placeholder = graph.placeholder("param")

    cur = placeholder

    # Optional pre-all_gather ops
    if pattern.endswith("_offload"):
        base_pattern = pattern.rsplit("_offload", 1)[0]
        cur = graph.call_function(torch.ops.aten._to_copy.default, (cur,))
    elif pattern == "simple_fsdp":
        # SimpleFSDP casts before all_gather
        cur = graph.call_function(
            torch.ops.prims.convert_element_type.default, (cur, torch.bfloat16)
        )
        base_pattern = pattern
    else:
        base_pattern = pattern.rsplit("_mp", 1)[0] if "_mp" in pattern else pattern

    # all_gather → wait_tensor (must use .default overloads for bucketing predicates)
    ag = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default, (cur, 2, "fake")
    )
    wait = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (ag,))

    # Post-wait_tensor ops depend on pattern
    terminal = wait
    if base_pattern == "shard_dim0":
        pass  # No post-processing for dim=0
    elif base_pattern == "shard_dim_nonzero":
        chunk = graph.call_function(torch.ops.aten.chunk.default, (wait, 2, 0))
        gi0 = graph.call_function(operator.getitem, (chunk, 0))
        gi1 = graph.call_function(operator.getitem, (chunk, 1))
        cat = graph.call_function(torch.ops.aten.cat.default, ([gi0, gi1], 1))
        terminal = cat
    elif base_pattern == "flat_shard":
        view = graph.call_function(torch.ops.aten.view.default, (wait, [4, 8]))
        terminal = view
    elif base_pattern == "simple_fsdp":
        slc = graph.call_function(torch.ops.aten.slice.Tensor, (wait, 0, 0, 4))
        terminal = slc

    # Optional mixed precision cast after terminal
    if "_mp" in pattern:
        cast = graph.call_function(
            torch.ops.prims.convert_element_type.default, (terminal, torch.bfloat16)
        )
        terminal = cast

    # Output
    graph.output(terminal)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm


def _get_annotated_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """Return nodes that have 'recompute' metadata set."""
    return [n for n in gm.graph.nodes if "recompute" in n.meta]


def _get_tagged_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """Return nodes that have 'flex_shard_placement' metadata set."""
    return [n for n in gm.graph.nodes if n.meta.get("flex_shard_placement")]


class TestShardDim0Pattern(unittest.TestCase):
    """Test Shard(dim=0) unshard pattern annotation."""

    def test_basic(self):
        """Annotates all_gather + wait_tensor for Shard(dim=0)."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim0")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        annotated = _get_annotated_nodes(gm)
        targets = {n.target for n in annotated}
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default, targets
        )
        self.assertIn(torch.ops._c10d_functional.wait_tensor.default, targets)
        self.assertEqual(len(annotated), 2)

    def test_with_offload(self):
        """Annotates .to() before all_gather for offloaded params."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim0_offload")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        annotated = _get_annotated_nodes(gm)
        targets = {n.target for n in annotated}
        self.assertIn(torch.ops.aten._to_copy.default, targets)
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default, targets
        )
        self.assertIn(torch.ops._c10d_functional.wait_tensor.default, targets)
        self.assertEqual(len(annotated), 3)

    def test_with_mp(self):
        """Annotates convert_element_type after wait_tensor for mixed precision."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim0_mp")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        annotated = _get_annotated_nodes(gm)
        targets = {n.target for n in annotated}
        self.assertIn(torch.ops.prims.convert_element_type.default, targets)
        self.assertEqual(len(annotated), 3)  # ag + wait + cast


class TestShardDimNonzeroPattern(unittest.TestCase):
    """Test Shard(dim!=0) unshard pattern annotation."""

    def test_basic(self):
        """Annotates chunk + getitem + cat for Shard(dim!=0)."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim_nonzero")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        annotated = _get_annotated_nodes(gm)
        targets = [n.target for n in annotated]
        # all_gather, wait, chunk, getitem x2, cat = 6 nodes
        self.assertEqual(len(annotated), 6)
        self.assertEqual(targets.count(operator.getitem), 2)
        self.assertIn(torch.ops.aten.chunk.default, targets)
        self.assertIn(torch.ops.aten.cat.default, targets)

    def test_with_mp(self):
        """Annotates chunk + cat + convert_element_type."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim_nonzero_mp")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        annotated = _get_annotated_nodes(gm)
        # all_gather, wait, chunk, getitem x2, cat, cast = 7
        self.assertEqual(len(annotated), 7)


class TestFlatShardPattern(unittest.TestCase):
    """Test FlatShard unshard pattern annotation."""

    def test_basic(self):
        """Annotates view after wait_tensor for FlatShard."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("flat_shard")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        annotated = _get_annotated_nodes(gm)
        targets = {n.target for n in annotated}
        self.assertIn(torch.ops.aten.view.default, targets)
        self.assertEqual(len(annotated), 3)  # ag + wait + view

    def test_with_mp(self):
        """Annotates view + convert_element_type."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("flat_shard_mp")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        annotated = _get_annotated_nodes(gm)
        self.assertEqual(len(annotated), 4)  # ag + wait + view + cast


class TestSimpleFSDPPattern(unittest.TestCase):
    """Test backward compatibility with SimpleFSDP pattern."""

    def test_basic(self):
        """Handles SimpleFSDP slice + pre-ag convert_element_type."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("simple_fsdp")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        annotated = _get_annotated_nodes(gm)
        targets = {n.target for n in annotated}
        # convert_element_type (pre-ag) + ag + wait + slice = 4
        self.assertIn(torch.ops.prims.convert_element_type.default, targets)
        self.assertIn(torch.ops.aten.slice.Tensor, targets)
        self.assertEqual(len(annotated), 4)


class TestRecomputePolicy(unittest.TestCase):
    """Test MUST_RECOMPUTE vs MUST_SAVE annotation."""

    def test_reshard_true_uses_must_recompute(self):
        """reshard_after_forward=True → MUST_RECOMPUTE."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim0")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        for n in _get_annotated_nodes(gm):
            self.assertEqual(n.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
            self.assertEqual(n.meta["ac_graph_id"], 100000)

    def test_reshard_false_uses_must_save(self):
        """reshard_after_forward=False → MUST_SAVE."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim0")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=False)

        for n in _get_annotated_nodes(gm):
            self.assertEqual(n.meta["recompute"], CheckpointPolicy.MUST_SAVE)


class TestMetadataTagging(unittest.TestCase):
    """Test flex_shard_placement metadata on terminal nodes."""

    def test_shard_dim0_tags_wait_tensor(self):
        """Shard(0): terminal is wait_tensor (no post-processing)."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim0")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        tagged = _get_tagged_nodes(gm)
        self.assertEqual(len(tagged), 1)
        self.assertEqual(
            tagged[0].target, torch.ops._c10d_functional.wait_tensor.default
        )

    def test_shard_dim_nonzero_tags_cat(self):
        """Shard(dim!=0): terminal is cat."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim_nonzero")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        tagged = _get_tagged_nodes(gm)
        self.assertEqual(len(tagged), 1)
        self.assertEqual(tagged[0].target, torch.ops.aten.cat.default)

    def test_flat_shard_tags_view(self):
        """FlatShard: terminal is view."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("flat_shard")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        tagged = _get_tagged_nodes(gm)
        self.assertEqual(len(tagged), 1)
        self.assertEqual(tagged[0].target, torch.ops.aten.view.default)

    def test_mp_tags_convert_element_type(self):
        """With mixed precision: terminal is convert_element_type."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            annotate_flex_shard_all_gather,
        )

        gm = _build_mock_graph("shard_dim0_mp")
        annotate_flex_shard_all_gather(gm, reshard_after_forward=True)

        tagged = _get_tagged_nodes(gm)
        self.assertEqual(len(tagged), 1)
        self.assertEqual(tagged[0].target, torch.ops.prims.convert_element_type.default)


class TestSACComposition(unittest.TestCase):
    """Test that reshard pass overrides SAC's PREFER_RECOMPUTE annotations."""

    def test_sac_then_reshard_preserves_must_recompute(self):
        """SAC → reshard: unshard nodes end up MUST_RECOMPUTE, not PREFER_RECOMPUTE."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            flex_shard_reshard_after_fwd_pass,
        )
        from torchtitan.experiments.graph_trainer.passes import apply_sac_pass

        for pattern in ("shard_dim0", "shard_dim_nonzero", "flat_shard"):
            with self.subTest(pattern=pattern):
                gm = _build_mock_graph(pattern)

                # SAC runs first — marks everything PREFER_RECOMPUTE
                gm = apply_sac_pass(gm)
                for n in gm.graph.nodes:
                    if n.op == "call_function" and "recompute" in n.meta:
                        self.assertIn(
                            n.meta["recompute"],
                            (
                                CheckpointPolicy.PREFER_RECOMPUTE,
                                CheckpointPolicy.MUST_SAVE,
                            ),
                        )

                # Reshard runs last — overwrites unshard nodes to MUST_RECOMPUTE
                gm = flex_shard_reshard_after_fwd_pass(
                    gm, example_inputs=None, reshard_after_forward=True
                )
                for n in _get_annotated_nodes(gm):
                    if n.meta.get("ac_graph_id") == 100000:
                        self.assertEqual(
                            n.meta["recompute"],
                            CheckpointPolicy.MUST_RECOMPUTE,
                            f"Node {n.target} should be MUST_RECOMPUTE after reshard pass",
                        )

    def test_sac_then_reshard_false_preserves_must_save(self):
        """SAC → reshard(False): unshard nodes end up MUST_SAVE."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            flex_shard_reshard_after_fwd_pass,
        )
        from torchtitan.experiments.graph_trainer.passes import apply_sac_pass

        gm = _build_mock_graph("shard_dim0")
        gm = apply_sac_pass(gm)
        gm = flex_shard_reshard_after_fwd_pass(
            gm, example_inputs=None, reshard_after_forward=False
        )
        for n in _get_annotated_nodes(gm):
            if n.meta.get("ac_graph_id") == 100000:
                self.assertEqual(n.meta["recompute"], CheckpointPolicy.MUST_SAVE)


class TestPassSignature(unittest.TestCase):
    """Test flex_shard_reshard_after_fwd_pass follows standard signature."""

    def test_returns_graph_module(self):
        """Pass returns GraphModule."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            flex_shard_reshard_after_fwd_pass,
        )

        gm = _build_mock_graph("shard_dim0")
        result = flex_shard_reshard_after_fwd_pass(
            gm, example_inputs=None, reshard_after_forward=True
        )
        self.assertIsInstance(result, torch.fx.GraphModule)

    def test_no_unshard_sequences(self):
        """Pass is a no-op on graphs without unshard sequences."""
        from torchtitan.experiments.flex_shard.reshard_after_forward import (
            flex_shard_reshard_after_fwd_pass,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, x))
        graph.output(add)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        result = flex_shard_reshard_after_fwd_pass(
            gm, example_inputs=None, reshard_after_forward=True
        )
        self.assertEqual(len(_get_annotated_nodes(result)), 0)


if __name__ == "__main__":
    unittest.main()
