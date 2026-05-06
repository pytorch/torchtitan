# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.fx as fx
from torch._inductor.fx_passes.control_dependencies import control_deps

from torchtitan.experiments.graph_trainer.opaque_modules import (
    _make_modules_opaque,
    _restore_modules,
    opaque_modules,
)


def _annotate(node: fx.Node, fqn: str) -> None:
    node.meta.setdefault("custom", {})["module_fqn"] = fqn


def _build_simple_gm() -> fx.GraphModule:
    """Build a small graph: x -> a -> b -> c -> y. Nodes a, b, c are tagged
    with module_fqn so we can selectively make them opaque."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    a = graph.call_function(torch.add, args=(x, 1.0))
    b = graph.call_function(torch.mul, args=(a, 2.0))
    c = graph.call_function(torch.sub, args=(b, 0.5))
    graph.output(c)
    gm = fx.GraphModule({}, graph)
    _annotate(a, "loss")
    _annotate(b, "loss")
    _annotate(c, "lm_head")
    return gm


def _ops_in_order(gm: fx.GraphModule) -> list:
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


class TestMakeModulesOpaque(unittest.TestCase):
    def test_wraps_only_matching_fqn(self):
        gm = _build_simple_gm()
        _make_modules_opaque(gm, ["loss"])

        ops = [n for n in gm.graph.nodes if n.op == "call_function"]
        targets = [n.target for n in ops]
        # loss-tagged nodes (a and b) get wrapped in control_deps;
        # lm_head-tagged node c is not in the fqn list, so untouched.
        self.assertEqual(targets.count(control_deps), 2)
        self.assertEqual(targets.count(torch.sub), 1)

    def test_no_match_is_noop(self):
        gm = _build_simple_gm()
        before = _ops_in_order(gm)
        _make_modules_opaque(gm, ["other"])
        after = _ops_in_order(gm)
        self.assertEqual(before, after)

    def test_pairs_consecutive_anchored_nodes(self):
        gm = _build_simple_gm()
        _make_modules_opaque(gm, ["loss", "lm_head"])

        wraps = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is control_deps
        ]
        # All three tagged nodes wrapped.
        self.assertEqual(len(wraps), 3)
        # First wrap has empty deps tuple, subsequent each depend on the
        # previous wrapped node (chain).
        self.assertEqual(tuple(wraps[0].args[0]), ())
        self.assertEqual(wraps[1].args[0], (wraps[0],))
        self.assertEqual(wraps[2].args[0], (wraps[1],))


class TestRestoreModules(unittest.TestCase):
    def test_restore_brings_back_original_ops(self):
        gm = _build_simple_gm()
        before_targets = _ops_in_order(gm)
        _make_modules_opaque(gm, ["loss", "lm_head"])
        # Sanity check: making opaque replaced ops with control_deps.
        self.assertNotEqual(_ops_in_order(gm), before_targets)

        _restore_modules(gm)

        after_targets = _ops_in_order(gm)
        self.assertEqual(after_targets, before_targets)

    def test_restore_is_noop_when_no_hops(self):
        gm = _build_simple_gm()
        before = _ops_in_order(gm)
        _restore_modules(gm)
        self.assertEqual(_ops_in_order(gm), before)

    def test_make_then_restore_preserves_semantics(self):
        torch.manual_seed(0)
        gm = _build_simple_gm()
        x = torch.randn(4)
        expected = gm(x)

        _make_modules_opaque(gm, ["loss", "lm_head"])
        _restore_modules(gm)
        actual = gm(x)

        torch.testing.assert_close(actual, expected)


class TestOpaqueModulesDecorator(unittest.TestCase):
    def test_decorator_preserves_pass_output(self):
        gm = _build_simple_gm()
        before = _ops_in_order(gm)

        def identity_pass(g, example_inputs=None):
            return g

        wrapped = opaque_modules(["loss"])(identity_pass)
        result = wrapped(gm)

        # After identity-pass + restore, graph should match the original.
        self.assertIs(result, gm)
        self.assertEqual(_ops_in_order(gm), before)

    def test_decorator_restores_even_on_pass_exception(self):
        gm = _build_simple_gm()
        before = _ops_in_order(gm)

        def bad_pass(g, example_inputs=None):
            raise RuntimeError("boom")

        wrapped = opaque_modules(["loss"])(bad_pass)
        with self.assertRaises(RuntimeError):
            wrapped(gm)

        # control_deps wrappers must be cleaned up despite the exception,
        # so subsequent passes see the original graph.
        for node in gm.graph.nodes:
            self.assertIsNot(node.target, control_deps)
        self.assertEqual(_ops_in_order(gm), before)

    def test_decorator_makes_pass_see_opaque_nodes(self):
        gm = _build_simple_gm()
        seen_targets: list = []

        def inspect_pass(g, example_inputs=None):
            seen_targets.extend(
                n.target for n in g.graph.nodes if n.op == "call_function"
            )
            return g

        wrapped = opaque_modules(["loss"])(inspect_pass)
        wrapped(gm)

        # The pass should observe control_deps in place of the loss-tagged
        # ops (torch.add and torch.mul) but the unanchored torch.sub
        # remains visible as itself.
        self.assertIn(control_deps, seen_targets)
        self.assertNotIn(torch.add, seen_targets)
        self.assertNotIn(torch.mul, seen_targets)
        self.assertIn(torch.sub, seen_targets)


if __name__ == "__main__":
    unittest.main()
