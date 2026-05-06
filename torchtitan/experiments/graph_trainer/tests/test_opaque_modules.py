# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.fx as fx
from torch._dynamo.testing import normalize_gm
from torch._inductor.fx_passes.control_dependencies import control_deps
from torch.testing._internal.common_utils import TestCase

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


class TestMakeModulesOpaque(TestCase):
    def test_wraps_only_matching_fqn(self):
        gm = _build_simple_gm()
        _make_modules_opaque(gm, ["loss"])
        # loss-tagged nodes (add, mul) are wrapped in control_deps with
        # the preceding-anchor chain; lm_head-tagged node (sub) stays as
        # the original op.
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        subgraph_add = self.subgraph_add

        add = torch.ops.higher_order.control_deps((), subgraph_add, x);  subgraph_add = x = None

        subgraph_mul = self.subgraph_mul

        mul = torch.ops.higher_order.control_deps((add,), subgraph_mul, add);  add = subgraph_mul = None

        sub = torch.sub(mul, 0.5);  mul = None
        return sub

    class subgraph_add(torch.nn.Module):
        def forward(self, arg_0):
            add = torch.add(arg_0, 1.0);  arg_0 = None
            return add

    class subgraph_mul(torch.nn.Module):
        def forward(self, arg_0):
            mul = torch.mul(arg_0, 2.0);  arg_0 = None
            return mul
""",
        )

    def test_no_match_is_noop(self):
        gm = _build_simple_gm()
        before = normalize_gm(gm.print_readable(print_output=False))
        _make_modules_opaque(gm, ["other"])
        after = normalize_gm(gm.print_readable(print_output=False))
        self.assertEqual(before, after)
        # Lock the no-op shape so a future change to the helper that
        # accidentally rewrites un-targeted nodes fails this test.
        self.assertExpectedInline(
            after,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        add = torch.add(x, 1.0);  x = None
        mul = torch.mul(add, 2.0);  add = None

        sub = torch.sub(mul, 0.5);  mul = None
        return sub
""",
        )

    def test_pairs_consecutive_anchored_nodes(self):
        gm = _build_simple_gm()
        _make_modules_opaque(gm, ["loss", "lm_head"])
        # Each anchored node carries a control_deps tuple referencing the
        # previous anchored node, forming the "loss → loss → lm_head"
        # chain that the wrapping pass must respect. The first wrap has
        # an empty deps tuple.
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        subgraph_add = self.subgraph_add

        add = torch.ops.higher_order.control_deps((), subgraph_add, x);  subgraph_add = x = None

        subgraph_mul = self.subgraph_mul

        mul = torch.ops.higher_order.control_deps((add,), subgraph_mul, add);  add = subgraph_mul = None

        subgraph_sub = self.subgraph_sub

        sub = torch.ops.higher_order.control_deps((mul,), subgraph_sub, mul);  mul = subgraph_sub = None
        return sub

    class subgraph_add(torch.nn.Module):
        def forward(self, arg_0):
            add = torch.add(arg_0, 1.0);  arg_0 = None
            return add

    class subgraph_mul(torch.nn.Module):
        def forward(self, arg_0):
            mul = torch.mul(arg_0, 2.0);  arg_0 = None
            return mul

    class subgraph_sub(torch.nn.Module):
        def forward(self, arg_0):
            sub = torch.sub(arg_0, 0.5);  arg_0 = None
            return sub
""",
        )


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
