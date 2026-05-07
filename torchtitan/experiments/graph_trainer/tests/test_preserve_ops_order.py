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

from torchtitan.experiments.graph_trainer.preserve_ops_order import (
    _pin_ops_order,
    _unpin_ops_order,
    preserve_ops_order,
)


def _annotate(node: fx.Node, fqn: str) -> None:
    node.meta.setdefault("custom", {})["module_fqn"] = fqn


def _build_simple_gm() -> fx.GraphModule:
    """Build a graph that mirrors the ChunkedCELoss buffer pattern: a
    fresh ``buf`` is filled by two in-place ``copy_`` writes into slice
    views (tagged ``loss``), then read back via ``clone(buf)`` (tagged
    ``lm_head``). The reader's only FX arg-edge dependency is on ``buf``
    itself — the read-after-write ordering between the writes and the
    clone is implicit through shared storage. This is the aliasing case
    a pass walking arg edges would freely reorder, and exactly what the
    preserve_ops_order pinning is meant to defend against."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    buf = graph.call_function(torch.zeros, args=((4,),))
    src0 = graph.call_function(torch.ops.aten.slice.Tensor, args=(x, 0, 0, 2))
    dst0 = graph.call_function(torch.ops.aten.slice.Tensor, args=(buf, 0, 0, 2))
    write0 = graph.call_function(torch.ops.aten.copy_.default, args=(dst0, src0))
    src1 = graph.call_function(torch.ops.aten.slice.Tensor, args=(x, 0, 2, 4))
    dst1 = graph.call_function(torch.ops.aten.slice.Tensor, args=(buf, 0, 2, 4))
    write1 = graph.call_function(torch.ops.aten.copy_.default, args=(dst1, src1))
    out = graph.call_function(torch.ops.aten.clone.default, args=(buf,))
    graph.output(out)
    gm = fx.GraphModule({}, graph)
    _annotate(write0, "loss")
    _annotate(write1, "loss")
    _annotate(out, "lm_head")
    return gm


def _ops_in_order(gm: fx.GraphModule) -> list:
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


class TestPinOpsOrder(TestCase):
    def test_wraps_only_matching_fqn(self):
        gm = _build_simple_gm()
        _pin_ops_order(gm, ["loss"])
        # loss-tagged copy_ writes get wrapped in a control_deps chain
        # so the bucketing pass sees an explicit write0 -> write1
        # ordering; the lm_head-tagged clone, the buffer alloc, and the
        # un-tagged slice views stay as plain ops. Note the clone has no
        # FX arg edge to either copy_ — without the wrapping chain a
        # reorder pass would be free to move it ahead of the writes.
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        zeros = torch.zeros((4,))
        slice_tensor = torch.ops.aten.slice.Tensor(x, 0, 0, 2)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(zeros, 0, 0, 2)
        subgraph_copy__default = self.subgraph_copy__default

        copy__default = torch.ops.higher_order.control_deps((), subgraph_copy__default, slice_tensor_1, slice_tensor);  subgraph_copy__default = slice_tensor_1 = slice_tensor = None

        slice_tensor_2 = torch.ops.aten.slice.Tensor(x, 0, 2, 4);  x = None
        slice_tensor_3 = torch.ops.aten.slice.Tensor(zeros, 0, 2, 4)
        subgraph_copy__default_1 = self.subgraph_copy__default_1

        copy__default_1 = torch.ops.higher_order.control_deps((copy__default,), subgraph_copy__default_1, slice_tensor_3, slice_tensor_2);  copy__default = subgraph_copy__default_1 = slice_tensor_3 = slice_tensor_2 = copy__default_1 = None

        clone_default = torch.ops.aten.clone.default(zeros);  zeros = None
        return clone_default

    class subgraph_copy__default(torch.nn.Module):
        def forward(self, arg_0, arg_1):
            copy__default = torch.ops.aten.copy_.default(arg_0, arg_1);  arg_0 = arg_1 = None
            return copy__default

    class subgraph_copy__default_1(torch.nn.Module):
        def forward(self, arg_0, arg_1):
            copy__default = torch.ops.aten.copy_.default(arg_0, arg_1);  arg_0 = arg_1 = None
            return copy__default
""",
        )

    def test_no_match_is_noop(self):
        gm = _build_simple_gm()
        before = normalize_gm(gm.print_readable(print_output=False))
        _pin_ops_order(gm, ["other"])
        after = normalize_gm(gm.print_readable(print_output=False))
        self.assertEqual(before, after)
        # Lock the no-op shape so a future change to the helper that
        # accidentally rewrites un-targeted nodes fails this test.
        self.assertExpectedInline(
            after,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        zeros = torch.zeros((4,))
        slice_tensor = torch.ops.aten.slice.Tensor(x, 0, 0, 2)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(zeros, 0, 0, 2)

        copy__default = torch.ops.aten.copy_.default(slice_tensor_1, slice_tensor);  slice_tensor_1 = slice_tensor = copy__default = None

        slice_tensor_2 = torch.ops.aten.slice.Tensor(x, 0, 2, 4);  x = None
        slice_tensor_3 = torch.ops.aten.slice.Tensor(zeros, 0, 2, 4)

        copy__default_1 = torch.ops.aten.copy_.default(slice_tensor_3, slice_tensor_2);  slice_tensor_3 = slice_tensor_2 = copy__default_1 = None

        clone_default = torch.ops.aten.clone.default(zeros);  zeros = None
        return clone_default
""",
        )

    def test_pairs_consecutive_pinned_nodes(self):
        gm = _build_simple_gm()
        _pin_ops_order(gm, ["loss", "lm_head"])
        # All loss + lm_head tagged nodes get linked into one shared
        # dependency chain (write0 -> write1 -> clone), so the bucketing
        # pass is forced to keep the writes before the read even though
        # FX arg edges don't connect them.
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        zeros = torch.zeros((4,))
        slice_tensor = torch.ops.aten.slice.Tensor(x, 0, 0, 2)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(zeros, 0, 0, 2)
        subgraph_copy__default = self.subgraph_copy__default

        copy__default = torch.ops.higher_order.control_deps((), subgraph_copy__default, slice_tensor_1, slice_tensor);  subgraph_copy__default = slice_tensor_1 = slice_tensor = None

        slice_tensor_2 = torch.ops.aten.slice.Tensor(x, 0, 2, 4);  x = None
        slice_tensor_3 = torch.ops.aten.slice.Tensor(zeros, 0, 2, 4)
        subgraph_copy__default_1 = self.subgraph_copy__default_1

        copy__default_1 = torch.ops.higher_order.control_deps((copy__default,), subgraph_copy__default_1, slice_tensor_3, slice_tensor_2);  copy__default = subgraph_copy__default_1 = slice_tensor_3 = slice_tensor_2 = None

        subgraph_clone_default = self.subgraph_clone_default

        clone_default = torch.ops.higher_order.control_deps((copy__default_1,), subgraph_clone_default, zeros);  copy__default_1 = subgraph_clone_default = zeros = None
        return clone_default

    class subgraph_copy__default(torch.nn.Module):
        def forward(self, arg_0, arg_1):
            copy__default = torch.ops.aten.copy_.default(arg_0, arg_1);  arg_0 = arg_1 = None
            return copy__default

    class subgraph_copy__default_1(torch.nn.Module):
        def forward(self, arg_0, arg_1):
            copy__default = torch.ops.aten.copy_.default(arg_0, arg_1);  arg_0 = arg_1 = None
            return copy__default

    class subgraph_clone_default(torch.nn.Module):
        def forward(self, arg_0):
            clone_default = torch.ops.aten.clone.default(arg_0);  arg_0 = None
            return clone_default
""",
        )


class TestUnpinOpsOrder(unittest.TestCase):
    def test_restore_brings_back_original_ops(self):
        gm = _build_simple_gm()
        before_targets = _ops_in_order(gm)
        _pin_ops_order(gm, ["loss", "lm_head"])
        # Sanity check: pinning replaced ops with control_deps wrappers.
        self.assertNotEqual(_ops_in_order(gm), before_targets)

        _unpin_ops_order(gm)

        after_targets = _ops_in_order(gm)
        self.assertEqual(after_targets, before_targets)

    def test_restore_is_noop_when_no_hops(self):
        gm = _build_simple_gm()
        before = _ops_in_order(gm)
        _unpin_ops_order(gm)
        self.assertEqual(_ops_in_order(gm), before)

    def test_make_then_restore_preserves_semantics(self):
        torch.manual_seed(0)
        gm = _build_simple_gm()
        x = torch.randn(4)
        expected = gm(x)

        _pin_ops_order(gm, ["loss", "lm_head"])
        _unpin_ops_order(gm)
        actual = gm(x)

        torch.testing.assert_close(actual, expected)


class TestPreserveOpsOrderDecorator(unittest.TestCase):
    def test_decorator_preserves_pass_output(self):
        gm = _build_simple_gm()
        before = _ops_in_order(gm)

        def identity_pass(g, example_inputs=None):
            return g

        wrapped = preserve_ops_order(["loss"])(identity_pass)
        result = wrapped(gm)

        # After identity-pass + restore, graph should match the original.
        self.assertIs(result, gm)
        self.assertEqual(_ops_in_order(gm), before)

    def test_decorator_restores_even_on_pass_exception(self):
        gm = _build_simple_gm()
        before = _ops_in_order(gm)

        def bad_pass(g, example_inputs=None):
            raise RuntimeError("boom")

        wrapped = preserve_ops_order(["loss"])(bad_pass)
        with self.assertRaises(RuntimeError):
            wrapped(gm)

        # control_deps wrappers must be cleaned up despite the exception,
        # so subsequent passes see the original graph.
        for node in gm.graph.nodes:
            self.assertIsNot(node.target, control_deps)
        self.assertEqual(_ops_in_order(gm), before)

    def test_decorator_makes_pass_see_pinned_nodes(self):
        gm = _build_simple_gm()
        seen_targets: list = []

        def inspect_pass(g, example_inputs=None):
            seen_targets.extend(
                n.target for n in g.graph.nodes if n.op == "call_function"
            )
            return g

        wrapped = preserve_ops_order(["loss"])(inspect_pass)
        wrapped(gm)

        # The pass should observe control_deps in place of the loss-tagged
        # copy_ writes, while the unpinned clone (lm_head) and slice
        # views remain visible as themselves.
        self.assertIn(control_deps, seen_targets)
        self.assertNotIn(torch.ops.aten.copy_.default, seen_targets)
        self.assertIn(torch.ops.aten.clone.default, seen_targets)
        self.assertIn(torch.ops.aten.slice.Tensor, seen_targets)


if __name__ == "__main__":
    unittest.main()
