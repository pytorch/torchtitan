# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

import torch
from torch.testing._internal.common_utils import TestCase
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
from torchtitan.experiments.graph_trainer.passes import apply_ilp_sac_pass


def _has_pulp() -> bool:
    try:
        import pulp  # noqa: F401

        return True
    except ImportError:
        return False


def _build_gm_with_layers(ops_per_layer, *, num_layers=2):
    """Build a GraphModule with forward and backward nodes across layers.

    Each layer gets the ops from ``ops_per_layer`` (a list of op targets).
    Forward nodes are annotated with module_fqn metadata and fake tensor
    metadata. A synthetic backward node consumes the last forward node
    to simulate an activation saved for backward.

    Returns (gm, forward_node_names_per_layer).
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(4, 16)

    y = graph.placeholder("y")
    y.meta["val"] = torch.empty(4, 16)

    forward_names: list[list[str]] = []
    last = x

    for layer_idx in range(num_layers):
        layer_names = []
        for op_target in ops_per_layer:
            node = graph.call_function(op_target, args=(last, y))
            node.meta["custom"] = {_MODULE_FQN: f"layers.{layer_idx}.feed_forward"}
            # Fake tensor metadata for memory estimation
            node.meta["val"] = torch.empty(4, 16)
            layer_names.append(node.name)
            last = node
        forward_names.append(layer_names)

    # Add a synthetic backward node that consumes the last forward node.
    # This makes the last forward node have backward consumers.
    bwd_node = graph.call_function(torch.ops.aten.neg.default, args=(last,))
    bwd_node.meta["autograd_backward"] = True
    bwd_node.meta["val"] = torch.empty(4, 16)

    # Also add backward consumers for intermediate nodes to make them
    # eligible as activations saved for backward.
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and not node.meta.get("autograd_backward", False)
            and node.target != operator.getitem
        ):
            bwd_consumer = graph.call_function(torch.ops.aten.neg.default, args=(node,))
            bwd_consumer.meta["autograd_backward"] = True
            bwd_consumer.meta["val"] = torch.empty(4, 16)

    graph.output(last)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm, forward_names


def _build_simple_gm(op_targets, *, layer_fqn="layers.0.feed_forward"):
    """Build a simple GraphModule with a chain of forward ops and backward consumers."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(4, 16)

    y = graph.placeholder("y")
    y.meta["val"] = torch.empty(4, 16)

    last = x
    for target in op_targets:
        if target is operator.getitem:
            node = graph.call_function(target, args=(last, 0))
        else:
            node = graph.call_function(target, args=(last, y))
        node.meta["custom"] = {_MODULE_FQN: layer_fqn}
        node.meta["val"] = torch.empty(4, 16)
        last = node

    # Add backward consumers for all forward nodes
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and not node.meta.get("autograd_backward", False)
            and node.target != operator.getitem
        ):
            bwd = graph.call_function(torch.ops.aten.neg.default, args=(node,))
            bwd.meta["autograd_backward"] = True
            bwd.meta["val"] = torch.empty(4, 16)

    graph.output(last)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _get_forward_policies(gm):
    """Extract recompute policies for forward call_function nodes."""
    policies = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.meta.get("autograd_backward", False):
            continue
        if "recompute" in node.meta:
            policies[node.name] = node.meta["recompute"]
    return policies


@unittest.skipUnless(_has_pulp(), "PuLP not installed")
class TestILPSACPass(TestCase):
    """Unit tests for the ILP-based SAC pass."""

    def test_budget_1_saves_everything(self):
        """With memory_budget=1.0, all candidate ops should be MUST_SAVE."""
        gm = _build_simple_gm(
            [
                torch.ops.aten.mm.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        apply_ilp_sac_pass(gm, memory_budget=1.0)
        policies = _get_forward_policies(gm)

        for name, policy in policies.items():
            self.assertEqual(
                policy,
                CheckpointPolicy.MUST_SAVE,
                f"Node {name} should be MUST_SAVE with budget=1.0",
            )

    def test_budget_0_recomputes_everything(self):
        """With memory_budget=0.0, all candidate ops should be PREFER_RECOMPUTE."""
        gm = _build_simple_gm(
            [
                torch.ops.aten.mm.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        apply_ilp_sac_pass(gm, memory_budget=0.0)
        policies = _get_forward_policies(gm)

        for name, policy in policies.items():
            self.assertEqual(
                policy,
                CheckpointPolicy.PREFER_RECOMPUTE,
                f"Node {name} should be PREFER_RECOMPUTE with budget=0.0",
            )

    def test_intermediate_budget_prioritizes_expensive_ops(self):
        """With a partial budget, the ILP should save expensive ops (mm) over cheap ones."""
        gm = _build_simple_gm(
            [
                torch.ops.aten.mm.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        # Use a budget that can save roughly 1 of 3 activations
        apply_ilp_sac_pass(gm, memory_budget=0.33)
        policies = _get_forward_policies(gm)

        # mm is compute-intensive, so it should be saved before add/relu
        mm_nodes = [
            name
            for name in policies
            if "mm" in name and policies[name] == CheckpointPolicy.MUST_SAVE
        ]
        add_relu_saved = [
            name
            for name in policies
            if ("add" in name or "relu" in name)
            and policies[name] == CheckpointPolicy.MUST_SAVE
        ]
        # With a tight budget, mm should be preferentially saved
        self.assertGreaterEqual(
            len(mm_nodes),
            len(add_relu_saved),
            "ILP should prioritize saving compute-intensive ops",
        )

    def test_comm_ops_always_saved(self):
        """Communication ops should always be MUST_SAVE regardless of budget."""
        gm = _build_simple_gm(
            [
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                torch.ops.aten.add.Tensor,
            ]
        )
        apply_ilp_sac_pass(gm, memory_budget=0.0)
        policies = _get_forward_policies(gm)

        for name, policy in policies.items():
            node = next(n for n in gm.graph.nodes if n.name == name)
            if node.target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
                self.assertEqual(
                    policy,
                    CheckpointPolicy.MUST_SAVE,
                    "Comm ops should always be MUST_SAVE",
                )

    def test_getitem_inherits_parent_policy(self):
        """getitem nodes should inherit the parent's recompute tag."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4, 16)
        y = graph.placeholder("y")
        y.meta["val"] = torch.empty(4, 16)

        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, y))
        mm.meta["custom"] = {_MODULE_FQN: "layers.0.attention"}
        mm.meta["val"] = torch.empty(4, 16)

        # Wrap in tuple for getitem
        def _make_tuple(t):
            return (t, t)

        tup = graph.call_function(_make_tuple, args=(mm,))
        tup.meta["custom"] = {_MODULE_FQN: "layers.0.attention"}
        tup.meta["val"] = (torch.empty(4, 16), torch.empty(4, 16))

        gi = graph.call_function(operator.getitem, args=(tup, 0))
        gi.meta["custom"] = {_MODULE_FQN: "layers.0.attention"}
        gi.meta["val"] = torch.empty(4, 16)

        # Add backward consumers
        for node in [mm, tup, gi]:
            bwd = graph.call_function(torch.ops.aten.neg.default, args=(node,))
            bwd.meta["autograd_backward"] = True
            bwd.meta["val"] = torch.empty(4, 16)

        graph.output(gi)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        apply_ilp_sac_pass(gm, memory_budget=1.0)

        gi_node = next(n for n in gm.graph.nodes if n.target is operator.getitem)
        parent_node = gi_node.args[0]
        self.assertIn("recompute", gi_node.meta)
        if "recompute" in parent_node.meta:
            self.assertEqual(
                gi_node.meta["recompute"],
                parent_node.meta["recompute"],
                "getitem should inherit parent policy",
            )

    def test_layer_boundary_forces_must_save(self):
        """Recomputable nodes at layer boundaries should be forced to MUST_SAVE."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4, 16)
        y = graph.placeholder("y")
        y.meta["val"] = torch.empty(4, 16)

        # Layer 0 node
        n0 = graph.call_function(torch.ops.aten.add.Tensor, args=(x, y))
        n0.meta["custom"] = {_MODULE_FQN: "layers.0.feed_forward"}
        n0.meta["val"] = torch.empty(4, 16)

        # Layer 1 node (consumes n0)
        n1 = graph.call_function(torch.ops.aten.add.Tensor, args=(n0, y))
        n1.meta["custom"] = {_MODULE_FQN: "layers.1.feed_forward"}
        n1.meta["val"] = torch.empty(4, 16)

        # Add backward consumers
        for node in [n0, n1]:
            bwd = graph.call_function(torch.ops.aten.neg.default, args=(node,))
            bwd.meta["autograd_backward"] = True
            bwd.meta["val"] = torch.empty(4, 16)

        graph.output(n1)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # budget=0 would normally recompute everything, but boundary constraint
        # forces n0 to MUST_SAVE since it feeds a higher layer.
        apply_ilp_sac_pass(gm, memory_budget=0.0)

        n0_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and not n.meta.get("autograd_backward", False)
            and n.meta.get("custom", {}).get(_MODULE_FQN) == "layers.0.feed_forward"
        )
        self.assertEqual(
            n0_node.meta["recompute"],
            CheckpointPolicy.MUST_SAVE,
            "Boundary node should be forced to MUST_SAVE",
        )

    def test_backward_nodes_not_tagged(self):
        """Backward nodes should not receive recompute tags."""
        gm = _build_simple_gm([torch.ops.aten.add.Tensor])
        apply_ilp_sac_pass(gm, memory_budget=0.5)

        for node in gm.graph.nodes:
            if node.meta.get("autograd_backward", False):
                self.assertNotIn(
                    "recompute",
                    node.meta,
                    f"Backward node {node.name} should not be tagged",
                )

    def test_invalid_budget_raises(self):
        """Memory budget outside [0.0, 1.0] should raise ValueError."""
        gm = _build_simple_gm([torch.ops.aten.add.Tensor])
        with self.assertRaises(ValueError):
            apply_ilp_sac_pass(gm, memory_budget=1.5)
        with self.assertRaises(ValueError):
            apply_ilp_sac_pass(gm, memory_budget=-0.1)

    def test_multi_layer_graph(self):
        """ILP should produce valid policies for a multi-layer graph."""
        gm, forward_names = _build_gm_with_layers(
            [torch.ops.aten.mm.default, torch.ops.aten.add.Tensor],
            num_layers=3,
        )
        apply_ilp_sac_pass(gm, memory_budget=0.5)
        policies = _get_forward_policies(gm)

        # Every forward node should have a policy
        for layer_names in forward_names:
            for name in layer_names:
                self.assertIn(name, policies, f"Node {name} should have a policy")
                self.assertIn(
                    policies[name],
                    (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_RECOMPUTE),
                    f"Node {name} has unexpected policy {policies[name]}",
                )

    def test_empty_graph(self):
        """An empty graph (no forward ops) should not crash."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # Should not raise
        apply_ilp_sac_pass(gm, memory_budget=0.5)


class TestILPSACPassWithoutPuLP(TestCase):
    """Test ILP SAC behavior when PuLP is not installed."""

    def test_import_error_message(self):
        """When PuLP is not available, a clear error message should be raised."""
        import unittest.mock

        gm = _build_simple_gm([torch.ops.aten.mm.default])

        # Mock the import to fail
        with unittest.mock.patch.dict("sys.modules", {"pulp": None}):
            with self.assertRaises(ImportError) as ctx:
                apply_ilp_sac_pass(gm, memory_budget=0.5)
            self.assertIn("PuLP", str(ctx.exception))


class TestNodeStatsCollection(TestCase):
    """Unit tests for node stats collection used by ILP."""

    def test_view_ops_get_forced_recompute(self):
        """View-like ops should have forced_policy = PREFER_RECOMPUTE."""
        from torchtitan.experiments.graph_trainer.sac_ilp import _collect_node_stats

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4, 16)

        view = graph.call_function(torch.ops.aten.view.default, args=(x, [4, 16]))
        view.meta["custom"] = {_MODULE_FQN: "layers.0.attention"}
        view.meta["val"] = torch.empty(4, 16)

        bwd = graph.call_function(torch.ops.aten.neg.default, args=(view,))
        bwd.meta["autograd_backward"] = True
        bwd.meta["val"] = torch.empty(4, 16)

        graph.output(view)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        layer_stats = _collect_node_stats(gm)
        # The view node should have forced_policy
        found = False
        for nodes in layer_stats.values():
            for ns in nodes:
                if ns.name == view.name:
                    self.assertEqual(
                        ns.forced_policy, CheckpointPolicy.PREFER_RECOMPUTE
                    )
                    found = True
        self.assertTrue(found, "View node should appear in stats")

    def test_comm_ops_get_forced_save(self):
        """Communication ops should have forced_policy = MUST_SAVE."""
        from torchtitan.experiments.graph_trainer.sac_ilp import _collect_node_stats

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4, 16)
        y = graph.placeholder("y")
        y.meta["val"] = torch.empty(4, 16)

        rs = graph.call_function(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            args=(x, y),
        )
        rs.meta["custom"] = {_MODULE_FQN: "layers.0.attention"}
        rs.meta["val"] = torch.empty(4, 16)

        bwd = graph.call_function(torch.ops.aten.neg.default, args=(rs,))
        bwd.meta["autograd_backward"] = True
        bwd.meta["val"] = torch.empty(4, 16)

        graph.output(rs)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        layer_stats = _collect_node_stats(gm)
        found = False
        for nodes in layer_stats.values():
            for ns in nodes:
                if ns.name == rs.name:
                    self.assertEqual(ns.forced_policy, CheckpointPolicy.MUST_SAVE)
                    found = True
        self.assertTrue(found, "Comm node should appear in stats")

    def test_compute_intensive_ops_have_high_cost(self):
        """Compute-intensive ops should have higher recompute cost than elementwise."""
        from torchtitan.experiments.graph_trainer.sac_ilp import _collect_node_stats

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4, 16)
        y = graph.placeholder("y")
        y.meta["val"] = torch.empty(4, 16)

        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, y))
        mm.meta["custom"] = {_MODULE_FQN: "layers.0.attention"}
        mm.meta["val"] = torch.empty(4, 16)

        add = graph.call_function(torch.ops.aten.add.Tensor, args=(mm, y))
        add.meta["custom"] = {_MODULE_FQN: "layers.0.attention"}
        add.meta["val"] = torch.empty(4, 16)

        for node in [mm, add]:
            bwd = graph.call_function(torch.ops.aten.neg.default, args=(node,))
            bwd.meta["autograd_backward"] = True
            bwd.meta["val"] = torch.empty(4, 16)

        graph.output(add)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        layer_stats = _collect_node_stats(gm)
        mm_cost = None
        add_cost = None
        for nodes in layer_stats.values():
            for ns in nodes:
                if ns.name == mm.name:
                    mm_cost = ns.recompute_cost
                elif ns.name == add.name:
                    add_cost = ns.recompute_cost

        self.assertIsNotNone(mm_cost)
        self.assertIsNotNone(add_cost)
        self.assertGreater(mm_cost, add_cost, "mm should cost more than add")


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
