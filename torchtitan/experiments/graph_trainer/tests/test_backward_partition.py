# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
import torch.fx as fx
from torch.testing._internal.common_utils import TestCase

from torchtitan.experiments.graph_trainer.backward_partition import (
    BackwardNodePartition,
    partition_backward_nodes,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import node_closure

aten = torch.ops.aten


def _custom_quantize(x):
    return x


def _custom_grouped_mm(a, b):
    return a


class TestBackwardNodePartition(TestCase):
    """Unit tests for dependency-based dI/dW backward node classification."""

    def _build_backward_graph(
        self,
        *,
        trunk_target=aten.mul.Tensor,
        di_target=aten.mm.default,
        dw_quant_target=aten.relu.default,
        dw_target=aten.mm.default,
    ):
        """Build a synthetic backward graph: shared trunk -> dI + dW branches.

        Returns the graph and a role -> node mapping. ``d_input`` is the input
        gradient output and ``d_weight`` is the parameter gradient output.
        """
        graph = fx.Graph()
        grad_out = graph.placeholder("grad_out")
        saved_act = graph.placeholder("saved_act")
        weight = graph.placeholder("weight")
        trunk = graph.call_function(trunk_target, args=(grad_out, saved_act))
        d_input = graph.call_function(di_target, args=(trunk, weight))
        dw_quant = graph.call_function(dw_quant_target, args=(trunk,))
        d_weight = graph.call_function(dw_target, args=(dw_quant, saved_act))
        graph.output((d_input, d_weight))
        nodes = {
            "grad_out": grad_out,
            "saved_act": saved_act,
            "weight": weight,
            "trunk": trunk,
            "di_out": d_input,
            "dw_quant": dw_quant,
            "dw_out": d_weight,
        }
        return graph, nodes

    def _partition(self, graph, nodes) -> BackwardNodePartition:
        return partition_backward_nodes(
            graph,
            input_grad_outputs=[nodes["di_out"]],
            param_grad_outputs=[nodes["dw_out"]],
        )

    def _assert_disjoint_partition(self, partition, nodes):
        classified = node_closure([nodes["di_out"]]) | node_closure([nodes["dw_out"]])
        self.assertEqual(
            partition.di_nodes | partition.dw_only_nodes | partition.shared_nodes,
            classified,
        )
        self.assertEqual(partition.di_nodes & partition.dw_only_nodes, set())
        self.assertEqual(partition.di_nodes & partition.shared_nodes, set())
        self.assertEqual(partition.dw_only_nodes & partition.shared_nodes, set())
        self.assertTrue(partition.movable_nodes <= partition.dw_only_nodes)

    def test_partition_is_disjoint_union_of_closures(self):
        graph, nodes = self._build_backward_graph()
        partition = self._partition(graph, nodes)
        self._assert_disjoint_partition(partition, nodes)

    def test_trunk_and_branch_membership(self):
        graph, nodes = self._build_backward_graph()
        partition = self._partition(graph, nodes)
        self.assertEqual(partition.di_nodes, {nodes["weight"], nodes["di_out"]})
        self.assertEqual(partition.dw_only_nodes, {nodes["dw_quant"], nodes["dw_out"]})
        self.assertEqual(
            partition.shared_nodes,
            {nodes["grad_out"], nodes["saved_act"], nodes["trunk"]},
        )
        self.assertEqual(partition.movable_nodes, {nodes["dw_quant"], nodes["dw_out"]})

    def test_collective_in_dw_branch_is_pinned(self):
        graph, nodes = self._build_backward_graph()
        with graph.inserting_before(nodes["dw_out"]):
            coll = graph.call_function(
                torch.ops._c10d_functional.all_to_all_single.default,
                args=(nodes["dw_quant"], [1], [1], "0"),
            )
            wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default, args=(coll,)
            )
        nodes["dw_out"].replace_input_with(nodes["dw_quant"], wait)
        partition = self._partition(graph, nodes)
        self._assert_disjoint_partition(partition, nodes)
        self.assertTrue({coll, wait} <= partition.dw_only_nodes)
        self.assertEqual(partition.movable_nodes, {nodes["dw_quant"], nodes["dw_out"]})

    def test_pre_write_reader_of_mutated_buffer_is_pinned(self):
        graph, nodes = self._build_backward_graph()
        # In-place write into saved_act, which both branches read. dw_quant
        # reads the buffer BEFORE the write without a data edge to it, so
        # deferring it could push the read past the write: pinned. dw_out
        # reads THROUGH the write (data edge keeps it after): movable.
        with graph.inserting_before(nodes["dw_out"]):
            mut = graph.call_function(aten.add_.Tensor, args=(nodes["saved_act"], 1.0))
        nodes["dw_out"].replace_input_with(nodes["saved_act"], mut)
        partition = self._partition(graph, nodes)
        self._assert_disjoint_partition(partition, nodes)
        self.assertIn(nodes["dw_quant"], partition.dw_only_nodes)
        self.assertNotIn(nodes["dw_quant"], partition.movable_nodes)
        self.assertIn(mut, partition.dw_only_nodes)
        self.assertNotIn(mut, partition.movable_nodes)
        self.assertEqual(partition.movable_nodes, {nodes["dw_out"]})

    def test_unresolvable_mutation_target_empties_movable(self):
        graph, nodes = self._build_backward_graph()
        # A write whose target is not a graph value cannot be reasoned about.
        with graph.inserting_before(nodes["dw_out"]):
            graph.call_function(aten.add_.Tensor, args=(3.0, 1.0))
        partition = self._partition(graph, nodes)
        self._assert_disjoint_partition(partition, nodes)
        self.assertEqual(partition.movable_nodes, set())

    def test_dw_local_mutation_pins_only_itself(self):
        graph, nodes = self._build_backward_graph()
        with graph.inserting_after(nodes["weight"]):
            dw_buf = graph.placeholder("dw_buf")
        with graph.inserting_before(nodes["dw_out"]):
            mut = graph.call_function(aten.add_.Tensor, args=(dw_buf, 1.0))
        nodes["dw_out"].replace_input_with(nodes["saved_act"], mut)
        partition = self._partition(graph, nodes)
        self._assert_disjoint_partition(partition, nodes)
        self.assertTrue({dw_buf, mut} <= partition.dw_only_nodes)
        self.assertEqual(partition.movable_nodes, {nodes["dw_quant"], nodes["dw_out"]})

    def test_cpu_scalar_read_is_pinned(self):
        graph, nodes = self._build_backward_graph()
        with graph.inserting_before(nodes["dw_out"]):
            sync = graph.call_function(
                aten._local_scalar_dense.default, args=(nodes["dw_quant"],)
            )
            scaled = graph.call_function(
                aten.mul.Tensor, args=(nodes["dw_quant"], sync)
            )
        nodes["dw_out"].replace_input_with(nodes["dw_quant"], scaled)
        partition = self._partition(graph, nodes)
        self._assert_disjoint_partition(partition, nodes)
        self.assertIn(sync, partition.dw_only_nodes)
        self.assertNotIn(sync, partition.movable_nodes)
        self.assertEqual(
            partition.movable_nodes,
            {nodes["dw_quant"], scaled, nodes["dw_out"]},
        )

    def test_device_to_host_copy_is_pinned(self):
        graph, nodes = self._build_backward_graph()
        fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        with fake_mode:
            nodes["dw_quant"].meta["val"] = torch.empty(4, device="cuda")
        with graph.inserting_before(nodes["dw_out"]):
            host_copy = graph.call_function(
                aten._to_copy.default,
                args=(nodes["dw_quant"],),
                kwargs={"device": torch.device("cpu")},
            )
        nodes["dw_out"].replace_input_with(nodes["dw_quant"], host_copy)
        partition = self._partition(graph, nodes)
        self.assertIn(host_copy, partition.dw_only_nodes)
        self.assertNotIn(host_copy, partition.movable_nodes)

    def test_sym_placeholder_feeding_both_sides_is_shared(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        graph, nodes = self._build_backward_graph()
        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            sym_batch = shape_env.create_unbacked_symint()
        with graph.inserting_after(nodes["weight"]):
            sym = graph.placeholder("sym_batch")
        sym.meta["val"] = sym_batch
        with graph.inserting_before(nodes["di_out"]):
            di_view = graph.call_function(
                aten.view.default, args=(nodes["di_out"].args[0], [sym, -1])
            )
        nodes["di_out"].replace_input_with(nodes["trunk"], di_view)
        with graph.inserting_before(nodes["dw_out"]):
            dw_view = graph.call_function(
                aten.view.default, args=(nodes["dw_quant"], [sym, -1])
            )
        nodes["dw_out"].replace_input_with(nodes["dw_quant"], dw_view)
        partition = self._partition(graph, nodes)
        self._assert_disjoint_partition(partition, nodes)
        self.assertIn(sym, partition.shared_nodes)
        self.assertNotIn(sym, partition.movable_nodes)
        self.assertIn(dw_view, partition.movable_nodes)

    def test_untracked_side_effect_user_pins_producer(self):
        graph, nodes = self._build_backward_graph()
        with graph.inserting_before(nodes["dw_out"]):
            graph.call_function(
                aten._assert_scalar.default, args=(nodes["dw_quant"], "msg")
            )
        partition = self._partition(graph, nodes)
        self.assertIn(nodes["dw_quant"], partition.dw_only_nodes)
        self.assertNotIn(nodes["dw_quant"], partition.movable_nodes)
        self.assertIn(nodes["dw_out"], partition.movable_nodes)

    def test_dead_getitem_users_do_not_pin_producer(self):
        # Mirrors the real MXFP8 wgrad chain: a multi-output quantize op
        # whose rowwise outputs are never consumed. The dead getitems are
        # inert and must not pin the producer.
        graph, nodes = self._build_backward_graph()
        with graph.inserting_after(nodes["dw_quant"]):
            multi = graph.call_function(_custom_quantize, args=(nodes["dw_quant"],))
        with graph.inserting_after(multi):
            dead_b = graph.call_function(operator.getitem, args=(multi, 1))
            used = graph.call_function(operator.getitem, args=(multi, 2))
            dead_a = graph.call_function(operator.getitem, args=(multi, 0))
        nodes["dw_out"].update_arg(0, used)
        partition = self._partition(graph, nodes)
        self.assertIn(multi, partition.movable_nodes)
        self.assertIn(used, partition.movable_nodes)
        self.assertIn(nodes["dw_quant"], partition.movable_nodes)
        classified = (
            partition.di_nodes | partition.dw_only_nodes | partition.shared_nodes
        )
        self.assertNotIn(dead_a, classified)
        self.assertNotIn(dead_b, classified)

    def test_classification_ignores_call_targets(self):
        aten_graph, aten_nodes = self._build_backward_graph()
        quant_graph, quant_nodes = self._build_backward_graph(
            dw_quant_target=_custom_quantize, dw_target=_custom_grouped_mm
        )
        aten_partition = self._partition(aten_graph, aten_nodes)
        quant_partition = self._partition(quant_graph, quant_nodes)

        def roles(partition, nodes, node_set):
            members = getattr(partition, node_set)
            return {role for role, node in nodes.items() if node in members}

        for node_set in (
            "di_nodes",
            "dw_only_nodes",
            "shared_nodes",
            "movable_nodes",
        ):
            self.assertEqual(
                roles(aten_partition, aten_nodes, node_set),
                roles(quant_partition, quant_nodes, node_set),
            )

    def test_no_input_grads_puts_everything_on_dw_side(self):
        graph, nodes = self._build_backward_graph()
        partition = partition_backward_nodes(
            graph,
            input_grad_outputs=[],
            param_grad_outputs=[nodes["di_out"], nodes["dw_out"]],
        )
        self.assertEqual(partition.di_nodes, set())
        self.assertEqual(partition.shared_nodes, set())
        self.assertEqual(partition.dw_only_nodes, set(nodes.values()))
        self.assertEqual(
            partition.movable_nodes,
            {nodes["trunk"], nodes["di_out"], nodes["dw_quant"], nodes["dw_out"]},
        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
