# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase
from torch.utils.checkpoint import checkpoint, CheckpointPolicy

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import _AC_REGION_ID
from torchtitan.experiments.graph_trainer.graph_utils import export_joint
from torchtitan.experiments.graph_trainer.passes import (
    apply_sac_pass,
    reassign_to_pg_pass,
    remove_detach_pass,
    remove_identity_slice_pass,
    remove_identity_view_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleList


class ToyModel(Module):
    """A small toy model with multiple linear layers and activation
    checkpointing so that the backward graph recomputes the forward
    all-gathers."""

    def __init__(self, dim=16, n_layers=3):
        super().__init__()

        def _make_linear():
            cfg = Linear.Config(in_features=dim, out_features=dim, bias=True)
            return cfg.build()

        self.layers = ModuleList([_make_linear() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(
                lambda m, inp: torch.relu(m(inp)),
                layer,
                x,
                use_reentrant=False,
            )
        return x


class TestReassignToPgPass(FSDPTest):
    """Integration tests: toy model + simple_fsdp + export_joint + reassign_to_pg_pass."""

    def _setup(self):
        """Set up ParallelDims and device mesh for FSDP."""
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )

    def _make_fsdp_model(self, dim=16, n_layers=3):
        """Create a toy model and apply simple_fsdp data_parallel."""
        model = ToyModel(dim, n_layers).cuda()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model = data_parallel(model, device_mesh=fsdp_mesh, mode="fully_shard")
        return model

    def _get_fsdp_pg_name(self):
        """Get the FSDP process group name from the mesh."""
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        return fsdp_mesh.get_group().group_name

    def _export_and_get_bw_graph(self, model, inputs):
        """Export the joint graph and capture the backward graph via
        aot_compile_joint_with_descriptors with a custom bw_compiler."""
        joint_with_descriptors, tracing_context = export_joint(model, (inputs,))

        captured_bw_gm = {}

        def capture_bw_compiler(gm, example_inputs):
            captured_bw_gm["gm"] = gm
            captured_bw_gm["example_inputs"] = example_inputs
            return gm

        with tracing(tracing_context):
            aot_compile_joint_with_descriptors(
                joint_with_descriptors,
                bw_compiler=capture_bw_compiler,
            )

        return captured_bw_gm["gm"], captured_bw_gm["example_inputs"]

    def _count_ag_nodes_with_pg(self, gm, pg_name):
        """Count all-gather nodes in the graph that use the given PG name."""
        count = 0
        for node in gm.graph.nodes:
            if is_all_gather(node) and node.args[2] == pg_name:
                count += 1
        return count

    def _count_all_ag_nodes(self, gm):
        """Count all all-gather nodes in the graph regardless of PG."""
        count = 0
        for node in gm.graph.nodes:
            if is_all_gather(node):
                count += 1
        return count

    def test_reassign_rewrites_ag_nodes(self):
        """Apply reassign_to_pg_pass on the real backward graph and verify
        that all-gather nodes are rewritten to the target PG."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()
        target_pg_name = "test_target_pg"

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        # Before: all AG nodes should use the FSDP PG
        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertGreater(ag_before, 0, "Expected AG nodes with FSDP PG name")

        # Apply the pass
        reassign_to_pg_pass(
            bw_gm,
            bw_example_inputs,
            source_pg_name=fsdp_pg_name,
            target_pg_name=target_pg_name,
        )

        # After: AG nodes should use the target PG
        ag_with_old = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        ag_with_new = self._count_ag_nodes_with_pg(bw_gm, target_pg_name)

        self.assertEqual(ag_with_old, 0, "No AG nodes should still use the old PG")
        self.assertEqual(
            ag_with_new, ag_before, "All AG nodes should now use the target PG"
        )

    def test_reassign_preserves_total_ag_count(self):
        """The pass should not add or remove AG nodes, only rewrite PG names."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        total_before = self._count_all_ag_nodes(bw_gm)
        reassign_to_pg_pass(
            bw_gm,
            bw_example_inputs,
            source_pg_name=fsdp_pg_name,
            target_pg_name="new_pg",
        )
        total_after = self._count_all_ag_nodes(bw_gm)

        self.assertEqual(total_before, total_after)

    def test_reassign_with_non_matching_pg_is_noop(self):
        """If the source PG name doesn't match any AG node, nothing changes."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)

        # Use a non-matching source PG name
        reassign_to_pg_pass(
            bw_gm,
            bw_example_inputs,
            source_pg_name="nonexistent_pg",
            target_pg_name="target_pg",
        )

        # FSDP AG nodes should be unchanged
        ag_after = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertEqual(ag_before, ag_after)

    def test_reassign_with_extra_pg(self):
        """Test the production-like flow: create an extra FSDP PG and
        reassign AG nodes to it."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        # Create an extra PG mirroring the FSDP topology
        from torchtitan.experiments.graph_trainer.common_utils import (
            create_extra_fsdp_pg,
            get_extra_fsdp_pg_name,
        )

        create_extra_fsdp_pg(self.parallel_dims)
        extra_pg_name = get_extra_fsdp_pg_name(fsdp_pg_name)

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertGreater(ag_before, 0)

        # Reassign to the real extra PG
        reassign_to_pg_pass(
            bw_gm,
            bw_example_inputs,
            source_pg_name=fsdp_pg_name,
            target_pg_name=extra_pg_name,
        )

        ag_old = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        ag_new = self._count_ag_nodes_with_pg(bw_gm, extra_pg_name)

        self.assertEqual(ag_old, 0)
        self.assertEqual(ag_new, ag_before)


class TestApplySACPass(TestCase):
    """Unit tests for the apply_sac_pass joint graph pass."""

    def _build_gm(self, op_targets):
        """Build a GraphModule with a chain of call_function nodes.

        Each op in op_targets becomes a call_function node. The graph
        structure is: placeholder(x), placeholder(y) -> op1 -> op2 -> ... -> output.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        last = x
        for i, target in enumerate(op_targets):
            if target is operator.getitem:
                last = graph.call_function(target, args=(last, 0))
            else:
                last = graph.call_function(target, args=(last, y))
                # If the next op is getitem, wrap in a tuple so getitem has
                # a proper tuple/list input.
                if i + 1 < len(op_targets) and op_targets[i + 1] is operator.getitem:
                    _make_tuple = lambda x: (x, x)
                    last = graph.call_function(_make_tuple, args=(last,))
        graph.output(last)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def _get_call_function_nodes(self, gm):
        """Return all call_function nodes from the graph."""
        return [n for n in gm.graph.nodes if n.op == "call_function"]

    def test_non_save_ops_marked_recompute(self):
        """Ops not in the save list should be marked PREFER_RECOMPUTE."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        apply_sac_pass(gm)
        for node in self._get_call_function_nodes(gm):
            self.assertEqual(node.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)

    def test_save_ops_marked_must_save(self):
        """Non-mm ops in the save list should be marked MUST_SAVE."""
        custom_save = {torch.ops.aten.add.Tensor}
        gm = self._build_gm([torch.ops.aten.add.Tensor])
        apply_sac_pass(gm, op_list_to_save=custom_save)
        nodes = self._get_call_function_nodes(gm)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_getitem_propagates_parent_tags(self):
        """operator.getitem nodes should inherit the parent's recompute tag and ac_graph_id."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                operator.getitem,
                torch.ops.aten.relu.default,
            ]
        )
        nodes = self._get_call_function_nodes(gm)
        # nodes: [add, make_tuple, getitem, relu]
        # make_tuple is the tuple-returning parent of getitem
        self.assertEqual(nodes[0].target, torch.ops.aten.add.Tensor)
        self.assertEqual(nodes[2].target, operator.getitem)

        # Set ac_region_id on the tuple-returning parent (the direct parent of getitem)
        nodes[1].meta["custom"] = {_AC_REGION_ID: 3}

        apply_sac_pass(gm)

        tuple_node = nodes[1]
        getitem_node = nodes[2]
        self.assertEqual(getitem_node.meta["recompute"], tuple_node.meta["recompute"])
        self.assertEqual(tuple_node.meta["ac_graph_id"], 3)
        self.assertEqual(getitem_node.meta["ac_graph_id"], 3)

    def test_wait_tensor_propagates_parent_tags(self):
        """wait_tensor nodes should inherit the parent's recompute tag and ac_graph_id."""
        custom_save = {torch.ops._c10d_functional.reduce_scatter_tensor.default}
        gm = self._build_gm(
            [
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                torch.ops._c10d_functional.wait_tensor.default,
            ]
        )
        nodes = self._get_call_function_nodes(gm)
        nodes[0].meta["custom"] = {_AC_REGION_ID: 3}

        apply_sac_pass(gm, op_list_to_save=custom_save)

        rs_node = nodes[0]
        wait_node = nodes[1]
        self.assertEqual(rs_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(wait_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(rs_node.meta["ac_graph_id"], 3)
        self.assertEqual(wait_node.meta["ac_graph_id"], 3)

    def test_ac_graph_id_defaults_to_zero(self):
        """Nodes without ac_region_id annotation should have ac_graph_id = 0."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.mm.default,
                torch.ops.aten.relu.default,
            ]
        )
        apply_sac_pass(gm)
        for node in self._get_call_function_nodes(gm):
            if node.target is not operator.getitem:
                self.assertEqual(node.meta["ac_graph_id"], 0)

    def test_ac_graph_id_from_annotation(self):
        """Nodes with _AC_REGION_ID_KEY in custom metadata should use that as ac_graph_id."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        nodes = self._get_call_function_nodes(gm)
        # Simulate annotate_fn setting custom metadata on different nodes
        nodes[0].meta["custom"] = {_AC_REGION_ID: 1}
        nodes[1].meta["custom"] = {_AC_REGION_ID: 2}

        apply_sac_pass(gm)

        self.assertEqual(nodes[0].meta["ac_graph_id"], 1)
        self.assertEqual(nodes[1].meta["ac_graph_id"], 2)

    def test_custom_op_list_to_save(self):
        """A custom op_list_to_save should override the defaults."""
        custom_save = {torch.ops.aten.relu.default}
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        apply_sac_pass(gm, op_list_to_save=custom_save)
        policies = {
            n.target: n.meta["recompute"] for n in self._get_call_function_nodes(gm)
        }
        self.assertEqual(
            policies[torch.ops.aten.add.Tensor], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertEqual(
            policies[torch.ops.aten.relu.default], CheckpointPolicy.MUST_SAVE
        )

    def test_mixed_mm_and_save_ops(self):
        """Graph with both mm and other save ops are annotated correctly."""
        custom_save = {torch.ops.aten.mm.default, torch.ops.aten.max.default}
        gm = self._build_gm(
            [
                torch.ops.aten.mm.default,  # 1st mm -> MUST_SAVE
                torch.ops.aten.max.default,  # in save list -> MUST_SAVE
                torch.ops.aten.mm.default,  # 2nd mm -> PREFER_RECOMPUTE
                torch.ops.aten.add.Tensor,  # not in save list -> PREFER_RECOMPUTE
                torch.ops.aten.mm.default,  # 3rd mm -> MUST_SAVE
            ]
        )
        apply_sac_pass(gm, op_list_to_save=custom_save)
        nodes = self._get_call_function_nodes(gm)
        expected = [
            (torch.ops.aten.mm.default, CheckpointPolicy.MUST_SAVE),
            (torch.ops.aten.max.default, CheckpointPolicy.MUST_SAVE),
            (torch.ops.aten.mm.default, CheckpointPolicy.PREFER_RECOMPUTE),
            (torch.ops.aten.add.Tensor, CheckpointPolicy.PREFER_RECOMPUTE),
            (torch.ops.aten.mm.default, CheckpointPolicy.MUST_SAVE),
        ]
        self.assertEqual(len(nodes), len(expected))
        for node, (target, policy) in zip(nodes, expected):
            self.assertEqual(node.target, target)
            self.assertEqual(node.meta["recompute"], policy, f"node {node.name}")


class TestRemoveDetachPass(TestCase):
    """Unit tests for the remove_detach_pass graph pass."""

    def _build_detach_gm(self, op_targets):
        """Build a GraphModule with a chain of call_function nodes.

        Each op in op_targets becomes a call_function node chained sequentially:
        placeholder(x) -> op1(x) -> op2(...) -> ... -> output.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        last = x
        for target in op_targets:
            last = graph.call_function(target, args=(last,))
        graph.output(last)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def _count_detach_nodes(self, gm):
        """Count aten.detach.default call_function nodes."""
        return sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.detach.default
        )

    def _count_call_function_nodes(self, gm):
        """Count all call_function nodes."""
        return sum(1 for n in gm.graph.nodes if n.op == "call_function")

    def test_detach_nodes_removed(self):
        """Detach nodes are removed from a simple graph containing them."""
        gm = self._build_detach_gm(
            [
                torch.ops.aten.relu.default,
                torch.ops.aten.detach.default,
                torch.ops.aten.neg.default,
            ]
        )
        self.assertEqual(self._count_detach_nodes(gm), 1)

        result = remove_detach_pass(gm)

        self.assertEqual(self._count_detach_nodes(result), 0)
        # relu and neg should remain
        self.assertEqual(self._count_call_function_nodes(result), 2)

    def test_graph_without_detach_unchanged(self):
        """Graphs without detach nodes are returned unchanged."""
        gm = self._build_detach_gm(
            [
                torch.ops.aten.relu.default,
                torch.ops.aten.neg.default,
            ]
        )
        num_nodes_before = len(list(gm.graph.nodes))

        result = remove_detach_pass(gm)

        self.assertIs(result, gm)
        self.assertEqual(len(list(result.graph.nodes)), num_nodes_before)

    def test_numerics_preserved(self):
        """Forward outputs are preserved after removing detach nodes."""
        gm = self._build_detach_gm(
            [
                torch.ops.aten.relu.default,
                torch.ops.aten.detach.default,
                torch.ops.aten.neg.default,
            ]
        )
        x = torch.randn(4, 4)
        expected = torch.neg(torch.detach_copy(torch.relu(x)))

        remove_detach_pass(gm)
        actual = gm(x)

        self.assertEqual(actual, expected)

    def test_detach_with_multiple_users(self):
        """Detach node with multiple users: all uses are replaced."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        detach = graph.call_function(torch.ops.aten.detach.default, args=(x,))
        relu = graph.call_function(torch.ops.aten.relu.default, args=(detach,))
        neg = graph.call_function(torch.ops.aten.neg.default, args=(detach,))
        graph.output((relu, neg))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        self.assertEqual(self._count_detach_nodes(gm), 1)

        remove_detach_pass(gm)

        self.assertEqual(self._count_detach_nodes(gm), 0)

        # Both relu and neg should now consume the placeholder directly
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in (
                torch.ops.aten.relu.default,
                torch.ops.aten.neg.default,
            ):
                self.assertEqual(node.args[0].op, "placeholder")

        # Verify numerics
        x = torch.randn(4, 4)
        relu_out, neg_out = gm(x)
        self.assertEqual(relu_out, torch.relu(x))
        self.assertEqual(neg_out, torch.neg(x))

    def test_nested_detach_chain(self):
        """Nested detach chain (detach -> detach -> detach) is fully removed."""
        gm = self._build_detach_gm(
            [
                torch.ops.aten.relu.default,
                torch.ops.aten.detach.default,
                torch.ops.aten.detach.default,
                torch.ops.aten.detach.default,
                torch.ops.aten.neg.default,
            ]
        )
        self.assertEqual(self._count_detach_nodes(gm), 3)

        remove_detach_pass(gm)

        self.assertEqual(self._count_detach_nodes(gm), 0)
        self.assertEqual(self._count_call_function_nodes(gm), 2)

        # Verify numerics
        x = torch.randn(4, 4)
        expected = torch.neg(torch.relu(x))
        self.assertEqual(gm(x), expected)


class TestRemoveIdentityViewPass(TestCase):
    """Unit tests for the remove_identity_view_pass graph pass."""

    _VIEW_TARGETS = [
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    ]

    def _build_view_gm(self, op_targets, *, shapes=None):
        """Build a GraphModule with a chain of call_function nodes.

        Each op in ``op_targets`` becomes a call_function node chained
        sequentially: placeholder(x) -> op1(x, shape) -> op2(..., shape) -> output.

        If ``shapes`` is provided it must have the same length as
        ``op_targets`` and supplies the shape argument for each view-like
        node.  Non-view nodes ignore the corresponding entry.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        last = x
        for i, target in enumerate(op_targets):
            if target in (
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten._unsafe_view.default,
            ):
                shape = shapes[i] if shapes else [4, 4]
                last = graph.call_function(target, args=(last, shape))
            else:
                last = graph.call_function(target, args=(last,))
        graph.output(last)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def _attach_fake_meta(self, gm, input_shape):
        """Attach fake tensor metadata to all nodes based on op semantics."""
        fake_mode = torch._subclasses.FakeTensorMode()
        with fake_mode:
            fake_input = torch.randn(input_shape)
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = fake_input
            elif node.op == "call_function":
                if node.target in (
                    torch.ops.aten.view.default,
                    torch.ops.aten.reshape.default,
                    torch.ops.aten._unsafe_view.default,
                ):
                    target_shape = node.args[1]
                    with fake_mode:
                        node.meta["val"] = torch.randn(target_shape)
                else:
                    # For unary ops like relu/neg, output shape == input shape.
                    node.meta["val"] = node.args[0].meta.get("val")

    def _count_view_nodes(self, gm):
        """Count view/reshape/_unsafe_view call_function nodes."""
        targets = {
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
        }
        return sum(
            1 for n in gm.graph.nodes if n.op == "call_function" and n.target in targets
        )

    def _count_call_function_nodes(self, gm):
        """Count all call_function nodes."""
        return sum(1 for n in gm.graph.nodes if n.op == "call_function")

    def test_identity_view_removed(self):
        """Identity view (same shape in and out) is removed for each op type."""
        for target in self._VIEW_TARGETS:
            with self.subTest(target=target):
                gm = self._build_view_gm(
                    [torch.ops.aten.relu.default, target, torch.ops.aten.neg.default],
                    shapes=[None, [4, 4], None],
                )
                self._attach_fake_meta(gm, (4, 4))
                self.assertEqual(self._count_view_nodes(gm), 1)

                result = remove_identity_view_pass(gm)

                self.assertEqual(self._count_view_nodes(result), 0)
                self.assertEqual(self._count_call_function_nodes(result), 2)

    def test_non_identity_view_preserved(self):
        """Non-identity view (shape changes) is kept."""
        gm = self._build_view_gm(
            [torch.ops.aten.view.default],
            shapes=[[2, 8]],
        )
        self._attach_fake_meta(gm, (4, 4))
        self.assertEqual(self._count_view_nodes(gm), 1)

        remove_identity_view_pass(gm)

        self.assertEqual(self._count_view_nodes(gm), 1)

    def test_view_without_metadata_skipped(self):
        """Nodes without tensor metadata are skipped safely."""
        gm = self._build_view_gm(
            [torch.ops.aten.view.default],
            shapes=[[4, 4]],
        )
        # Do NOT attach fake meta — nodes have no "val" in meta.

        # Should not raise and should not modify the graph.
        remove_identity_view_pass(gm)

        self.assertEqual(self._count_view_nodes(gm), 1)

    def test_numerics_preserved(self):
        """Forward outputs are preserved after removing identity views."""
        gm = self._build_view_gm(
            [
                torch.ops.aten.relu.default,
                torch.ops.aten.view.default,
                torch.ops.aten.neg.default,
            ],
            shapes=[None, [4, 4], None],
        )
        self._attach_fake_meta(gm, (4, 4))

        x = torch.randn(4, 4)
        expected = torch.neg(torch.relu(x).view(4, 4))

        remove_identity_view_pass(gm)
        actual = gm(x)

        self.assertEqual(actual, expected)

    def test_view_with_multiple_users(self):
        """Identity view with multiple users: all uses are replaced."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        view = graph.call_function(torch.ops.aten.view.default, args=(x, [4, 4]))
        relu = graph.call_function(torch.ops.aten.relu.default, args=(view,))
        neg = graph.call_function(torch.ops.aten.neg.default, args=(view,))
        graph.output((relu, neg))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # Attach metadata
        fake_mode = torch._subclasses.FakeTensorMode()
        with fake_mode:
            fake_input = torch.randn(4, 4)
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = fake_input
            elif node.target is torch.ops.aten.view.default:
                node.meta["val"] = fake_input  # same shape
            elif node.op == "call_function":
                node.meta["val"] = fake_input

        self.assertEqual(self._count_view_nodes(gm), 1)

        remove_identity_view_pass(gm)

        self.assertEqual(self._count_view_nodes(gm), 0)

        # Both relu and neg should now consume the placeholder directly
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in (
                torch.ops.aten.relu.default,
                torch.ops.aten.neg.default,
            ):
                self.assertEqual(node.args[0].op, "placeholder")

        # Verify numerics
        x = torch.randn(4, 4)
        relu_out, neg_out = gm(x)
        self.assertEqual(relu_out, torch.relu(x))
        self.assertEqual(neg_out, torch.neg(x))

    def test_chain_of_identity_views(self):
        """Chain of identity views (view -> view -> view) is fully removed."""
        gm = self._build_view_gm(
            [
                torch.ops.aten.relu.default,
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten._unsafe_view.default,
                torch.ops.aten.neg.default,
            ],
            shapes=[None, [4, 4], [4, 4], [4, 4], None],
        )
        self._attach_fake_meta(gm, (4, 4))
        self.assertEqual(self._count_view_nodes(gm), 3)

        remove_identity_view_pass(gm)

        self.assertEqual(self._count_view_nodes(gm), 0)
        self.assertEqual(self._count_call_function_nodes(gm), 2)

        # Verify numerics
        x = torch.randn(4, 4)
        expected = torch.neg(torch.relu(x))
        self.assertEqual(gm(x), expected)

    def test_graph_without_views_unchanged(self):
        """Graphs without view nodes are returned unchanged."""
        gm = self._build_view_gm(
            [torch.ops.aten.relu.default, torch.ops.aten.neg.default],
            shapes=[None, None],
        )
        self._attach_fake_meta(gm, (4, 4))
        num_nodes_before = len(list(gm.graph.nodes))

        result = remove_identity_view_pass(gm)

        self.assertIs(result, gm)
        self.assertEqual(len(list(result.graph.nodes)), num_nodes_before)


class TestRemoveIdentitySlicePass(TestCase):
    """Unit tests for the remove_identity_slice_pass graph pass."""

    def _build_slice_gm(self, input_shape, dim, start, end, step=1):
        """Build a GraphModule with a single aten.slice.Tensor node.

        Creates: placeholder(x) -> slice(x, dim, start, end, step) -> output.
        The placeholder is annotated with fake tensor metadata of the given shape.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        sliced = graph.call_function(
            torch.ops.aten.slice.Tensor, args=(x, dim, start, end, step)
        )
        graph.output(sliced)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # Annotate placeholder with fake tensor metadata
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as fake_mode:
            fake_val = fake_mode.from_tensor(torch.empty(*input_shape))
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = fake_val
        return gm

    def _count_slice_nodes(self, gm):
        """Count aten.slice.Tensor nodes in the graph."""
        return sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.slice.Tensor
        )

    def test_full_dim_slice_is_removed(self):
        """A slice selecting the full dimension (start=0, end>=dim_size, step=1)
        should be removed."""
        gm = self._build_slice_gm(input_shape=(8, 16), dim=0, start=0, end=8, step=1)
        self.assertEqual(self._count_slice_nodes(gm), 1)

        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 0)

    def test_full_dim_slice_large_end_is_removed(self):
        """A slice with end > dim_size should also be removed (identity)."""
        import sys

        gm = self._build_slice_gm(
            input_shape=(8, 16), dim=0, start=0, end=sys.maxsize, step=1
        )
        self.assertEqual(self._count_slice_nodes(gm), 1)

        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 0)

    def test_partial_slice_start_preserved(self):
        """A slice with start > 0 is not an identity and should be preserved."""
        gm = self._build_slice_gm(input_shape=(8, 16), dim=0, start=2, end=8, step=1)
        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 1)

    def test_partial_slice_end_preserved(self):
        """A slice with end < dim_size is not an identity and should be preserved."""
        gm = self._build_slice_gm(input_shape=(8, 16), dim=0, start=0, end=4, step=1)
        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 1)

    def test_partial_slice_step_preserved(self):
        """A slice with step > 1 is not an identity and should be preserved."""
        gm = self._build_slice_gm(input_shape=(8, 16), dim=0, start=0, end=8, step=2)
        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 1)

    def test_no_metadata_skipped(self):
        """Slice nodes without fake tensor metadata should be skipped safely."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        sliced = graph.call_function(
            torch.ops.aten.slice.Tensor, args=(x, 0, 0, 100, 1)
        )
        graph.output(sliced)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # No metadata set -- pass should not crash
        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 1)

    def test_multi_dim_slice(self):
        """Identity slice on a non-zero dimension should be removed."""
        gm = self._build_slice_gm(
            input_shape=(8, 16, 32), dim=2, start=0, end=32, step=1
        )
        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 0)

    def test_numerics_preserved(self):
        """The pass should not change the numerical output of the graph."""
        gm = self._build_slice_gm(input_shape=(4, 8), dim=0, start=0, end=4, step=1)

        # Run before the pass
        x = torch.randn(4, 8)
        out_before = gm(x)

        remove_identity_slice_pass(gm)

        out_after = gm(x)
        self.assertTrue(torch.equal(out_before, out_after))

    def test_chained_identity_slices(self):
        """Multiple chained identity slices should all be removed."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        s1 = graph.call_function(torch.ops.aten.slice.Tensor, args=(x, 0, 0, 8, 1))
        s2 = graph.call_function(torch.ops.aten.slice.Tensor, args=(s1, 1, 0, 16, 1))
        graph.output(s2)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as fake_mode:
            fake_val = fake_mode.from_tensor(torch.empty(8, 16))
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = fake_val

        # Also annotate s1 with metadata so s2 can check its input's shape
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops.aten.slice.Tensor
            ):
                node.meta["val"] = fake_val
                break  # Only need the first slice node (s1)

        self.assertEqual(self._count_slice_nodes(gm), 2)
        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 0)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
