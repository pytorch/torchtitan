# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from unittest.mock import patch

import torch
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch.cuda._graph_annotations import _is_tools_id_unavailable
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase
from torch.utils.checkpoint import checkpoint, CheckpointPolicy

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    _AC_REGION_ID,
    _MODULE_FQN,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.graph_utils import export_joint
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    trace_train_step,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_sac_pass,
    insert_kernel_annotations_pass,
    reassign_to_pg_pass,
    remove_detach_pass,
    remove_identity_slice_pass,
    remove_identity_view_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.graph_trainer.tests.test_custom_codegen import (  # noqa: F401
    TestCustomCodegenPass,
)
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

                    def _make_tuple(x):
                        return (x, x)

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



class TestCpuOffloadPass(TestCase):
    """Unit tests for the CPU offload pass on synthetic FX graphs.

    Tests use hand-built graphs that simulate the joint fwd+bwd structure
    produced by make_fx, with ``seq_nr`` for forward/backward classification
    and ``ac_region_id`` for layer boundaries.
    """

    def _make_fake_val(self, shape=(64, 64), dtype=torch.float32, device="cuda:0"):
        """Create a real tensor for use as node.meta["val"].

        The offload pass reads .nelement(), .element_size(), .is_contiguous(),
        .device, and .to() from node metadata values.
        """
        return torch.empty(shape, dtype=dtype, device=device)

    def _build_joint_graph(self, num_layers=3, ops_per_layer=2):
        """Build a synthetic joint fwd+bwd graph with layer annotations.

        Structure per layer:
          Forward: mm -> relu (seq_nr = layer_id * ops_per_layer + op_idx)
          Backward: mm_bwd -> relu_bwd (same seq_nr as their fwd counterparts)

        All forward nodes get ``ac_region_id = layer_id`` in their custom metadata.
        Backward nodes also get ``ac_region_id`` (simulating _copy_fwd_metadata_to_bw_nodes).

        Forward mm nodes have backward consumers (relu_bwd uses fwd mm output),
        making them eligible for offloading.

        Returns:
            (gm, fwd_nodes, bwd_nodes) where fwd/bwd_nodes are lists of nodes
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        fwd_nodes = []
        bwd_nodes = []
        last_fwd = x

        # Forward pass: layer 0 -> layer N-1
        for layer_id in range(num_layers):
            for op_idx in range(ops_per_layer):
                seq_nr = layer_id * ops_per_layer + op_idx
                if op_idx == 0:
                    node = graph.call_function(
                        torch.ops.aten.mm.default, args=(last_fwd, last_fwd)
                    )
                else:
                    node = graph.call_function(
                        torch.ops.aten.relu.default, args=(last_fwd,)
                    )
                node.meta["seq_nr"] = seq_nr
                node.meta["custom"] = {"ac_region_id": layer_id}
                node.meta["val"] = self._make_fake_val()
                fwd_nodes.append(node)
                last_fwd = node

        # Backward pass: layer N-1 -> layer 0
        last_bwd = last_fwd
        for layer_id in reversed(range(num_layers)):
            for op_idx in reversed(range(ops_per_layer)):
                seq_nr = layer_id * ops_per_layer + op_idx
                # Backward node: uses the corresponding forward node's output
                fwd_idx = layer_id * ops_per_layer + op_idx
                fwd_node = fwd_nodes[fwd_idx]
                node = graph.call_function(
                    torch.ops.aten.mm.default, args=(last_bwd, fwd_node)
                )
                node.meta["seq_nr"] = seq_nr
                node.meta["custom"] = {"ac_region_id": layer_id}
                node.meta["val"] = self._make_fake_val()
                bwd_nodes.append(node)
                last_bwd = node

        graph.output(last_bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        return gm, fwd_nodes, bwd_nodes

    def test_tag_all_offloadable_activations_basic(self):
        """Forward nodes with backward consumers should be tagged MUST_CPU_OFFLOAD."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        gm, fwd_nodes, bwd_nodes = self._build_joint_graph(num_layers=3)
        tag_all_offloadable_activations(gm)

        # Last layer (layer 2) should NOT be tagged (skipped)
        # Layers 0 and 1 should have some tagged nodes
        tagged_nodes = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        ]
        self.assertGreater(len(tagged_nodes), 0, "Expected some nodes to be tagged")

        # Verify no last-layer nodes are tagged
        for node in tagged_nodes:
            layer_id = node.meta.get("custom", {}).get("ac_region_id")
            self.assertNotEqual(
                layer_id, 2, "Last layer nodes should not be tagged for offload"
            )

    def test_tag_no_backward_consumers(self):
        """Forward nodes without backward consumers should NOT be tagged."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        # Build a graph with only forward nodes (no backward)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        node1 = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        node1.meta["seq_nr"] = 0
        node1.meta["custom"] = {"ac_region_id": 0}
        node1.meta["val"] = self._make_fake_val()

        node2 = graph.call_function(torch.ops.aten.relu.default, args=(node1,))
        node2.meta["seq_nr"] = 1
        node2.meta["custom"] = {"ac_region_id": 0}
        node2.meta["val"] = self._make_fake_val()

        graph.output(node2)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tag_all_offloadable_activations(gm)

        tagged = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        ]
        self.assertEqual(
            len(tagged), 0, "No nodes should be tagged without bwd consumers"
        )

    def test_offload_pass_noop_when_no_tags(self):
        """cpu_offload_pass should be a no-op when no nodes are tagged."""
        from torchtitan.experiments.graph_trainer.cpu_offload import cpu_offload_pass

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()
        node = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        node.meta["seq_nr"] = 0
        node.meta["val"] = self._make_fake_val()
        graph.output(node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        node_count_before = len(list(gm.graph.nodes))
        gm = cpu_offload_pass(gm)
        node_count_after = len(list(gm.graph.nodes))

        self.assertEqual(
            node_count_before, node_count_after, "No-op pass should not add nodes"
        )

    def test_offload_reload_ops_inserted(self):
        """Verify offload/reload/wait_tensor ops are inserted for tagged nodes."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        gm, fwd_nodes, bwd_nodes = self._build_joint_graph(num_layers=3)
        tag_all_offloadable_activations(gm)

        # Count tagged nodes before applying offload pass
        tagged_count = sum(
            1
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        )
        self.assertGreater(tagged_count, 0)

        gm = apply_cpu_offload_pass(gm)

        # Verify offload, reload, and wait_tensor ops were inserted
        offload_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.offload.default
        ]
        reload_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.reload.default
        ]
        wait_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.wait_tensor.default
        ]

        self.assertGreater(len(offload_ops), 0, "Expected offload ops")
        self.assertGreater(len(reload_ops), 0, "Expected reload ops")
        # Each offloaded tensor gets 2 waits: one for offload, one for reload
        self.assertEqual(
            len(wait_ops),
            len(offload_ops) + len(reload_ops),
            "Each offload/reload should have a matching wait_tensor",
        )

    def test_view_ops_not_tagged(self):
        """View ops should never be tagged for offload."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        # Forward: mm -> view (layer 0)
        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        mm.meta["seq_nr"] = 0
        mm.meta["custom"] = {"ac_region_id": 0}
        mm.meta["val"] = self._make_fake_val()

        view = graph.call_function(torch.ops.aten.view.default, args=(mm, [64, 64]))
        view.meta["seq_nr"] = 1
        view.meta["custom"] = {"ac_region_id": 0}
        view.meta["val"] = self._make_fake_val()

        # Forward layer 1 to avoid single-layer skip
        mm2 = graph.call_function(torch.ops.aten.mm.default, args=(view, view))
        mm2.meta["seq_nr"] = 2
        mm2.meta["custom"] = {"ac_region_id": 1}
        mm2.meta["val"] = self._make_fake_val()

        # Backward: uses view output
        bwd = graph.call_function(torch.ops.aten.mm.default, args=(mm2, view))
        bwd.meta["seq_nr"] = 1  # same seq_nr as view -> backward
        bwd.meta["custom"] = {"ac_region_id": 0}
        bwd.meta["val"] = self._make_fake_val()

        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tag_all_offloadable_activations(gm)

        # The view node should not be tagged
        view_node = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.view.default
        ]
        self.assertEqual(len(view_node), 1)
        self.assertIsNot(
            view_node[0].meta.get("recompute"),
            CheckpointPolicy.MUST_CPU_OFFLOAD,
            "View ops should not be tagged for offload",
        )

    def test_small_tensors_not_tagged(self):
        """Tensors smaller than _MIN_OFFLOAD_BYTES should not be tagged."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        # Small tensor: 2x2 float32 = 16 bytes < 4096
        small_val = self._make_fake_val(shape=(2, 2))
        x.meta["val"] = small_val

        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        mm.meta["seq_nr"] = 0
        mm.meta["custom"] = {"ac_region_id": 0}
        mm.meta["val"] = small_val

        mm2 = graph.call_function(torch.ops.aten.mm.default, args=(mm, mm))
        mm2.meta["seq_nr"] = 1
        mm2.meta["custom"] = {"ac_region_id": 1}
        mm2.meta["val"] = small_val

        bwd = graph.call_function(torch.ops.aten.mm.default, args=(mm2, mm))
        bwd.meta["seq_nr"] = 0
        bwd.meta["custom"] = {"ac_region_id": 0}
        bwd.meta["val"] = small_val

        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tag_all_offloadable_activations(gm)

        tagged = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        ]
        self.assertEqual(len(tagged), 0, "Small tensors should not be tagged")

    def test_single_layer_tagged(self):
        """With only one layer, nodes are still tagged (last-layer skip only applies with multiple layers)."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        gm, _, _ = self._build_joint_graph(num_layers=1)
        tag_all_offloadable_activations(gm)

        tagged = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        ]
        self.assertGreater(
            len(tagged), 0, "Single layer nodes can be tagged (last_layer_id=None)"
        )


class TestAnnotateModuleFqns(TestCase):
    """Unit tests for annotate_module_fqns and insert_kernel_annotations_pass."""

    def _trace_and_get_fqns(self, model, *args):
        """Trace fwd+bwd with trace_train_step and return module_fqn annotations."""

        def fwd_step(model, *inputs):
            pred = model(inputs[0])
            loss = pred.sum()
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params)
            return [loss] + list(grads)

        traced = trace_train_step(fwd_step)(model, *args)
        fqns = set()
        for node in traced.gm.graph.nodes:
            fqn = (node.meta.get("custom") or {}).get(_MODULE_FQN)
            if fqn:
                fqns.add(fqn)
        return fqns

    def test_annotate_transformer_like_model(self):
        """Module FQNs survive trace_train_step for a transformer-like model
        with distinct submodule classes (norm, attention, ffn)."""

        class Norm(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm = torch.nn.LayerNorm(dim)

            def forward(self, x):
                return self.norm(x)

        class Attention(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.wq = torch.nn.Linear(dim, dim, bias=False)
                self.wo = torch.nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.wo(torch.relu(self.wq(x)))

        class FFN(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.w1 = torch.nn.Linear(dim, dim * 2, bias=False)
                self.w2 = torch.nn.Linear(dim * 2, dim, bias=False)

            def forward(self, x):
                return self.w2(torch.relu(self.w1(x)))

        class TransformerBlock(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.attention_norm = Norm(dim)
                self.attention = Attention(dim)
                self.ffn_norm = Norm(dim)
                self.feed_forward = FFN(dim)

            def forward(self, x):
                h = x + self.attention(self.attention_norm(x))
                return h + self.feed_forward(self.ffn_norm(h))

        class Model(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.layer = TransformerBlock(dim)

            def forward(self, x):
                return self.layer(x)

        dim = 16
        model = Model(dim)
        annotate_module_fqns(model)
        fqns = self._trace_and_get_fqns(model, torch.randn(4, dim))

        # Verify key module paths are present.  The Norm wrapper has no
        # ops of its own, so its inner LayerNorm gets the deepest path.
        self.assertIn("layer.attention_norm.norm", fqns)
        self.assertIn("layer.attention", fqns)
        self.assertIn("layer.attention.wq", fqns)
        self.assertIn("layer.attention.wo", fqns)
        self.assertIn("layer.ffn_norm.norm", fqns)
        self.assertIn("layer.feed_forward", fqns)
        self.assertIn("layer.feed_forward.w1", fqns)
        self.assertIn("layer.feed_forward.w2", fqns)

    def test_same_class_instances_get_distinct_fqns(self):
        """Two parameterless instances of the same class get distinct fqns.

        Uses minimal_fx_tracer directly (not trace_train_step) because
        parameterless models cannot produce gradients via autograd.grad.
        """

        class Block(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Block()
                self.b = Block()

            def forward(self, x):
                return self.b(self.a(x))

        model = Model()
        annotate_module_fqns(model)

        def fwd_only(state, x):
            return model(x)

        traced = minimal_fx_tracer(fwd_only)({}, torch.randn(4))
        fqns = set()
        for node in traced.gm.graph.nodes:
            fqn = (node.meta.get("custom") or {}).get(_MODULE_FQN)
            if fqn:
                fqns.add(fqn)

        self.assertIn("a", fqns)
        self.assertIn("b", fqns)

    def test_same_class_parameterless_works_with_make_fx(self):
        """Same-class parameterless instances get distinct fqns with plain make_fx."""

        class Block(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Block()
                self.b = Block()

            def forward(self, x):
                return self.b(self.a(x))

        model = Model()
        annotate_module_fqns(model)

        with preserve_node_meta():
            gm = make_fx(model)(torch.randn(4))

        fqns = set()
        for node in gm.graph.nodes:
            fqn = (node.meta.get("custom") or {}).get(_MODULE_FQN)
            if fqn:
                fqns.add(fqn)

        self.assertIn("a", fqns)
        self.assertIn("b", fqns)

    def test_same_class_instances_with_params_get_distinct_fqns(self):
        """Two instances of the same class with parameters get distinct fqns."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(4, 4)
                self.b = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.b(self.a(x))

        model = Model()
        annotate_module_fqns(model)
        fqns = self._trace_and_get_fqns(model, torch.randn(2, 4))

        self.assertIn("a", fqns)
        self.assertIn("b", fqns)

    def _build_annotated_gm(self):
        """Build a GraphModule with module_fqn annotations on its nodes."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        n1 = graph.call_function(torch.relu, (x,))
        n1.meta["custom"] = {_MODULE_FQN: "attn"}
        n2 = graph.call_function(torch.sigmoid, (n1,))
        n2.meta["custom"] = {_MODULE_FQN: "attn"}
        n3 = graph.call_function(torch.tanh, (n2,))
        n3.meta["custom"] = {_MODULE_FQN: "ffn"}
        graph.output(n3)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def test_insert_kernel_annotations_pass_inserts_calls(self):
        """When tools ID is available, the pass inserts enter/exit calls."""
        if _is_tools_id_unavailable():
            self.skipTest("cudaGraphNodeGetToolsId not available")

        gm = self._build_annotated_gm()
        num_before = sum(1 for n in gm.graph.nodes if n.op == "call_function")

        insert_kernel_annotations_pass(gm)

        num_after = sum(1 for n in gm.graph.nodes if n.op == "call_function")
        # 2 scopes (attn, ffn) = 2 enters + 2 exits = 4 new nodes
        self.assertEqual(num_after - num_before, 4)

    def test_insert_kernel_annotations_pass_noop_when_unavailable(self):
        """When tools ID is unavailable, the pass leaves the graph unchanged."""
        gm = self._build_annotated_gm()
        num_before = len(list(gm.graph.nodes))

        with patch(
            "torch.cuda._graph_annotations._is_tools_id_unavailable",
            return_value=True,
        ):
            insert_kernel_annotations_pass(gm)

        num_after = len(list(gm.graph.nodes))
        self.assertEqual(num_before, num_after)

    def test_insert_kernel_annotations_pass_noop_without_metadata(self):
        """The pass should not insert anything when no custom metadata exists."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        n1 = graph.call_function(torch.relu, (x,))
        graph.output(n1)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        num_before = len(list(gm.graph.nodes))
        insert_kernel_annotations_pass(gm)
        num_after = len(list(gm.graph.nodes))

        self.assertEqual(num_before, num_after)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
