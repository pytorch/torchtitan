# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch.cuda._graph_annotations import _is_tools_id_unavailable
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.traceback import preserve_node_meta
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase
from torch.utils.checkpoint import checkpoint, CheckpointPolicy

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    _EP_TOKEN_EXCHANGE,
    _MODULE_FQN,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapConfig,
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.cudagraph import (
    insert_kernel_annotations_pass,
    is_cudagraphable,
    is_full_cudagraphable,
)
from torchtitan.experiments.graph_trainer.ep_chunk_pass import (
    _chunk_copied_meta,
    _materialize_symint_arg,
    _Region,
    _rewrite_chunk_symint,
    apply_chunk_pass,
    ep_overlap_chunk_pass,
    mark_chunk_dynamic_dims,
    populate_chunk_dim_metadata_pass,
    prepare_ep_overlap_trace_call_inputs,
    prepare_ep_overlap_trace_inputs,
)
from torchtitan.experiments.graph_trainer.ep_eager_chunk import (
    maybe_apply_ep_overlap_eager_chunking,
    populate_eager_chunk_metadata_pass,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    CHUNK_SYMBOL_HINTS_META,
    concretize_ep_chunk_symbolic_shapes_pass,
)
from torchtitan.experiments.graph_trainer.ep_process_group_pass import (
    isolate_ep_process_group_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    reassign_collective_pgs_pass,
)
from torchtitan.experiments.graph_trainer.graph_utils import export_joint
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _make_default_memory_policy,
    _make_full_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.passes import (
    compile_time_passes,
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    canonicalize_graph_pass,
    eliminate_dead_code_pass,
    normalize_view_ops_as_reshape,
    remove_b2b_transpose_pass,
    remove_detach_pass,
    remove_identity_slice_pass,
    remove_identity_view_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.graph_trainer.tests.test_cpu_offload import (  # noqa: F401
    TestCpuOffloadPass,
)
from torchtitan.experiments.graph_trainer.tests.test_custom_codegen import (  # noqa: F401
    TestCustomCodegenPass,
)
from torchtitan.experiments.graph_trainer.tests.test_performance_passes import (  # noqa: F401
    TestAnnotateRMSNormForRegionalInductorPass,
)
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.module import Module, ModuleList


class TestDefaultTransformerBlockBuckets(TestCase):
    def test_compile_time_passes_enable_chunked_loss_bucket_only_when_needed(self):
        from torchtitan.components.loss import ChunkedCELoss, CrossEntropyLoss
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.passes import compile_time_passes

        def make_config(loss):
            return SimpleNamespace(
                compile=GraphTrainerCompileConfig(inductor_compilation="full"),
                loss=loss,
                model_spec=SimpleNamespace(model=SimpleNamespace(layers=[0, 1])),
                parallelism=SimpleNamespace(enable_async_tensor_parallel=False),
            )

        traced_result = SimpleNamespace(state_fqns=[])
        with patch(
            "torchtitan.experiments.graph_trainer.common_utils."
            "get_default_transformer_block_buckets",
            return_value=[],
        ) as mock_bucket_plan:
            compile_time_passes(traced_result, make_config(CrossEntropyLoss.Config()))
            compile_time_passes(traced_result, make_config(ChunkedCELoss.Config()))

        self.assertEqual(
            [
                call.kwargs["chunked_loss_enabled"]
                for call in mock_bucket_plan.call_args_list
            ],
            [False, True],
        )


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


class TestReassignCollectivePgsPass(FSDPTest):
    """Integration tests: toy model + simple_fsdp + export_joint + reassign_collective_pgs_pass."""

    def _setup(self):
        """Set up ParallelDims and device mesh for FSDP."""
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
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

    def _count_ep_a2a_nodes_with_pg(self, gm, pg_name):
        return sum(
            1
            for node in gm.graph.nodes
            if node.op == "call_function"
            and "all_to_all_single" in str(node.target)
            and node.args[3] == pg_name
        )

    def test_overlap_rewrites_ag_nodes(self):
        """Apply reassign_collective_pgs_pass on the real backward graph and verify
        that FSDP AG nodes are rewritten to the auto-created extra PG."""
        from torchtitan.experiments.graph_trainer.fsdp_passes import (
            _EXTRA_FSDP_PG_REGISTRY,
        )

        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        # Before: all AG nodes should use the FSDP PG
        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertGreater(ag_before, 0, "Expected AG nodes with FSDP PG name")

        _EXTRA_FSDP_PG_REGISTRY.pop(fsdp_pg_name, None)
        reassign_collective_pgs_pass(bw_gm, bw_example_inputs)

        extra_pg_name = _EXTRA_FSDP_PG_REGISTRY[fsdp_pg_name]
        ag_with_old = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        ag_with_new = self._count_ag_nodes_with_pg(bw_gm, extra_pg_name)

        self.assertEqual(ag_with_old, 0, "No AG nodes should still use the old PG")
        self.assertEqual(
            ag_with_new,
            ag_before,
            "All AG nodes should now use the extra PG",
        )

    def test_overlap_preserves_total_ag_count(self):
        """The pass should not add or remove AG nodes, only rewrite PG names."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        total_before = self._count_all_ag_nodes(bw_gm)
        reassign_collective_pgs_pass(bw_gm, bw_example_inputs)
        total_after = self._count_all_ag_nodes(bw_gm)

        self.assertEqual(total_before, total_after)

    def test_overlap_rewrites_multiple_pgs(self):
        """When the graph has AG nodes from multiple FSDP PGs (e.g. FSDP +
        expert-FSDP), each source PG should be mapped to its own extra PG."""
        import torch.distributed as dist

        from torchtitan.experiments.graph_trainer.fsdp_passes import (
            _EXTRA_FSDP_PG_REGISTRY,
        )

        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        # Create a second PG to simulate expert-FSDP.
        second_pg = dist.new_group(
            ranks=list(range(self.world_size)),
            use_local_synchronization=True,
        )
        second_pg_name = second_pg.group_name

        # Rewrite half the AG nodes to use the second PG.
        ag_nodes = [n for n in bw_gm.graph.nodes if is_all_gather(n)]
        self.assertGreater(len(ag_nodes), 1)
        half = len(ag_nodes) // 2
        for node in ag_nodes[:half]:
            node.args = (node.args[0], node.args[1], second_pg_name)

        ag_pg1_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        ag_pg2_before = self._count_ag_nodes_with_pg(bw_gm, second_pg_name)
        self.assertGreater(ag_pg1_before, 0)
        self.assertGreater(ag_pg2_before, 0)

        _EXTRA_FSDP_PG_REGISTRY.pop(fsdp_pg_name, None)
        _EXTRA_FSDP_PG_REGISTRY.pop(second_pg_name, None)
        reassign_collective_pgs_pass(bw_gm, bw_example_inputs)

        # Both source PGs should have their own extra PG.
        self.assertIn(fsdp_pg_name, _EXTRA_FSDP_PG_REGISTRY)
        self.assertIn(second_pg_name, _EXTRA_FSDP_PG_REGISTRY)
        extra_pg1 = _EXTRA_FSDP_PG_REGISTRY[fsdp_pg_name]
        extra_pg2 = _EXTRA_FSDP_PG_REGISTRY[second_pg_name]
        self.assertNotEqual(
            extra_pg1, extra_pg2, "Each source PG must map to a distinct extra PG"
        )

        # No AG nodes should still use original PGs.
        self.assertEqual(self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name), 0)
        self.assertEqual(self._count_ag_nodes_with_pg(bw_gm, second_pg_name), 0)

        # All AG nodes should use their respective extra PGs.
        self.assertEqual(self._count_ag_nodes_with_pg(bw_gm, extra_pg1), ag_pg1_before)
        self.assertEqual(self._count_ag_nodes_with_pg(bw_gm, extra_pg2), ag_pg2_before)

    def test_overlap_rewrites_ep_a2a_on_fsdp_pg_to_separate_pg(self):
        from torchtitan.experiments.graph_trainer.ep_process_group_pass import (
            _EXTRA_EP_PG_REGISTRY,
        )
        from torchtitan.experiments.graph_trainer.fsdp_passes import (
            _EXTRA_FSDP_PG_REGISTRY,
        )

        self._setup()
        fsdp_pg_name = self._get_fsdp_pg_name()
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        ag = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, fsdp_pg_name)
        )
        wait = graph.call_function(c10d.wait_tensor.default, args=(ag,))
        rs = graph.call_function(
            c10d.reduce_scatter_tensor.default, args=(x, "sum", 1, fsdp_pg_name)
        )
        rs_wait = graph.call_function(c10d.wait_tensor.default, args=(rs,))
        a2a = graph.call_function(
            c10d.all_to_all_single.default, args=(x, [], [], fsdp_pg_name)
        )
        a2a.meta["custom"] = {
            _MODULE_FQN: "layers.0.moe",
            "EP": "dispatch",
            _EP_TOKEN_EXCHANGE: "dispatch",
        }
        graph.output((wait, rs_wait, a2a))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        _EXTRA_FSDP_PG_REGISTRY.pop(fsdp_pg_name, None)
        _EXTRA_EP_PG_REGISTRY.pop(fsdp_pg_name, None)
        reassign_collective_pgs_pass(gm, ())
        isolate_ep_process_group_pass(gm, ())

        fsdp_extra_pg = _EXTRA_FSDP_PG_REGISTRY[fsdp_pg_name]
        ep_extra_pg = _EXTRA_EP_PG_REGISTRY[fsdp_pg_name]
        self.assertNotEqual(fsdp_extra_pg, ep_extra_pg)
        self.assertEqual(self._count_ag_nodes_with_pg(gm, fsdp_extra_pg), 1)
        self.assertEqual(self._count_ep_a2a_nodes_with_pg(gm, ep_extra_pg), 1)

    def test_overlap_preserves_distinct_ep_pg_with_same_fsdp_ranks(self):
        import torch.distributed as dist

        from torchtitan.experiments.graph_trainer.ep_process_group_pass import (
            _EXTRA_EP_PG_REGISTRY,
        )
        from torchtitan.experiments.graph_trainer.fsdp_passes import (
            _EXTRA_FSDP_PG_REGISTRY,
        )

        self._setup()
        fsdp_pg_name = self._get_fsdp_pg_name()
        ep_pg = dist.new_group(
            ranks=list(range(self.world_size)),
            use_local_synchronization=True,
        )
        ep_pg_name = ep_pg.group_name

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        ag = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, fsdp_pg_name)
        )
        wait = graph.call_function(c10d.wait_tensor.default, args=(ag,))
        rs = graph.call_function(
            c10d.reduce_scatter_tensor.default, args=(x, "sum", 1, fsdp_pg_name)
        )
        rs_wait = graph.call_function(c10d.wait_tensor.default, args=(rs,))
        a2a = graph.call_function(
            c10d.all_to_all_single.default, args=(x, [], [], ep_pg_name)
        )
        a2a.meta["custom"] = {
            _MODULE_FQN: "layers.0.moe",
            "EP": "combine",
            _EP_TOKEN_EXCHANGE: "combine",
        }
        graph.output((wait, rs_wait, a2a))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        _EXTRA_FSDP_PG_REGISTRY.pop(fsdp_pg_name, None)
        _EXTRA_EP_PG_REGISTRY.pop(ep_pg_name, None)
        reassign_collective_pgs_pass(gm, ())
        isolate_ep_process_group_pass(gm, ())

        self.assertNotIn(ep_pg_name, _EXTRA_EP_PG_REGISTRY)
        self.assertEqual(self._count_ep_a2a_nodes_with_pg(gm, ep_pg_name), 1)

    def test_ep_pg_pass_rewrites_all_ep_a2a_on_tp_pg_to_separate_pg(self):
        from torchtitan.experiments.graph_trainer.ep_process_group_pass import (
            _EXTRA_EP_PG_REGISTRY,
        )

        self._setup()
        tp_pg_name = self._get_fsdp_pg_name()
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        ag = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, tp_pg_name)
        )
        a2a = graph.call_function(
            c10d.all_to_all_single.default, args=(x, [], [], tp_pg_name)
        )
        a2a.meta["custom"] = {
            _MODULE_FQN: "layers.0.moe",
            "EP": "dispatch",
        }
        generic_ep_a2a = graph.call_function(
            c10d.all_to_all_single.default, args=(x, [], [], tp_pg_name)
        )
        generic_ep_a2a.meta["custom"] = {
            _MODULE_FQN: "layers.0.moe",
            "EP": "combine",
        }
        graph.output((ag, a2a, generic_ep_a2a))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        _EXTRA_EP_PG_REGISTRY.pop(tp_pg_name, None)
        isolate_ep_process_group_pass(gm, ())

        ep_extra_pg = _EXTRA_EP_PG_REGISTRY[tp_pg_name]
        self.assertEqual(self._count_ep_a2a_nodes_with_pg(gm, ep_extra_pg), 2)

    def test_overlap_is_noop_when_no_fsdp_ag(self):
        """If the graph has no FSDP all-gathers, the pass is a no-op."""
        self._setup()
        # Plain (non-FSDP) module: a graph without FSDP all-gathers.
        gm = torch.fx.symbolic_trace(torch.nn.Linear(4, 4))
        ag_before = self._count_all_ag_nodes(gm)
        reassign_collective_pgs_pass(gm, ())
        ag_after = self._count_all_ag_nodes(gm)
        self.assertEqual(ag_before, 0)
        self.assertEqual(ag_after, 0)


class TestApplySACPass(TestCase):
    """Unit tests for the tag_sac_policy joint graph pass."""

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
        tag_sac_policy(gm)
        for node in self._get_call_function_nodes(gm):
            self.assertEqual(node.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)

    def test_save_ops_marked_must_save(self):
        """Non-mm ops in the save list should be marked MUST_SAVE."""
        custom_save = {torch.ops.aten.add.Tensor}
        gm = self._build_gm([torch.ops.aten.add.Tensor])
        tag_sac_policy(gm, policy_fn=_make_default_memory_policy(custom_save))
        nodes = self._get_call_function_nodes(gm)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_sym_size_ops_always_saved(self):
        """Sym-int nodes are forced MUST_SAVE regardless of policy: recomputing a
        shape read would pin the parent tensor alive just to reread its size."""
        # sym_size produces a SymInt only for symbolic dims, so tag the node's
        # meta["val"] with a real SymInt — that is what is_sym_node keys off.
        shape_env = ShapeEnv()
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        sym = graph.call_function(torch.ops.aten.sym_size.int, args=(relu, 0))
        sym.meta["val"] = shape_env.create_unbacked_symint()
        graph.output(relu)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # Even a recompute-everything policy must save sym_size.
        tag_sac_policy(gm, policy_fn=_make_full_memory_policy())

        tags = {
            n.target: n.meta["recompute"]
            for n in gm.graph.nodes
            if n.op == "call_function"
        }
        self.assertEqual(tags[torch.ops.aten.sym_size.int], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(
            tags[torch.ops.aten.relu.default], CheckpointPolicy.MUST_RECOMPUTE
        )

    def test_getitem_propagates_parent_tags(self):
        """operator.getitem nodes should inherit the parent's recompute tag."""
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

        tag_sac_policy(gm)

        tuple_node = nodes[1]
        getitem_node = nodes[2]
        self.assertEqual(getitem_node.meta["recompute"], tuple_node.meta["recompute"])

    def test_wait_tensor_propagates_parent_tags(self):
        """wait_tensor nodes should inherit the parent's recompute tag."""
        custom_save = {torch.ops._c10d_functional.reduce_scatter_tensor.default}
        gm = self._build_gm(
            [
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                torch.ops._c10d_functional.wait_tensor.default,
            ]
        )
        nodes = self._get_call_function_nodes(gm)
        nodes[0].meta["custom"] = {_MODULE_FQN: "layers.3.attention"}

        tag_sac_policy(gm, policy_fn=_make_default_memory_policy(custom_save))

        rs_node = nodes[0]
        wait_node = nodes[1]
        self.assertEqual(rs_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(wait_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_boundary_nodes_forced_to_must_save(self):
        """Nodes at AC region boundaries should be forced to MUST_SAVE."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        nodes = self._get_call_function_nodes(gm)
        nodes[0].meta["custom"] = {_MODULE_FQN: "layers.0.feed_forward"}
        nodes[1].meta["custom"] = {_MODULE_FQN: "layers.1.attention"}

        tag_sac_policy(gm)

        # add is at the boundary (layer 0 -> layer 1), forced to MUST_SAVE
        self.assertEqual(nodes[0].meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(nodes[1].meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)

    def test_custom_op_list_to_save(self):
        """A custom op_list_to_save should override the defaults."""
        custom_save = {torch.ops.aten.relu.default}
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        tag_sac_policy(gm, policy_fn=_make_default_memory_policy(custom_save))
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
                torch.ops.aten.mm.default,  # in save list -> MUST_SAVE
                torch.ops.aten.max.default,  # in save list -> MUST_SAVE
                torch.ops.aten.mm.default,  # in save list -> MUST_SAVE
                torch.ops.aten.add.Tensor,  # not in save list -> PREFER_RECOMPUTE
                torch.ops.aten.mm.default,  # in save list -> MUST_SAVE
            ]
        )
        tag_sac_policy(gm, policy_fn=_make_default_memory_policy(custom_save))
        nodes = self._get_call_function_nodes(gm)
        expected = [
            (torch.ops.aten.mm.default, CheckpointPolicy.MUST_SAVE),
            (torch.ops.aten.max.default, CheckpointPolicy.MUST_SAVE),
            (torch.ops.aten.mm.default, CheckpointPolicy.MUST_SAVE),
            (torch.ops.aten.add.Tensor, CheckpointPolicy.PREFER_RECOMPUTE),
            (torch.ops.aten.mm.default, CheckpointPolicy.MUST_SAVE),
        ]
        self.assertEqual(len(nodes), len(expected))
        for node, (target, policy) in zip(nodes, expected):
            self.assertEqual(node.target, target)
            self.assertEqual(node.meta["recompute"], policy, f"node {node.name}")

    def test_remat_uses_autograd_backward_without_phase_annotation(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        fwd = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
        bwd = graph.call_function(torch.ops.aten.mul.Tensor, args=(fwd, 2))
        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fwd.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        gm = selective_activation_remat_pass(gm)

        recomputed_nodes = [
            node for node in gm.graph.nodes if node.name == "add_tensor_recomputed"
        ]
        self.assertEqual(len(recomputed_nodes), 1)
        self.assertTrue(recomputed_nodes[0].meta["autograd_backward"])

    def test_remat_dup_gets_independent_custom_meta(self):
        # fx.Graph.node_copy shallow-copies node.meta, so without intervention a
        # recompute dup shares the SAME nested meta["custom"] dict as its forward
        # original -- annotating one would silently mutate the other. The pass must
        # give the dup its own copy (preserving the values).
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        fwd = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
        # A forward consumer keeps the original alive (remat erases originals whose
        # consumers are all backward) so we can compare it against the dup.
        fwd_use = graph.call_function(torch.ops.aten.relu.default, args=(fwd,))
        bwd = graph.call_function(torch.ops.aten.mul.Tensor, args=(fwd, 2))
        graph.output((fwd_use, bwd))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fwd.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        fwd.meta["custom"] = {_MODULE_FQN: "layers.0.attention_norm"}
        bwd.meta["autograd_backward"] = True

        gm = selective_activation_remat_pass(gm)

        fwd_node = next(n for n in gm.graph.nodes if n.name == "add_tensor")
        dup = next(n for n in gm.graph.nodes if n.name == "add_tensor_recomputed")
        # Independent dict object, same values preserved.
        self.assertIsNot(dup.meta["custom"], fwd_node.meta["custom"])
        self.assertEqual(dup.meta["custom"][_MODULE_FQN], "layers.0.attention_norm")
        # Mutating the dup's annotation must not leak into the original.
        dup.meta["custom"]["cudagraph_partition"] = "cudagraph_9"
        self.assertNotIn("cudagraph_partition", fwd_node.meta["custom"])


class TestFullMemoryPolicy(TestCase):
    """Unit tests for the full recompute memory policy."""

    def _build_gm(self, op_targets, layer_fqns=None):
        """Build a GraphModule with call_function nodes and optional layer FQNs."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        last = x
        nodes = []
        for target in op_targets:
            last = graph.call_function(target, args=(last, y))
            nodes.append(last)
        graph.output(last)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        if layer_fqns:
            for node, fqn in zip(nodes, layer_fqns):
                if fqn is not None:
                    node.meta["custom"] = {_MODULE_FQN: fqn}
        return gm

    def _get_call_function_nodes(self, gm):
        return [n for n in gm.graph.nodes if n.op == "call_function"]

    def test_all_ops_marked_recompute(self):
        """All ops should be marked MUST_RECOMPUTE with full policy."""
        gm = self._build_gm(
            [
                torch.ops.aten.mm.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        tag_sac_policy(gm, policy_fn=_make_full_memory_policy())
        for node in self._get_call_function_nodes(gm):
            self.assertEqual(
                node.meta["recompute"],
                CheckpointPolicy.MUST_RECOMPUTE,
                f"node {node.name} should be MUST_RECOMPUTE",
            )

    def test_save_ops_also_recomputed(self):
        """Compute-intensive ops (linear, max) are recomputed under full."""
        gm = self._build_gm(
            [
                torch.ops.aten.linear.default,
                torch.ops.aten.max.default,
            ]
        )
        tag_sac_policy(gm, policy_fn=_make_full_memory_policy())
        for node in self._get_call_function_nodes(gm):
            self.assertEqual(
                node.meta["recompute"],
                CheckpointPolicy.MUST_RECOMPUTE,
            )

    def test_rng_ops_saved(self):
        """RNG ops (nondeterministic_seeded) are saved: the remat pass cannot
        replay their random state, unlike eager AC's preserve_rng_state."""
        policy_fn = _make_full_memory_policy()
        gm = self._build_gm([torch.ops.aten.native_dropout.default])
        node = self._get_call_function_nodes(gm)[0]
        self.assertEqual(policy_fn(node), CheckpointPolicy.MUST_SAVE)

    def test_higher_order_ops_recomputed(self):
        """Higher-order ops (flex_attention) are recomputed, not saved: the
        remat pass duplicates the HOP together with its subgraph get_attrs."""
        policy_fn = _make_full_memory_policy()
        gm = self._build_gm([torch.ops.higher_order.flex_attention])
        node = self._get_call_function_nodes(gm)[0]
        self.assertEqual(policy_fn(node), CheckpointPolicy.MUST_RECOMPUTE)

    def test_layer_boundary_forced_to_must_save(self):
        """Nodes at layer boundaries should still be forced to MUST_SAVE."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ],
            layer_fqns=[
                "layers.0.feed_forward",
                "layers.1.attention",
            ],
        )
        tag_sac_policy(gm, policy_fn=_make_full_memory_policy())
        nodes = self._get_call_function_nodes(gm)
        # add crosses from layer 0 to layer 1 — forced to MUST_SAVE
        self.assertEqual(nodes[0].meta["recompute"], CheckpointPolicy.MUST_SAVE)
        # relu has no higher-layer consumer — stays MUST_RECOMPUTE
        self.assertEqual(nodes[1].meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)

    def test_same_layer_nodes_all_recomputed(self):
        """Within a single layer, all ops should be recomputed."""
        gm = self._build_gm(
            [
                torch.ops.aten.mm.default,
                torch.ops.aten.linear.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ],
            layer_fqns=[
                "layers.0.attention",
                "layers.0.attention",
                "layers.0.feed_forward",
                "layers.0.feed_forward",
            ],
        )
        tag_sac_policy(gm, policy_fn=_make_full_memory_policy())
        for node in self._get_call_function_nodes(gm):
            self.assertEqual(
                node.meta["recompute"],
                CheckpointPolicy.MUST_RECOMPUTE,
                f"node {node.name} in single layer should be MUST_RECOMPUTE",
            )


class TestBucketingPrefetchOrder(FSDPTest):
    """Guard that SAC + bucketing produces correct all_gather prefetch order.

    Uses the real Llama3 debug model with FSDP via the GraphTrainer path.
    Verifies that bucketed all_gather starts follow forward layer order
    (0, 1, 2, ...) and not reverse order (which was a prior bug).
    """

    BATCH_SIZE = 4
    SEQ_LEN = 128

    @staticmethod
    def _get_bucketed_ag_layer_order(gm):
        """Extract layer IDs from bucketed all_gather_into_tensor_out nodes.

        For each bucketed all_gather, searches its transitive users for
        a node with module_fqn under ``layers.<N>`` and records N.
        Returns deduplicated layer IDs in graph order.
        """
        layer_ids = []
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if "all_gather_into_tensor_out" not in str(node.target):
                continue
            # BFS through users to find a node with layers.N FQN
            visited = set()
            queue = list(node.users)
            found_lid = None
            while queue and found_lid is None:
                u = queue.pop(0)
                if u in visited:
                    continue
                visited.add(u)
                fqn = u.meta.get("custom", {}).get(_MODULE_FQN, "")
                parts = fqn.split(".")
                if parts[0] == "layers" and len(parts) >= 2:
                    try:
                        found_lid = int(parts[1])
                    except ValueError:
                        pass
                else:
                    queue.extend(u.users)
            if found_lid is not None and (not layer_ids or layer_ids[-1] != found_lid):
                layer_ids.append(found_lid)
        return layer_ids

    def _run_and_get_layer_ids(self, fsdp_reshard_after_forward: str):
        """Run a single forward+backward step and return bucketed AG layer ids."""
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.experiments.graph_trainer.llama3 import (
            model_registry as llama3_model_registry,
        )
        from torchtitan.experiments.graph_trainer.llama3.parallelize import (
            annotate_llama,
        )
        from torchtitan.experiments.graph_trainer.simple_fsdp import (
            data_parallel,
            MixedPrecisionPolicy,
        )
        from torchtitan.experiments.graph_trainer.tests._trainer_test_utils import (
            build_minimal_trainer,
        )
        from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )

        model_spec = llama3_model_registry("debugmodel")
        model_config = model_spec.model
        vocab_size = model_config.vocab_size

        with torch.device("meta"):
            model = model_config.build()

        annotate_llama(model)
        fsdp_mesh = parallel_dims.get_mesh("fsdp")
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        model = data_parallel(
            model, device_mesh=fsdp_mesh, mode="fully_shard", mp_policy=mp_policy
        )
        model.to_empty(device="cuda")
        with torch.no_grad():
            model.init_states(buffer_device=None)
        model.train()

        # Use GraphTrainer's full path: trace + construct_default_graph_passes
        trainer = build_minimal_trainer(
            model,
            model_config,
            GraphTrainer,
            tokenizer=HuggingFaceTokenizer(tokenizer_path="./tests/assets/tokenizer"),
            fsdp_reshard_after_forward=fsdp_reshard_after_forward,
        )

        inputs = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device="cuda"
        )
        labels = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device="cuda"
        )
        # The dataloader always supplies per-document positions, which the
        # trainer requires to build the (block_causal) FlexAttention mask. A
        # single document (sequential positions) is fine here.
        positions = (
            torch.arange(self.SEQ_LEN, device="cuda", dtype=torch.int32)
            .unsqueeze(0)
            .expand(self.BATCH_SIZE, self.SEQ_LEN)
        )
        global_valid_tokens = torch.tensor(
            self.BATCH_SIZE * self.SEQ_LEN, dtype=torch.float, device="cuda"
        )

        # One forward_backward_step triggers _make_fx_forward_backward_step
        # which traces the model and applies all graph passes.
        trainer.forward_backward_step(
            input_dict={"input": inputs, "positions": positions},
            labels=labels,
            global_valid_tokens=global_valid_tokens,
        )

        layer_ids = self._get_bucketed_ag_layer_order(trainer._traced_step.gm)
        self.assertGreater(len(layer_ids), 0, "No layer all_gather nodes found")
        return layer_ids

    def test_forward_allgather_prefetch_follows_layer_order(self):
        """Without reshard-after-forward, all all_gathers are in forward and
        must appear in non-decreasing layer order 0 → N."""
        layer_ids = self._run_and_get_layer_ids(fsdp_reshard_after_forward="never")

        for i in range(1, len(layer_ids)):
            self.assertGreaterEqual(
                layer_ids[i],
                layer_ids[i - 1],
                f"Forward all_gather prefetch order violated: "
                f"layer {layer_ids[i]} before layer {layer_ids[i - 1]} "
                f"(full order: {layer_ids})",
            )

    def test_allgather_prefetch_with_reshard_after_forward(self):
        """With reshard-after-forward, backward also issues all_gathers.
        The graph-order sequence must be forward (0 → N) then backward (N → 0)."""
        layer_ids = self._run_and_get_layer_ids(fsdp_reshard_after_forward="always")

        # Split forward (ascending) and backward (descending) at the peak.
        peak = max(range(len(layer_ids)), key=lambda i: layer_ids[i])
        forward_ids = layer_ids[: peak + 1]
        backward_ids = layer_ids[peak:]

        # Backward all_gathers should exist when reshard-after-forward is on.
        self.assertGreater(
            len(backward_ids),
            1,
            f"Expected backward all_gathers with reshard-after-forward, "
            f"got order: {layer_ids}",
        )

        for i in range(1, len(forward_ids)):
            self.assertGreaterEqual(
                forward_ids[i],
                forward_ids[i - 1],
                f"Forward all_gather prefetch order violated: "
                f"layer {forward_ids[i]} before layer {forward_ids[i - 1]} "
                f"(full order: {layer_ids})",
            )

        for i in range(1, len(backward_ids)):
            self.assertLessEqual(
                backward_ids[i],
                backward_ids[i - 1],
                f"Backward all_gather prefetch order violated: "
                f"layer {backward_ids[i]} after layer {backward_ids[i - 1]} "
                f"(full order: {layer_ids})",
            )

    def test_drops_assert_async_and_dead_chain(self):
        # _assert_async is side-effectful, so plain DCE keeps it (and its whole
        # le/all condition chain). The pass erases the assert, then DCE reaps the
        # now-orphaned chain; unrelated live nodes are untouched.
        aten = torch.ops.aten
        g = torch.fx.Graph()
        x = g.placeholder("x")
        le = g.call_function(aten.le.Scalar, (x, 5))
        reduced = g.call_function(aten.all.default, (le,))
        g.call_function(aten._assert_async.msg, (reduced, "cond"))  # side-effect
        out = g.call_function(aten.relu.default, (x,))
        g.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)

        eliminate_dead_code_pass(gm)
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertNotIn(aten._assert_async.msg, targets)
        self.assertNotIn(aten.le.Scalar, targets)
        self.assertNotIn(aten.all.default, targets)
        self.assertIn(aten.relu.default, targets)


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


class TestChunkPasses(TestCase):
    def _assert_symbolic_dim_from_sources(self, actual, source, expected_hint: int):
        from torch.fx.experimental.symbolic_shapes import (
            free_symbols,
            optimization_hint,
        )

        self.assertEqual(free_symbols(actual), free_symbols(source))
        self.assertEqual(optimization_hint(actual), expected_hint)

    def _symbolic_batch_fake_mode(self, batch: int = 4):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            sym_batch = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(sym_batch, batch)
        return fake_mode, sym_batch

    def _chunk_batch(self, gm, **kwargs):
        return apply_chunk_pass(gm, mode="batch", **kwargs)

    def _chunk_seq(self, gm, **kwargs):
        return apply_chunk_pass(gm, mode="seq", **kwargs)

    def test_concretize_chunk_symbols_updates_nested_graph_modules(self):
        from torch.fx.experimental.symbolic_shapes import free_symbols

        def call_with_subgraph(subgraph, x):
            del subgraph
            return x

        fake_mode, sym_batch = self._symbolic_batch_fake_mode(4)
        batch_symbol = next(iter(free_symbols(sym_batch)))
        with fake_mode:
            x_val = torch.empty(sym_batch, 3)

        sub_graph = torch.fx.Graph()
        sub_x = sub_graph.placeholder("sub_x")
        sub_graph.output(sub_x)
        sub_gm = torch.fx.GraphModule(torch.nn.Module(), sub_graph)
        sub_x.meta["val"] = x_val

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        call = graph.call_function(call_with_subgraph, args=(sub_gm, x))
        graph.output(call)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        x.meta["val"] = x_val
        x.meta[CHUNK_SYMBOL_HINTS_META] = {batch_symbol: 4}
        call.meta["val"] = x_val

        concretize_ep_chunk_symbolic_shapes_pass(gm)

        self.assertEqual(free_symbols(x.meta["val"].shape[0]), set())
        self.assertEqual(free_symbols(sub_x.meta["val"].shape[0]), set())

    def test_concretize_chunk_symbols_updates_nested_scalar_example_values(self):
        from torch.fx.experimental.symbolic_shapes import free_symbols

        def call_with_subgraph(subgraph, length):
            del subgraph
            return length

        _, sym_batch = self._symbolic_batch_fake_mode(4)
        batch_symbol = next(iter(free_symbols(sym_batch)))

        sub_graph = torch.fx.Graph()
        sub_length = sub_graph.placeholder("sub_length")
        sub_graph.output(sub_length)
        sub_gm = torch.fx.GraphModule(torch.nn.Module(), sub_graph)
        sub_length.meta["example_value"] = sym_batch

        graph = torch.fx.Graph()
        length = graph.placeholder("length")
        call = graph.call_function(call_with_subgraph, args=(sub_gm, length))
        graph.output(call)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        length.meta["example_value"] = sym_batch
        length.meta[CHUNK_SYMBOL_HINTS_META] = {batch_symbol: 4}
        call.meta["example_value"] = sym_batch

        concretize_ep_chunk_symbolic_shapes_pass(gm)

        self.assertEqual(length.meta["example_value"], 4)
        self.assertEqual(sub_length.meta["example_value"], 4)

    def _build_linear_region_gm(
        self, *, input_shape=(4, 3), fqn="layers.0", mode: str = "batch"
    ):
        graph = torch.fx.Graph()
        w = graph.placeholder("w")
        x = graph.placeholder("x")
        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, w))
        relu = graph.call_function(torch.ops.aten.relu.default, args=(mm,))
        graph.output(relu)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        dim = {"batch": 0, "seq": 1}[mode]
        fake_mode, sym_extent = self._symbolic_batch_fake_mode(input_shape[dim])
        x_shape = list(input_shape)
        x_shape[dim] = sym_extent
        with fake_mode:
            w_val = torch.empty(input_shape[-1], input_shape[-1])
            x_val = torch.empty(*x_shape)
            out_val = torch.empty(*x_shape)

        w.meta["val"] = w_val
        x.meta["val"] = x_val
        for node in (mm, relu):
            node.meta["val"] = out_val
            node.meta["custom"] = {_MODULE_FQN: fqn}
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        return gm

    def _build_view_region_gm(
        self, *, shape_arg, fqn: str = "layers.0.moe"
    ) -> torch.fx.GraphModule:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        view = graph.call_function(torch.ops.aten.view.default, args=(x, shape_arg))
        graph.output(view)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode(4)
        with fake_mode:
            x_val = torch.empty(sym_batch, 3)
            out_val = torch.empty(sym_batch, 3)

        x.meta["val"] = x_val
        view.meta["val"] = out_val
        view.meta["custom"] = {_MODULE_FQN: fqn}
        view.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        view.stack_trace = "model.py:12 in forward\n    x = x.view(4, 3)"
        return gm

    def _nodes_by_target(self, gm, target):
        return [
            n for n in gm.graph.nodes if n.op == "call_function" and n.target is target
        ]

    def _assert_no_raw_selected_symbol_args(self, gm, symbol_hints):
        from torch.fx.experimental.symbolic_shapes import free_symbols
        from torch.utils._pytree import tree_leaves

        chunk_symbols = symbol_hints.keys()
        for node in gm.graph.nodes:
            if node.meta.get("chunked_region_role") is None:
                continue
            for value in tree_leaves((node.args, node.kwargs)):
                if isinstance(
                    value, (torch.SymInt, torch.SymFloat, torch.SymBool)
                ) and (free_symbols(value) & chunk_symbols):
                    self.fail(
                        "chunk-created executable args must materialize "
                        f"selected symbols as FX nodes: node={node.name}, "
                        f"value={value}"
                    )

    def test_static_view_shape_constant_errors_with_source_stack(self):
        gm = self._build_view_region_gm(shape_arg=[4, 3])

        with self.assertRaises(ValueError) as cm:
            self._chunk_batch(gm, module_patterns=["layers.*.moe"])
        message = str(cm.exception)
        self.assertIn("baked a Python constant", message)
        self.assertIn("model.py:12", message)
        self.assertIn("x.shape", message)
        self.assertIn("-1", message)

    def test_view_inferred_dim_is_valid_symbolic_shape_source(self):
        gm = self._build_view_region_gm(shape_arg=[-1, 3])

        self._chunk_batch(gm, module_patterns=["layers.*.moe"])

    def test_view_dim_list_is_not_treated_as_shape(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        squeeze = graph.call_function(torch.ops.aten.squeeze.dims, args=(x, [1]))
        graph.output(squeeze)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode(4)
        with fake_mode:
            x.meta["val"] = torch.empty(sym_batch, 1, 3)
            squeeze.meta["val"] = torch.empty(sym_batch, 3)
        squeeze.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
        squeeze.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE

        self._chunk_batch(gm, module_patterns=["layers.*.moe"])

    def _set_fake_tensor_meta(
        self,
        node,
        val,
        *,
        fqn: str | None = None,
        backward: bool = False,
    ):
        node.meta["val"] = val
        if fqn is not None:
            node.meta["custom"] = {_MODULE_FQN: fqn}
        if backward:
            node.meta["autograd_backward"] = True
        return node

    def _mark_chunk_body(
        self,
        node,
        *,
        fqn: str = "layers.0.moe",
        chunk_id: int,
        backward: bool = False,
        ep: str | None = None,
        token_exchange: bool = False,
        producer: str | None = None,
    ):
        custom = dict(node.meta.get("custom", {}))
        custom[_MODULE_FQN] = fqn
        if ep is not None:
            custom["EP"] = ep
            if token_exchange:
                custom[_EP_TOKEN_EXCHANGE] = ep
        node.meta["custom"] = custom
        node.meta["chunk_id"] = chunk_id
        node.meta["chunked_region_fqn"] = fqn
        node.meta["chunked_region_role"] = "body"
        if producer is not None:
            node.meta["chunked_region_producer"] = producer
        if backward:
            node.meta["autograd_backward"] = True
        return node

    def _build_backward_grad_chain_gm(
        self,
        *,
        cast: bool = False,
        collective: str | None = None,
        wait: bool = False,
        full_consumer: bool = False,
    ):
        graph = torch.fx.Graph()
        w = graph.placeholder("w")
        x = graph.placeholder("x")
        grad_out = graph.placeholder("grad_out")
        x_t = graph.call_function(torch.ops.aten.t.default, args=(x,))
        grad_w = graph.call_function(torch.ops.aten.mm.default, args=(x_t, grad_out))
        value = grad_w
        cast_node = None
        if cast:
            cast_node = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(value,),
                kwargs={"dtype": torch.float32},
            )
            value = cast_node

        c10d = torch.ops._c10d_functional
        collective_node = None
        if collective == "reduce_scatter":
            collective_node = graph.call_function(
                c10d.reduce_scatter_tensor.default,
                args=(value, "sum", 2, "dp"),
            )
            value = collective_node
        elif collective == "all_reduce":
            collective_node = graph.call_function(
                c10d.all_reduce.default,
                args=(value, "sum", "dp"),
            )
            value = collective_node
        elif collective is not None:
            raise AssertionError(f"unknown collective {collective}")

        wait_node = None
        if wait:
            if collective_node is None:
                raise AssertionError("wait=True requires a collective")
            wait_node = graph.call_function(c10d.wait_tensor.default, args=(value,))
            value = wait_node

        full_consumer_node = None
        if full_consumer:
            full_consumer_node = graph.call_function(
                torch.ops.aten.neg.default, args=(value,)
            )
            value = full_consumer_node

        graph.output((grad_out, value))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        dtype = torch.bfloat16 if cast else torch.float32
        with fake_mode:
            w_val = torch.empty(3, 3, dtype=dtype)
            act_val = torch.empty(sym_batch, 3, dtype=dtype)
            x_t_val = torch.empty(3, sym_batch, dtype=dtype)
            grad_val = torch.empty(3, 3, dtype=dtype)
            grad_fp32_val = torch.empty(3, 3, dtype=torch.float32)

        self._set_fake_tensor_meta(w, w_val)
        self._set_fake_tensor_meta(x, act_val)
        self._set_fake_tensor_meta(grad_out, act_val)
        self._set_fake_tensor_meta(x_t, x_t_val, fqn="layers.0", backward=True)
        self._set_fake_tensor_meta(grad_w, grad_val, fqn="layers.0", backward=True)
        if cast_node is not None:
            self._set_fake_tensor_meta(
                cast_node, grad_fp32_val, fqn="layers.0", backward=True
            )
        if collective_node is not None:
            self._set_fake_tensor_meta(
                collective_node,
                grad_fp32_val if cast else grad_val,
                fqn="layers.0",
                backward=True,
            )
        if wait_node is not None:
            self._set_fake_tensor_meta(
                wait_node,
                grad_fp32_val if cast else grad_val,
                fqn="layers.0",
                backward=True,
            )
        if full_consumer_node is not None:
            self._set_fake_tensor_meta(
                full_consumer_node,
                grad_fp32_val if cast else grad_val,
                fqn="layers.0",
                backward=True,
            )
        return gm, {
            "grad_w": grad_w,
            "cast": cast_node,
            "collective": collective_node,
            "wait": wait_node,
            "full_consumer": full_consumer_node,
        }

    def _build_dense_then_moe_gm(self, *, include_all_to_all: bool = True):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        dense = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        moe = graph.call_function(torch.ops.aten.neg.default, args=(dense,))
        if include_all_to_all:
            a2a = graph.call_function(
                torch.ops._c10d_functional.all_to_all_single.default,
                args=(moe, [], [], "ep"),
            )
            graph.output(a2a)
        else:
            a2a = None
            graph.output(moe)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            val = torch.empty(sym_batch, 3)

        x.meta["val"] = val
        node_fqns = [
            (dense, "layers.0"),
            (moe, "layers.1.moe"),
        ]
        if a2a is not None:
            node_fqns.append((a2a, "layers.1.moe"))
        for node, fqn in node_fqns:
            node.meta["val"] = val
            node.meta["custom"] = {_MODULE_FQN: fqn}
        moe.meta["custom"]["EP"] = "compute"
        if a2a is not None:
            a2a.meta["custom"]["EP"] = "dispatch"
        return gm

    def _build_previous_module_live_in_gm(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        prev = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        cur = graph.call_function(torch.ops.aten.neg.default, args=(prev,))
        graph.output(cur)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            val = torch.empty(sym_batch, 3)

        x.meta["val"] = val
        for node, fqn in ((prev, "layers.0"), (cur, "layers.1")):
            node.meta["val"] = val
            node.meta["custom"] = {_MODULE_FQN: fqn}
        return gm

    def _build_scalar_live_out_gm(self, *, valid: bool):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        size = graph.call_function(torch.ops.aten.sym_size.int, args=(relu, 0))
        if valid:
            scalar = size
        else:
            neg = graph.call_function(torch.ops.aten.neg.default, args=(x,))
            other_size = graph.call_function(torch.ops.aten.sym_size.int, args=(neg, 0))
            scalar = graph.call_function(operator.add, args=(size, other_size))
        graph.output((relu, scalar))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            val = torch.empty(sym_batch, 3)

        x.meta["val"] = val
        relu.meta["val"] = val
        relu.meta["custom"] = {_MODULE_FQN: "layers.0"}
        size.meta["val"] = sym_batch
        size.meta["custom"] = {_MODULE_FQN: "layers.0"}
        if not valid:
            neg.meta["val"] = val
            neg.meta["custom"] = {_MODULE_FQN: "layers.0"}
            other_size.meta["val"] = sym_batch
            other_size.meta["custom"] = {_MODULE_FQN: "layers.0"}
            scalar.meta["val"] = sym_batch + sym_batch
            scalar.meta["custom"] = {_MODULE_FQN: "layers.0"}
        return gm

    def _build_tuple_dead_getitem_gm(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        max_tuple = graph.call_function(torch.ops.aten.max.dim, args=(x, 1))
        values = graph.call_function(operator.getitem, args=(max_tuple, 0))
        graph.call_function(operator.getitem, args=(max_tuple, 1))
        neg = graph.call_function(torch.ops.aten.neg.default, args=(values,))
        graph.output(neg)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            val = torch.empty(sym_batch, 3)
            out_val = torch.empty(sym_batch)

        x.meta["val"] = val
        max_tuple.meta["custom"] = {_MODULE_FQN: "layers.0"}
        max_tuple.meta["autograd_backward"] = True
        values.meta["val"] = out_val
        for node in (values, neg):
            node.meta["custom"] = {_MODULE_FQN: "layers.0"}
            node.meta["autograd_backward"] = True
        neg.meta["val"] = out_val
        return gm

    def _build_opposite_direction_gm(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        backward_node = graph.call_function(torch.ops.aten.neg.default, args=(x,))
        forward_node = graph.call_function(
            torch.ops.aten.relu.default, args=(backward_node,)
        )
        graph.output(forward_node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            val = torch.empty(sym_batch, 3)

        x.meta["val"] = val
        backward_node.meta["val"] = val
        backward_node.meta["autograd_backward"] = True
        forward_node.meta["val"] = val
        forward_node.meta["custom"] = {_MODULE_FQN: "layers.0"}
        return gm

    def _build_forward_non_additive_no_dim_live_out_gm(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        amax = graph.call_function(torch.ops.aten.amax.default, args=(relu, [0], False))
        graph.output((relu, amax))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            val = torch.empty(sym_batch, 3)
            reduced_val = torch.empty(3)

        x.meta["val"] = val
        relu.meta["val"] = val
        relu.meta["custom"] = {_MODULE_FQN: "layers.0"}
        amax.meta["val"] = reduced_val
        amax.meta["custom"] = {_MODULE_FQN: "layers.0"}
        return gm

    def _build_buffer_mutation_gm(self):
        graph = torch.fx.Graph()
        buf = graph.placeholder("buf")
        x = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        count = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(x, [0]))
        add_ = graph.call_function(torch.ops.aten.add_.Tensor, args=(buf, count))
        graph.output(relu)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            val = torch.empty(sym_batch, 3)
            reduced_val = torch.empty(3)

        buf.meta["val"] = reduced_val
        x.meta["val"] = val
        relu.meta["val"] = val
        relu.meta["custom"] = {_MODULE_FQN: "layers.0"}
        for node in (count, add_):
            node.meta["val"] = reduced_val
            node.meta["custom"] = {_MODULE_FQN: "layers.0"}
        return gm

    def _build_backward_internal_no_dim_live_out_gm(self):
        graph = torch.fx.Graph()
        w = graph.placeholder("w")
        x = graph.placeholder("x")
        grad_out = graph.placeholder("grad_out")
        grad_act = graph.call_function(torch.ops.aten.mm.default, args=(grad_out, w))
        x_t = graph.call_function(torch.ops.aten.t.default, args=(x,))
        grad_w = graph.call_function(torch.ops.aten.mm.default, args=(x_t, grad_out))
        post = graph.call_function(torch.ops.aten.neg.default, args=(grad_w,))
        graph.output((grad_act, post))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            w_val = torch.empty(3, 3)
            val = torch.empty(sym_batch, 3)
            x_t_val = torch.empty(3, sym_batch)

        w.meta["val"] = w_val
        x.meta["val"] = val
        grad_out.meta["val"] = val
        grad_act.meta["val"] = val
        x_t.meta["val"] = x_t_val
        grad_w.meta["val"] = w_val
        post.meta["val"] = w_val
        for node in (grad_act, x_t, grad_w):
            node.meta["custom"] = {_MODULE_FQN: "layers.0"}
            node.meta["autograd_backward"] = True
        return gm

    def _build_indirect_per_chunk_live_out_gm(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        grad = graph.placeholder("grad")
        fwd = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        saved = graph.call_function(torch.ops.aten.amax.default, args=(fwd, [0], False))
        indirect_helper = graph.call_function(
            torch.ops.aten.unsqueeze.default, args=(saved, 0)
        )
        bwd = graph.call_function(
            torch.ops.aten.add.Tensor, args=(grad, indirect_helper)
        )
        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            val = torch.empty(sym_batch, 3)
            chunkless_val = torch.empty(3)
            helper_val = torch.empty(1, 3)

        for node in (x, grad):
            node.meta["val"] = val
        fwd.meta["val"] = val
        fwd.meta["custom"] = {_MODULE_FQN: "layers.0"}
        saved.meta["val"] = chunkless_val
        saved.meta["custom"] = {_MODULE_FQN: "layers.0"}
        for node, meta_val in ((indirect_helper, helper_val), (bwd, val)):
            node.meta["val"] = meta_val
            node.meta["custom"] = {_MODULE_FQN: "layers.1"}
            node.meta["autograd_backward"] = True
        return gm

    def test_chunk_batch_forward_region_semantics(self):
        gm = self._build_linear_region_gm()
        w = torch.randn(3, 3)
        x = torch.randn(4, 3)
        expected = gm(w, x)

        self._chunk_batch(
            gm,
            module_patterns=["layers.*"],
            num_static_inputs=1,
        )

        actual = gm(w, x)
        self.assertEqual(actual, expected)

        split_nodes = self._nodes_by_target(gm, torch.ops.aten.split_with_sizes.default)
        cat_nodes = self._nodes_by_target(gm, torch.ops.aten.cat.default)
        mm_nodes = self._nodes_by_target(gm, torch.ops.aten.mm.default)

        self.assertEqual(len(split_nodes), 1)
        self.assertEqual(split_nodes[0].args[0].name, "x")
        self.assertEqual(len(cat_nodes), 1)
        self.assertEqual(cat_nodes[0].args[1], 0)
        self.assertEqual(
            cat_nodes[0].meta.get("chunked_region_role"), "materialization"
        )
        self.assertEqual(
            len(self._nodes_by_target(gm, torch.ops.aten.clone.default)), 0
        )
        self.assertNotIn("chunk_id", cat_nodes[0].meta)
        self.assertEqual(len(mm_nodes), 2)
        self.assertEqual({n.meta.get("chunk_id") for n in mm_nodes}, {0, 1})
        self.assertEqual([n.meta.get("chunk_id") for n in mm_nodes], [0, 1])
        self.assertTrue(
            all(n.meta.get("chunked_region_role") == "body" for n in mm_nodes)
        )

    def test_chunk_batch_treats_previous_module_output_as_live_in(self):
        gm = self._build_previous_module_live_in_gm()
        inp = torch.randn(4, 3)
        expected = gm(inp)

        self._chunk_batch(gm, module_patterns=["layers.1"])

        actual = gm(inp)
        self.assertEqual(actual, expected)

        relu_nodes = self._nodes_by_target(gm, torch.ops.aten.relu.default)
        neg_nodes = self._nodes_by_target(gm, torch.ops.aten.neg.default)
        self.assertEqual(len(relu_nodes), 1)
        self.assertEqual(len(neg_nodes), 2)
        self.assertEqual({n.meta.get("chunk_id") for n in neg_nodes}, {0, 1})

    def test_chunk_batch_materializes_between_selected_roots(self):
        gm = self._build_previous_module_live_in_gm()

        self._chunk_batch(gm, module_patterns=["layers.*"])

        cat_nodes = self._nodes_by_target(gm, torch.ops.aten.cat.default)
        split_nodes = self._nodes_by_target(gm, torch.ops.aten.split_with_sizes.default)
        self.assertEqual(len(cat_nodes), 2)
        self.assertEqual(len(split_nodes), 2)
        self.assertIs(split_nodes[1].args[0], cat_nodes[0])

    def test_chunk_batch_materializes_scalar_live_out(self):
        gm = self._build_scalar_live_out_gm(valid=True)
        inp = torch.randn(4, 3)
        expected = gm(inp)

        self._chunk_batch(gm, module_patterns=["layers.*"])

        actual = gm(inp)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])

    def test_chunk_batch_rejects_invalid_region_boundaries(self):
        with self.assertRaisesRegex(ValueError, "Cannot split selected chunk"):
            self._chunk_batch(
                self._build_linear_region_gm(input_shape=(3, 3)),
                module_patterns=["layers.*"],
                num_static_inputs=1,
            )

        with self.assertRaisesRegex(ValueError, "opposite graph direction"):
            self._chunk_batch(
                self._build_opposite_direction_gm(),
                module_patterns=["layers.*"],
            )

        with self.assertRaisesRegex(NotImplementedError, "full-K/V"):
            self._chunk_seq(
                self._build_linear_region_gm(
                    input_shape=(2, 4, 3),
                    fqn="layers.0.attention",
                    mode="seq",
                ),
                module_patterns=["layers.*.attention"],
                num_static_inputs=1,
            )

    def test_chunk_seq_preserves_split_view_stride_scalar_live_in(self):
        fake_mode, sym_seq = self._symbolic_batch_fake_mode(4)
        with fake_mode:
            x_val = torch.empty(2, sym_seq, 256)

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        stride0 = graph.call_function(torch.ops.aten.sym_stride.int, args=(x, 0))
        view = graph.call_function(
            torch.ops.aten.as_strided.default,
            args=(x, [2, sym_seq, 256], [stride0, 256, 1]),
        )
        graph.output(view)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        x.meta["val"] = x_val
        stride0.meta["val"] = x_val.stride(0)
        view.meta["val"] = x_val
        view.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
        view.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE

        self._chunk_seq(gm, module_patterns=["layers.*.moe"])

    def test_chunk_seq_keeps_parent_shape_template_full(self):
        fake_mode, sym_seq = self._symbolic_batch_fake_mode(4)
        with fake_mode:
            x_val = torch.empty(8, sym_seq, 256)
            moe_val = torch.empty(8, sym_seq, 256)
            template_flat_val = torch.empty(8 * sym_seq, 256)
            template_val = torch.empty(8, sym_seq, 256)

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        seq = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 1))
        flat = graph.call_function(operator.mul, args=(8, seq))
        body_zeros = graph.call_function(
            torch.ops.aten.zeros.default,
            args=([flat, 256],),
            kwargs={
                "dtype": torch.bfloat16,
                "device": torch.device("cpu"),
                "pin_memory": False,
            },
        )
        body_view = graph.call_function(
            torch.ops.aten.view.default, args=(body_zeros, [8, seq, 256])
        )
        moe_out = graph.call_function(torch.ops.aten.add.Tensor, args=(x, body_view))
        template_flat = graph.call_function(
            torch.ops.aten.empty_strided.default,
            args=([flat, 256], [256, 1]),
            kwargs={
                "dtype": torch.bfloat16,
                "device": torch.device("cpu"),
                "pin_memory": False,
            },
        )
        template = graph.call_function(
            torch.ops.aten.view.default, args=(template_flat, [8, seq, 256])
        )
        stride0 = graph.call_function(torch.ops.aten.sym_stride.int, args=(template, 0))
        parent_empty = graph.call_function(
            torch.ops.aten.empty_strided.default,
            args=([8, seq, 256], [stride0, 256, 1]),
            kwargs={
                "dtype": torch.bfloat16,
                "device": torch.device("cpu"),
                "pin_memory": False,
            },
        )
        graph.output((moe_out, parent_empty))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        x.meta["val"] = x_val
        seq.meta["val"] = sym_seq
        flat.meta["val"] = 8 * sym_seq
        body_zeros.meta["val"] = template_flat_val
        body_view.meta["val"] = template_val
        moe_out.meta["val"] = moe_val
        template_flat.meta["val"] = template_flat_val
        template.meta["val"] = template_val
        stride0.meta["val"] = template_val.stride(0)
        parent_empty.meta["val"] = template_val
        for node in (body_zeros, body_view, moe_out, template_flat, template, stride0):
            node.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
        parent_empty.meta["custom"] = {_MODULE_FQN: "layers.0"}

        self._chunk_seq(gm, module_patterns=["layers.*.moe"])

        self.assertIs(stride0.args[0], template)
        self.assertIs(parent_empty.args[1][0], stride0)
        self.assertIs(flat.args[0], 8)
        self.assertIs(flat.args[1], seq)
        self.assertNotEqual(stride0.meta.get("chunked_region_role"), "body")
        self.assertNotEqual(template.meta.get("chunked_region_role"), "body")

    def test_chunk_seq_matches_eager_symbolic_flatten_view_contract(self):
        class FlatMoe(torch.nn.Module):
            def forward(self, x):
                batch, seq, dim = x.shape
                flat = x.view(-1, dim)
                return (flat + 1).view(batch, seq, dim)

        class ChunkModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].moe = FlatMoe()

            def forward(self, x):
                return self.layers[0].moe(x)

        def trace_model(strategy: str):
            model = ChunkModel()
            annotate_module_fqns(model)
            if strategy == "eager":
                maybe_apply_ep_overlap_eager_chunking(
                    model,
                    GraphTrainerCompileConfig(
                        enable=True,
                        ep_overlap=EpOverlapConfig(
                            enabled=True,
                            strategy="eager",
                            chunk_dim="seq",
                            module_fqn="layers.*.moe",
                        ),
                    ),
                )

            x = torch.randn(2, 4, 8)
            mark_chunk_dynamic_dims(x, mode="seq")
            traced = minimal_fx_tracer(lambda inp: model(inp), module=model)(x)
            if strategy == "eager":
                populate_eager_chunk_metadata_pass(traced.gm)
            else:
                populate_chunk_dim_metadata_pass(
                    traced.gm, traced.example_inputs, mode="seq"
                )
                self._chunk_seq(
                    traced.gm,
                    module_patterns=["layers.*.moe"],
                    num_static_inputs=traced.num_static_inputs,
                )
            return model, traced

        eager_model, eager = trace_model("eager")
        graph_model, graph = trace_model("graph")

        eager_clones = self._nodes_by_target(eager.gm, torch.ops.aten.clone.default)
        graph_clones = [
            node
            for node in self._nodes_by_target(graph.gm, torch.ops.aten.clone.default)
            if node.meta.get("chunked_region_role") == "chunk_input"
        ]
        self.assertEqual(len(eager_clones), 2)
        self.assertEqual(len(graph_clones), 2)
        self.assertEqual({node.meta.get("chunk_id") for node in graph_clones}, {0, 1})
        self.assertEqual(
            [tuple(node.meta["val"].shape) for node in graph_clones],
            [tuple(node.meta["val"].shape) for node in eager_clones],
        )
        self.assertEqual(
            [node.meta["val"].stride() for node in graph_clones],
            [node.meta["val"].stride() for node in eager_clones],
        )

        graph_body_targets = {
            node.meta.get("chunk_id"): [
                body.target
                for body in graph.gm.graph.nodes
                if body.meta.get("chunked_region_role") == "body"
                and body.meta.get("chunk_id") == node.meta.get("chunk_id")
            ]
            for node in graph_clones
        }
        for targets in graph_body_targets.values():
            self.assertIn(torch.ops.aten.view.default, targets)
            self.assertIn(torch.ops.aten.add.Tensor, targets)

        x_input = torch.randn(2, 4, 8)
        self.assertEqual(
            run_traced(graph, module=graph_model)(x_input),
            run_traced(eager, module=eager_model)(x_input),
        )

    def test_chunk_seq_rewrites_dependent_symbolic_scalar_live_in(self):
        fake_mode, sym_seq = self._symbolic_batch_fake_mode(4)
        with fake_mode:
            x_val = torch.empty(8, sym_seq, 256)
            out_val = torch.empty_strided(
                (8, sym_seq, 256),
                (256 * torch.sym_max(1, sym_seq), 256, 1),
            )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        stride0 = graph.call_function(torch.ops.aten.sym_stride.int, args=(x, 0))
        empty = graph.call_function(
            torch.ops.aten.empty_strided.default,
            args=([8, sym_seq, 256], [stride0, 256, 1]),
            kwargs={
                "dtype": torch.float32,
                "device": torch.device("cpu"),
                "pin_memory": False,
            },
        )
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, empty))
        graph.output(add)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        x.meta["val"] = x_val
        stride0.meta["val"] = x_val.stride(0)
        empty.meta["val"] = out_val
        add.meta["val"] = out_val
        for node in (empty, add):
            node.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE

        self._chunk_seq(gm, module_patterns=["layers.*.moe"])
        concretize_ep_chunk_symbolic_shapes_pass(gm)

        out = gm(torch.empty(8, 4, 256))
        self.assertEqual(out.shape, (8, 4, 256))
        self.assertEqual(out.stride(), (1024, 256, 1))

    def test_chunk_batch_rewrites_mixed_symbolic_scalar_live_in(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            batch = shape_env.create_unbacked_symint()
            width = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(batch, 4)
            torch._dynamo.override_optimization_hint(width, 5)
            x_val = torch.empty(batch, width)
            flat_val = torch.empty(batch * width)

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        batch_size = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 0))
        width_size = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 1))
        flat_size = graph.call_function(operator.mul, args=(batch_size, width_size))
        view = graph.call_function(torch.ops.aten.view.default, args=(x, [flat_size]))
        graph.output(view)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        x.meta["val"] = x_val
        batch_size.meta["val"] = batch
        width_size.meta["val"] = width
        flat_size.meta["val"] = batch * width
        view.meta["val"] = flat_val
        view.meta["custom"] = {_MODULE_FQN: "layers.0"}
        view.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE

        self._chunk_batch(gm, module_patterns=["layers.*"])
        self._assert_no_raw_selected_symbol_args(gm, {batch.node.expr: 4})

        chunk_views = [
            node
            for node in self._nodes_by_target(gm, torch.ops.aten.view.default)
            if node.meta.get("chunked_region_role") == "body"
        ]
        self.assertEqual(len(chunk_views), 2)
        self.assertTrue(
            all(isinstance(node.args[1][0], torch.fx.Node) for node in chunk_views)
        )
        self.assertEqual(
            {str(node.meta["val"].shape[0]) for node in chunk_views},
            {f"{width}*(({batch}//2))"},
        )
        self.assertEqual(gm(torch.randn(4, 5)).shape, (20,))

    def test_chunk_rewrite_materializes_nested_floor_div(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            seq = shape_env.create_unbacked_symint()
            width = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(seq, 8)
            torch._dynamo.override_optimization_hint(width, 5)
            x_val = torch.empty(seq, width)
            tp_local_flat = (seq // 2) * width

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        x.meta["val"] = x_val

        chunked_flat = _rewrite_chunk_symint(tp_local_flat, {seq.node.expr: 8})
        self.assertNotIn("floor(", str(chunked_flat))

        region = _Region(
            root_fqn="layers.0",
            synthetic_parent_fqn="layers.0",
            is_backward=False,
            nodes=(),
        )
        with gm.graph.inserting_after(x):
            materialized = _materialize_symint_arg(
                gm, chunked_flat, region=region, role="chunk_input"
            )

        self.assertIsInstance(materialized, torch.fx.Node)
        self.assertEqual(materialized.target, operator.mul)
        self.assertTrue(
            any(node.target is operator.floordiv for node in gm.graph.nodes)
        )

    def test_chunk_dim_classifies_derived_shapes_without_hint(self):
        from torch.fx.experimental.symbolic_shapes import free_symbols, ShapeEnv

        from torchtitan.experiments.graph_trainer.ep_chunk_pass import (
            _chunk_dim_from_symbols,
        )
        from torchtitan.experiments.graph_trainer.ep_pass_utils import (
            statically_equals_hint,
        )

        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            batch = shape_env.create_unbacked_symint()
            routed = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(batch, 4)
            torch._check(batch >= 2)
            val = torch.empty(batch + routed, 8)

        batch_symbol = next(iter(free_symbols(batch)))
        symbol_hints = {batch_symbol: 4}
        chunk_dim = _chunk_dim_from_symbols(val, symbol_hints)
        self.assertIsNotNone(chunk_dim)
        self.assertEqual(chunk_dim.dim, 0)
        self.assertIsNone(chunk_dim.hint)

        invariant = torch.sym_min(0, 2048 * batch)
        self.assertTrue(statically_equals_hint(invariant, 0))

    def test_chunk_batch_rejects_unsupported_live_outs(self):
        with self.assertRaisesRegex(ValueError, "cannot materialize scalar live-out"):
            self._chunk_batch(
                self._build_scalar_live_out_gm(valid=False),
                module_patterns=["layers.*"],
            )

        with self.assertRaisesRegex(
            ValueError,
            "forward accumulation proof or a backward parameter-gradient consumer",
        ):
            self._chunk_batch(
                self._build_forward_non_additive_no_dim_live_out_gm(),
                module_patterns=["layers.*"],
            )

        gm = self._build_indirect_per_chunk_live_out_gm()
        with self.assertRaisesRegex(ValueError, "without chunk dimension"):
            self._chunk_batch(gm, module_patterns=["layers.*"])

    def test_chunk_batch_ignores_dead_tuple_getitem_user(self):
        gm = self._build_tuple_dead_getitem_gm()
        self._chunk_batch(gm, module_patterns=["layers.*"])

        max_nodes = self._nodes_by_target(gm, torch.ops.aten.max.dim)
        self.assertEqual(len(max_nodes), 2)
        self.assertEqual({node.meta.get("chunk_id") for node in max_nodes}, {0, 1})

    def test_chunk_batch_backward_sums_internal_no_dim_live_out(self):
        gm = self._build_backward_internal_no_dim_live_out_gm()
        self._chunk_batch(gm, module_patterns=["layers.*"], num_static_inputs=1)
        sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
        self.assertEqual(len(sum_nodes), 1)
        self.assertTrue(
            all(
                node.meta.get("chunked_region_role") != "body"
                for node in self._nodes_by_target(gm, torch.ops.aten.neg.default)
            )
        )

    def test_chunk_batch_preserves_buffer_mutation_order(self):
        gm = self._build_buffer_mutation_gm()
        self._chunk_batch(gm, module_patterns=["layers.*"], num_static_inputs=1)
        add_mutations = self._nodes_by_target(gm, torch.ops.aten.add_.Tensor)
        self.assertEqual(len(add_mutations), 2)
        self.assertEqual({node.meta.get("chunk_id") for node in add_mutations}, {0, 1})
        by_chunk = {node.meta.get("chunk_id"): node for node in add_mutations}
        self.assertIs(by_chunk[1].args[0], by_chunk[0])

    def test_ep_overlap_chunk_accepts_moe_module_pattern(self):
        self._chunk_batch(
            self._build_linear_region_gm(fqn="layers.0.moe"),
            module_patterns=["layers.*.moe"],
            num_static_inputs=1,
        )

    def _compile_config_for_ep_overlap_test(self):
        from types import SimpleNamespace

        traced_result = SimpleNamespace(num_static_inputs=2, state_fqns=[])
        config = SimpleNamespace(
            model_spec=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
            parallelism=SimpleNamespace(
                enable_async_tensor_parallel=False,
                expert_parallel_degree=1,
                fsdp_reshard_after_forward="default",
                pipeline_parallel_degree=1,
            ),
            compile=GraphTrainerCompileConfig(
                enable=True,
                ep_overlap=EpOverlapConfig(
                    enabled=True,
                    chunk_dim="batch",
                    strategy="graph",
                    module_fqn="layers.*",
                    disable_early_grad_accumulation=False,
                ),
                cpu_offload_prefetch_n_layers=1,
                cpu_offload_defer_n_layers=1,
                cpu_offload_budget_gb=1.0,
                memory_policy="default",
                inductor_compilation="full",
                numerics_changing_optim=False,
                enable_fsdp_ag_rs_overlap=False,
            ),
        )
        return traced_result, config

    def test_ep_overlap_pass_pipeline_order(self):
        traced_result, config = self._compile_config_for_ep_overlap_test()

        def pass_name(pass_fn):
            return (
                pass_fn.func.__name__ if hasattr(pass_fn, "func") else pass_fn.__name__
            )

        names = [
            pass_name(pass_fn)
            for pass_fn in compile_time_passes(
                traced_result, config, use_cudagraph=False
            )
        ]
        dead_code_indices = [
            i for i, name in enumerate(names) if name == "eliminate_dead_code_pass"
        ]
        self.assertLess(
            names.index("canonicalize_graph_pass"),
            names.index("tag_with_memory_policy_pass"),
        )
        self.assertLess(
            names.index("apply_cpu_offload_pass"),
            names.index("selective_activation_remat_pass"),
        )
        self.assertLess(
            names.index("selective_activation_remat_pass"),
            names.index("populate_chunk_dim_metadata_pass"),
        )
        self.assertLess(
            names.index("populate_chunk_dim_metadata_pass"),
            names.index("ep_overlap_chunk_pass"),
        )
        self.assertLess(
            names.index("ep_overlap_chunk_pass"),
            dead_code_indices[-1],
        )
        self.assertLess(
            dead_code_indices[-1],
            names.index("joint_transformer_block_bucketing_reordering_pass"),
        )
        self.assertLess(
            names.index("joint_transformer_block_bucketing_reordering_pass"),
            names.index("concretize_ep_chunk_symbolic_shapes_pass"),
        )
        self.assertLess(
            names.index("concretize_ep_chunk_symbolic_shapes_pass"),
            names.index("full_inductor_compilation_pass"),
        )

    def test_graph_ep_seq_chunking_rejects_tensor_parallel(self):
        traced_result, config = self._compile_config_for_ep_overlap_test()
        config.compile.ep_overlap.chunk_dim = "seq"
        config.compile.ep_overlap.module_fqn = "layers.*.moe"
        config.parallelism.tensor_parallel_degree = 2

        with self.assertRaisesRegex(
            ValueError,
            "Graph EP seq chunking does not support tensor_parallel_degree > 1",
        ):
            compile_time_passes(traced_result, config, use_cudagraph=False)

    def test_graph_ep_batch_chunking_allows_tensor_parallel_guard(self):
        traced_result, config = self._compile_config_for_ep_overlap_test()
        config.compile.ep_overlap.chunk_dim = "batch"
        config.compile.ep_overlap.module_fqn = "layers.*"
        config.parallelism.tensor_parallel_degree = 2

        compile_time_passes(traced_result, config, use_cudagraph=False)

    def test_prepare_ep_overlap_trace_inputs_marks_batch_dims(self):
        _traced_result, config = self._compile_config_for_ep_overlap_test()
        x = torch.randn(4, 4)
        labels = torch.ones(4, 4)
        positions = torch.arange(4).repeat(4, 1)
        prepare_ep_overlap_trace_inputs(
            config.compile,
            (x, labels, torch.tensor(16), {}, {"positions": positions}),
            {},
        )
        self.assertIn(0, x._dynamo_unbacked_indices)
        self.assertIn(0, labels._dynamo_unbacked_indices)
        self.assertIn(0, positions._dynamo_unbacked_indices)
        self.assertEqual(x._dynamo_unbacked_bounds[0], (2, 4))

    def test_prepare_ep_overlap_trace_inputs_marks_seq_dims(self):
        _traced_result, config = self._compile_config_for_ep_overlap_test()
        config.compile.ep_overlap.chunk_dim = "seq"
        config.compile.ep_overlap.module_fqn = "layers.*.moe"
        x = torch.randn(4, 4)
        positions = torch.arange(4).repeat(4, 1)
        prepare_ep_overlap_trace_inputs(
            config.compile,
            (x, torch.ones(4, 4), torch.tensor(16), {}, {"positions": positions}),
            {},
        )
        self.assertIn(1, x._dynamo_unbacked_indices)
        self.assertIn(1, positions._dynamo_unbacked_indices)
        self.assertEqual(x._dynamo_unbacked_bounds[1], (2, 4))

    def test_prepare_ep_overlap_trace_inputs_bounds_seq_dim_to_original_half(self):
        _traced_result, config = self._compile_config_for_ep_overlap_test()
        config.compile.ep_overlap.chunk_dim = "seq"
        config.compile.ep_overlap.module_fqn = "layers.*.moe"
        x = torch.randn(2, 32, 4)
        labels = torch.ones(2, 32, dtype=torch.long)
        prepare_ep_overlap_trace_inputs(
            config.compile,
            (x, labels, torch.tensor(64), {}, {}),
            {},
        )

        self.assertEqual(x._dynamo_unbacked_bounds[1], (16, 32))
        self.assertEqual(labels._dynamo_unbacked_bounds[1], (16, 32))

    def test_seq_chunk_marker_traces_chunked_loss_backward(self):
        from torchtitan.components.loss import ChunkedCELoss

        torch.manual_seed(42)
        batch, seq_len, dim, vocab_size = 2, 32, 4, 8
        lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        loss_fn = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=8))
        loss_fn.lm_head = lm_head

        hidden_states = torch.randn(batch, seq_len, dim, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch, seq_len))
        mark_chunk_dynamic_dims(hidden_states, mode="seq")
        mark_chunk_dynamic_dims(labels, mode="seq")

        traced = minimal_fx_tracer(lambda h, y: loss_fn(h, y))(hidden_states, labels)

        self.assertGreater(len(list(traced.gm.graph.nodes)), 0)

    def test_prepare_ep_overlap_trace_inputs_rejects_empty_module_pattern(self):
        _traced_result, config = self._compile_config_for_ep_overlap_test()
        config.compile.ep_overlap.module_fqn = ""
        with self.assertRaisesRegex(ValueError, "ep_overlap.module_fqn"):
            prepare_ep_overlap_trace_inputs(config.compile, (torch.randn(4, 4),), {})

    def test_prepare_ep_overlap_trace_call_inputs_rebinds_block_mask_seq_lengths(self):
        from torch._dynamo.source import LocalSource
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            StatelessSymbolicContext,
        )
        from torch.nn.attention.flex_attention import create_block_mask

        _traced_result, config = self._compile_config_for_ep_overlap_test()
        config.compile.ep_overlap.chunk_dim = "seq"
        config.compile.ep_overlap.module_fqn = "layers.*.moe"

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
        positions = fake_mode.from_tensor(
            torch.arange(4).repeat(2, 1),
            source=LocalSource("positions", is_input=True),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.STATIC, DimDynamic.UNBACKED],
                shape_ids={1: "torchtitan_chunk_seq"},
            ),
        )
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=2,
            H=None,
            Q_LEN=4,
            KV_LEN=4,
            device="cpu",
        )

        prepared = prepare_ep_overlap_trace_call_inputs(
            config.compile,
            (
                torch.empty(2, 4),
                torch.empty(2, 4),
                torch.tensor(8),
                {"positions": positions, "attention_masks": block_mask},
            ),
            {},
        )

        self.assertIsNotNone(prepared)
        prepared_args, _ = prepared
        rebound_mask = prepared_args[3]["attention_masks"]
        seq_len = positions.shape[1]
        self.assertEqual(rebound_mask.seq_lengths[0].node.expr, seq_len.node.expr)
        self.assertEqual(rebound_mask.seq_lengths[1].node.expr, seq_len.node.expr)

    def test_eager_chunking_traces_overlap_metadata(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return torch.relu(self.linear(x))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x):
                return self.layers[0](x)

        model = Model()
        annotate_module_fqns(model)
        maybe_apply_ep_overlap_eager_chunking(
            model,
            GraphTrainerCompileConfig(
                enable=True,
                ep_overlap=EpOverlapConfig(
                    enabled=True,
                    strategy="eager",
                    module_fqn="layers.*",
                ),
            ),
        )

        def step(inputs):
            y = model(inputs)
            loss = y.sum()
            params = [p for p in model.parameters() if p.requires_grad]
            return [loss] + list(torch.autograd.grad(loss, params))

        traced = minimal_fx_tracer(step, module=model)(torch.randn(4, 3))
        gm = populate_eager_chunk_metadata_pass(traced.gm)

        body_chunks = {
            node.meta.get("chunk_id")
            for node in gm.graph.nodes
            if node.meta.get("chunked_region_role") == "body"
        }
        backward_body_chunks = {
            node.meta.get("chunk_id")
            for node in gm.graph.nodes
            if node.meta.get("chunked_region_role") == "body"
            and node.meta.get("autograd_backward", False)
        }
        boundary_roles = {
            node.meta.get("chunked_region_role")
            for node in gm.graph.nodes
            if node.meta.get("chunked_region_fqn") == "layers.0"
        }
        self.assertEqual(body_chunks, {0, 1})
        self.assertEqual(backward_body_chunks, {0, 1})
        self.assertIn("split_boundary", boundary_roles)
        self.assertIn("materialization", boundary_roles)

    def test_ep_overlap_chunk_skips_regions_without_ep_metadata(self):
        gm = self._build_dense_then_moe_gm()
        ep_overlap_chunk_pass(gm, mode="batch", module_pattern="layers.*")

        chunked_roots = {
            node.meta.get("chunked_region_fqn")
            for node in gm.graph.nodes
            if node.meta.get("chunked_region_role") == "body"
        }
        self.assertEqual(chunked_roots, {"layers.1"})

        with self.assertRaisesRegex(ValueError, "No EP all-to-all regions"):
            ep_overlap_chunk_pass(
                self._build_linear_region_gm(fqn="layers.0"),
                mode="batch",
                module_pattern="layers.*",
            )

    def test_ep_overlap_chunk_single_rank_fallback_uses_ep_annotations(self):
        gm = self._build_dense_then_moe_gm(include_all_to_all=False)

        ep_overlap_chunk_pass(
            gm,
            mode="batch",
            module_pattern="layers.*",
            require_all_to_all=False,
        )

        chunked_roots = {
            node.meta.get("chunked_region_fqn")
            for node in gm.graph.nodes
            if node.meta.get("chunked_region_role") == "body"
        }
        self.assertEqual(chunked_roots, {"layers.1"})

    def test_concretize_ep_chunk_symbolic_shapes_preserves_tensor_strides(self):
        from torch.fx.experimental.symbolic_shapes import free_symbols, ShapeEnv

        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            batch = shape_env.create_unbacked_symint()
            other = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(batch, 4)
            base_meta = torch.empty_strided(
                (batch, 4096, 16, 192),
                (4096 * 16 * 192, 16 * 192, 192, 1),
                device="cuda",
            )
            transpose_meta = base_meta.transpose(1, 2)
            mixed_meta = torch.empty(other + batch, device="cuda")

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        tuple_meta_node = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        sym_size = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 0))
        ge = graph.call_function(operator.ge, args=(sym_size, 0))
        graph.call_function(
            torch.ops.aten._assert_scalar.default,
            args=(ge, "chunk dim must be non-negative"),
        )
        sym_ite = graph.call_function(torch.sym_ite, args=(ge, 1, 0))
        transpose = graph.call_function(torch.ops.aten.transpose.int, args=(x, 1, 2))
        graph.output(transpose)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        x.meta["val"] = base_meta
        x.meta[CHUNK_SYMBOL_HINTS_META] = {next(iter(free_symbols(batch))): 4}
        tuple_meta_node.meta["val"] = (base_meta, transpose_meta, mixed_meta)
        sym_size.meta["val"] = batch
        ge.meta["val"] = batch >= 0
        sym_ite.meta["val"] = torch.sym_ite(batch >= 0, 1, 0)
        transpose.meta["val"] = transpose_meta

        fake_inputs = [base_meta]
        fake_inputs[0]._dynamo_unbacked_indices = {0}

        concretize_ep_chunk_symbolic_shapes_pass(gm, fake_inputs)

        self.assertEqual(tuple(transpose.meta["val"].shape), (4, 16, 4096, 192))
        self.assertEqual(
            transpose.meta["val"].stride(), (4096 * 16 * 192, 192, 16 * 192, 1)
        )
        self.assertEqual(
            tuple(tuple_meta_node.meta["val"][0].shape), (4, 4096, 16, 192)
        )
        self.assertEqual(
            tuple(tuple_meta_node.meta["val"][1].shape), (4, 16, 4096, 192)
        )
        mixed_symbols = free_symbols(tuple_meta_node.meta["val"][2].shape[0])
        self.assertIn(other.node.expr, mixed_symbols)
        self.assertNotIn(batch.node.expr, mixed_symbols)
        self.assertEqual(tuple(fake_inputs[0].shape), (4, 4096, 16, 192))
        self.assertFalse(hasattr(fake_inputs[0], "_dynamo_unbacked_indices"))
        self.assertNotIn(ge, gm.graph.nodes)
        self.assertNotIn(sym_ite, gm.graph.nodes)

    def test_concretize_ep_chunk_symbolic_shapes_rejects_false_guard(self):
        from torch.fx.experimental.symbolic_shapes import free_symbols, ShapeEnv

        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            batch = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(batch, 4)
            x_meta = torch.empty(batch, 16, device="cuda")

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        sym_size = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 0))
        eq = graph.call_function(operator.eq, args=(sym_size, 8))
        graph.call_function(
            torch.ops.aten._assert_scalar.default,
            args=(eq, "chunked split factor must match traced unroll count"),
        )
        graph.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        x.meta["val"] = x_meta
        x.meta[CHUNK_SYMBOL_HINTS_META] = {next(iter(free_symbols(batch))): 4}
        sym_size.meta["val"] = batch
        eq.meta["val"] = batch == 8

        with self.assertRaisesRegex(ValueError, "evaluates to False"):
            concretize_ep_chunk_symbolic_shapes_pass(gm, [x_meta])

    def test_concretize_ep_chunk_symbolic_shapes_does_not_replay_fake_metadata(self):
        from torch.fx.experimental.symbolic_shapes import free_symbols, ShapeEnv

        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            batch = shape_env.create_unbacked_symint()
            stale = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(batch, 4)
            x_meta = torch.empty(256 * batch, device="cuda")
            stale_slice_meta = torch.empty(stale, device="cuda")

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        size = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 0))
        sliced = graph.call_function(torch.ops.aten.slice.Tensor, args=(x, 0, 0, size))
        graph.output(sliced)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        x.meta["val"] = x_meta
        x.meta[CHUNK_SYMBOL_HINTS_META] = {next(iter(free_symbols(batch))): 4}
        size.meta["val"] = 256 * batch
        sliced.meta["val"] = stale_slice_meta

        concretize_ep_chunk_symbolic_shapes_pass(gm, [x_meta])

        self.assertEqual(sliced.args[3], 1024)
        self.assertEqual(tuple(sliced.meta["val"].shape), (stale,))

    def test_concretize_ep_chunk_symbolic_shapes_uses_placeholder_hints_for_codegen(
        self,
    ):
        from torch.fx.experimental.symbolic_shapes import free_symbols, ShapeEnv

        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            batch = shape_env.create_unbacked_symint()
            seq = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(batch, 4)
            torch._dynamo.override_optimization_hint(seq, 8)
            x_meta = torch.empty(batch, seq, device="cuda")

        batch_symbol = next(iter(free_symbols(batch)))
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        x.meta["val"] = x_meta
        x.meta[CHUNK_SYMBOL_HINTS_META] = {batch_symbol: 4}

        fake_inputs = [x_meta]
        concretize_ep_chunk_symbolic_shapes_pass(gm, fake_inputs)

        self.assertEqual(tuple(x.meta["val"].shape), (4, 8))
        self.assertEqual(tuple(fake_inputs[0].shape), (4, 8))
        self.assertFalse(free_symbols(x.meta["val"].shape[1]))
        self.assertNotIn(batch_symbol, free_symbols(x.meta["val"].shape[0]))

    def test_chunk_copied_meta_rewrites_chunk_symbol_inside_product(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        with fake_mode:
            batch = shape_env.create_unbacked_symint()
            width = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(batch, 4)
            torch._dynamo.override_optimization_hint(width, 8)
            val = torch.empty(batch * width, device="cuda")

        copied = _chunk_copied_meta({"val": val}, {batch.node.expr: 4})
        chunked_extent = copied["val"].shape[0]

        self.assertEqual(str(chunked_extent), f"{width}*(({batch}//2))")

    def test_chunk_batch_freshens_body_local_unbacked_bindings(self):
        from torch.fx.experimental.symbolic_shapes import free_symbols

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        shape_env = sym_batch.node.shape_env
        with fake_mode:
            x_val = torch.empty(sym_batch, 3)
            scalar_tensor_val = torch.empty(())
            split_val = shape_env.create_unbacked_symint()
            other_split_val = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(split_val, 2)
            torch._dynamo.override_optimization_hint(other_split_val, 2)
            total_val = split_val + other_split_val
            torch._check(split_val <= sym_batch, lambda: "split fits the token grid")
            torch._check(
                other_split_val <= sym_batch,
                lambda: "other split fits the token grid",
            )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        scalar_tensor = graph.call_function(torch.ops.aten.sum.default, args=(x,))
        split = graph.call_function(
            torch.ops.aten._local_scalar_dense.default, args=(scalar_tensor,)
        )
        other_scalar_tensor = graph.call_function(torch.ops.aten.sum.default, args=(x,))
        other_split = graph.call_function(
            torch.ops.aten._local_scalar_dense.default, args=(other_scalar_tensor,)
        )
        size = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 0))
        total = graph.call_function(operator.add, args=(split, other_split))
        eq = graph.call_function(operator.eq, args=(total, size))
        graph.call_function(
            torch.ops.aten._assert_scalar.default,
            args=(eq, "split sizes must cover the local token grid"),
        )
        view = graph.call_function(torch.ops.aten.view.default, args=(x, [total, -1]))
        graph.output(view)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        batch_symbol = next(iter(free_symbols(sym_batch)))
        split_symbol = split_val.node.expr
        other_split_symbol = other_split_val.node.expr
        x.meta["val"] = x_val
        x.meta[CHUNK_SYMBOL_HINTS_META] = {batch_symbol: 4}
        for node in (
            scalar_tensor,
            split,
            other_scalar_tensor,
            other_split,
            size,
            total,
            eq,
            view,
        ):
            node.meta["custom"] = {_MODULE_FQN: "layers.0"}
        scalar_tensor.meta["val"] = scalar_tensor_val
        other_scalar_tensor.meta["val"] = scalar_tensor_val
        split.meta["val"] = split_val
        split.meta["unbacked_bindings"] = {split_symbol: ()}
        other_split.meta["val"] = other_split_val
        other_split.meta["unbacked_bindings"] = {other_split_symbol: ()}
        size.meta["val"] = sym_batch
        total.meta["val"] = total_val
        eq.meta["val"] = total.meta["val"] == size.meta["val"]
        view.meta["val"] = x_val

        self._chunk_batch(gm, module_patterns=["layers.*"])

        split_nodes = [
            node
            for node in gm.graph.nodes
            if node.target is torch.ops.aten._local_scalar_dense.default
            and node.meta.get("chunked_region_role") == "body"
        ]
        self.assertEqual(len(split_nodes), 4)
        copied_symbols = {
            next(iter(node.meta["unbacked_bindings"])) for node in split_nodes
        }
        self.assertEqual(len(copied_symbols), 4)
        self.assertNotIn(split_symbol, copied_symbols)
        self.assertNotIn(other_split_symbol, copied_symbols)
        self.assertEqual(
            {shape_env.var_to_hint_override[symbol] for symbol in copied_symbols},
            {1},
        )

        view_nodes = [
            node
            for node in gm.graph.nodes
            if node.target is torch.ops.aten.view.default
            and node.meta.get("chunked_region_role") == "body"
        ]
        self.assertEqual(len(view_nodes), 2)
        for node in view_nodes:
            shape_arg_symbols = set()
            for arg in node.args[1]:
                value = arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg
                shape_arg_symbols.update(free_symbols(value))
            self.assertFalse(split_symbol in shape_arg_symbols)
            self.assertFalse(other_split_symbol in shape_arg_symbols)
            self.assertTrue(copied_symbols & shape_arg_symbols)
            self._assert_symbolic_dim_from_sources(
                node.meta["val"].shape[0], sym_batch, expected_hint=2
            )

    def test_chunk_batch_backward_cats_activation_grad_and_sums_param_grad(self):
        graph = torch.fx.Graph()
        w = graph.placeholder("w")
        x = graph.placeholder("x")
        grad_out = graph.placeholder("grad_out")
        grad_act = graph.call_function(torch.ops.aten.mm.default, args=(grad_out, w))
        x_t = graph.call_function(torch.ops.aten.t.default, args=(x,))
        grad_w = graph.call_function(torch.ops.aten.mm.default, args=(x_t, grad_out))
        graph.output((grad_act, grad_w))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            w_val = torch.empty(3, 3)
            act_val = torch.empty(sym_batch, 3)
            x_t_val = torch.empty(3, sym_batch)

        w.meta["val"] = w_val
        x.meta["val"] = act_val
        grad_out.meta["val"] = act_val
        grad_act.meta["val"] = act_val
        x_t.meta["val"] = x_t_val
        grad_w.meta["val"] = w_val
        for node in (grad_act, x_t, grad_w):
            node.meta["custom"] = {_MODULE_FQN: "layers.0"}
            node.meta["autograd_backward"] = True

        w_real = torch.randn(3, 3)
        x_real = torch.randn(4, 3)
        grad_out_real = torch.randn(4, 3)
        expected = gm(w_real, x_real, grad_out_real)

        self._chunk_batch(
            gm,
            module_patterns=["layers.*"],
            num_static_inputs=1,
        )

        actual = gm(w_real, x_real, grad_out_real)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])

        cat_nodes = self._nodes_by_target(gm, torch.ops.aten.cat.default)
        sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
        self.assertEqual(len(cat_nodes), 1)
        self.assertEqual(len(sum_nodes), 1)
        self.assertNotIn("chunk_id", cat_nodes[0].meta)
        self.assertNotIn("chunk_id", sum_nodes[0].meta)
        self.assertTrue(all(n.meta.get("autograd_backward") for n in sum_nodes))
        body_chunk_ids = [
            node.meta.get("chunk_id")
            for node in gm.graph.nodes
            if node.meta.get("chunked_region_role") == "body"
            and node.meta.get("chunked_region_fqn") == "layers.0"
        ]
        self.assertIn(1, body_chunk_ids)
        self.assertIn(0, body_chunk_ids)
        self.assertTrue(
            all(
                n.meta.get("autograd_backward")
                for n in gm.graph.nodes
                if n.meta.get("chunked_region_fqn") == "layers.0"
            )
        )

    def test_chunk_batch_backward_sums_grad_before_grad_collective(self):
        c10d = torch.ops._c10d_functional
        collective_specs = {
            "reduce_scatter": c10d.reduce_scatter_tensor.default,
            "all_reduce": c10d.all_reduce.default,
        }
        for name, target in collective_specs.items():
            with self.subTest(name=name):
                gm, refs = self._build_backward_grad_chain_gm(collective=name)

                self._chunk_batch(gm, module_patterns=["layers.*"], num_static_inputs=1)

                sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
                self.assertEqual(len(sum_nodes), 1)
                self.assertEqual(len(self._nodes_by_target(gm, target)), 1)
                self.assertIs(refs["collective"].args[0], sum_nodes[0])
                self.assertNotIn("chunk_id", sum_nodes[0].meta)

    def test_chunk_batch_backward_sums_after_dtype_change(self):
        gm, refs = self._build_backward_grad_chain_gm(cast=True)

        self._chunk_batch(gm, module_patterns=["layers.*"], num_static_inputs=1)

        sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
        self.assertEqual(len(sum_nodes), 1)
        output = next(node for node in gm.graph.nodes if node.op == "output")
        self.assertIs(output.args[0][1], sum_nodes[0])
        peer_cast = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops.aten._to_copy.default
            and node.meta.get("chunk_id") == 1
        )
        self.assertEqual(sum_nodes[0].args, (refs["cast"], peer_cast))
        self.assertEqual(sum_nodes[0].meta["val"].dtype, torch.float32)

    def test_chunk_batch_backward_replays_grad_collective_after_dtype_change(self):
        c10d = torch.ops._c10d_functional
        gm, _refs = self._build_backward_grad_chain_gm(
            cast=True, collective="reduce_scatter", wait=True
        )

        self._chunk_batch(gm, module_patterns=["layers.*"], num_static_inputs=1)

        sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
        rs_nodes = self._nodes_by_target(gm, c10d.reduce_scatter_tensor.default)
        self.assertEqual(len(sum_nodes), 1)
        self.assertEqual(len(rs_nodes), 1)
        self.assertIs(rs_nodes[0].args[0], sum_nodes[0])
        self.assertEqual(sum_nodes[0].meta["val"].dtype, torch.float32)

    def test_chunk_batch_backward_keeps_same_fqn_grad_plumbing_chunked(self):
        x_real = torch.randn(4, 3)
        grad_out_real = torch.randn(4, 3)
        w_real = torch.randn(3, 3)
        gm, refs = self._build_backward_grad_chain_gm(full_consumer=True)
        expected = gm(w_real, x_real, grad_out_real)

        self._chunk_batch(gm, module_patterns=["layers.*"], num_static_inputs=1)

        actual = gm(w_real, x_real, grad_out_real)
        self.assertEqual(actual, expected)
        sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
        self.assertEqual(len(sum_nodes), 1)
        self.assertNotEqual(
            refs["full_consumer"].meta.get("chunked_region_role"), "body"
        )
        self.assertIs(refs["full_consumer"].args[0], sum_nodes[0])
        self.assertNotIn("chunk_id", sum_nodes[0].meta)

    def test_chunk_batch_keeps_no_dim_control_path_to_ep_marker(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        counts = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(x, [0]))
        scale = graph.call_function(torch.ops.aten.sum.default, args=(counts,))
        y = graph.call_function(torch.ops.aten.mul.Tensor, args=(x, scale))
        a2a = graph.call_function(
            torch.ops._c10d_functional.all_to_all_single.default,
            args=(y, [], [], "ep"),
        )
        graph.output(a2a)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            x_val = torch.empty(sym_batch, 3)
            counts_val = torch.empty(3)
            scale_val = torch.empty(())

        x.meta["val"] = x_val
        counts.meta["val"] = counts_val
        scale.meta["val"] = scale_val
        y.meta["val"] = x_val
        a2a.meta["val"] = x_val
        for node in (counts, scale, y, a2a):
            node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "dispatch"}

        self._chunk_batch(gm, module_patterns=["layers.*.moe"])

        self.assertEqual(counts.meta.get("chunked_region_role"), "body")
        self.assertEqual(scale.meta.get("chunked_region_role"), "body")
        sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
        self.assertEqual(len(sum_nodes), 0)

    def test_chunk_batch_on_traced_toy_model(self):
        class ChunkBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return torch.relu(self.linear(x))

        class ChunkModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([ChunkBlock()])

            def forward(self, x):
                return checkpoint(self.layers[0], x, use_reentrant=False)

        model = ChunkModel()
        annotate_module_fqns(model)
        x = torch.randn(4, 4)
        mark_chunk_dynamic_dims(x, mode="batch")

        def step(inputs):
            y = model(inputs)
            loss = y.sum()
            params = [p for p in model.parameters() if p.requires_grad]
            return [loss] + list(torch.autograd.grad(loss, params))

        traced = minimal_fx_tracer(step, module=model)(x)
        expected = step(x)

        populate_chunk_dim_metadata_pass(
            traced.gm,
            traced.example_inputs,
            mode="batch",
        )
        self._chunk_batch(
            traced.gm,
            module_patterns=["layers.*"],
            num_static_inputs=traced.num_static_inputs,
        )

        actual = run_traced(traced, module=model)(x)
        self.assertEqual(len(actual), len(expected))
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor, expected_tensor)
        self.assertGreater(
            sum(1 for node in traced.gm.graph.nodes if node.meta.get("chunk_id") == 0),
            0,
        )

    def _trace_and_chunk_batch_module(self, block):
        class ChunkModel(torch.nn.Module):
            def __init__(self, block):
                super().__init__()
                self.layers = torch.nn.ModuleList([block])

            def forward(self, x):
                return self.layers[0](x)

        model = ChunkModel(block)
        annotate_module_fqns(model)
        x = torch.randn(4, 4)
        mark_chunk_dynamic_dims(x, mode="batch")
        traced = minimal_fx_tracer(lambda inp: model(inp), module=model)(x)

        populate_chunk_dim_metadata_pass(traced.gm, traced.example_inputs, mode="batch")
        self._chunk_batch(
            traced.gm,
            module_patterns=["layers.*"],
            num_static_inputs=traced.num_static_inputs,
        )
        return model, traced, x

    def test_chunk_batch_symbolic_factory_metadata(self):
        class NewZerosBlock(torch.nn.Module):
            def forward(self, x):
                return x + x.new_zeros((x.shape[0], x.shape[1]))

        class DerivedNewZerosBlock(torch.nn.Module):
            def forward(self, x):
                zeros = x.new_zeros((x.shape[0] + 2, x.shape[1]))
                return x + zeros[: x.shape[0]]

        class ZerosBlock(torch.nn.Module):
            def forward(self, x):
                zeros = torch.zeros(
                    (x.shape[0] * 2, x.shape[1]), device=x.device, dtype=x.dtype
                )
                return x + zeros[: x.shape[0]]

        class EmptyStridedBlock(torch.nn.Module):
            def forward(self, x):
                empty = torch.empty_strided(
                    (x.shape[0] * 2, x.shape[1]),
                    (x.shape[1], 1),
                    device=x.device,
                    dtype=x.dtype,
                )
                return x + empty[: x.shape[0]]

        cases = (
            ("new_zeros", NewZerosBlock(), torch.ops.aten.new_zeros.default, 2),
            (
                "derived_new_zeros",
                DerivedNewZerosBlock(),
                torch.ops.aten.new_zeros.default,
                4,
            ),
            ("zeros", ZerosBlock(), torch.ops.aten.zeros.default, 4),
            (
                "empty_strided",
                EmptyStridedBlock(),
                torch.ops.aten.empty_strided.default,
                4,
            ),
        )
        for name, block, target, expected_hint in cases:
            with self.subTest(name=name):
                _model, traced, _x = self._trace_and_chunk_batch_module(block)
                factory_nodes = [
                    node
                    for node in traced.gm.graph.nodes
                    if node.target is target
                    and node.meta.get("chunked_region_role") == "body"
                ]
                self.assertEqual(len(factory_nodes), 2)
                self.assertEqual(
                    {node.meta.get("chunk_id") for node in factory_nodes}, {0, 1}
                )
                batch_dim = traced.example_inputs[0].shape[0]
                for node in factory_nodes:
                    self._assert_symbolic_dim_from_sources(
                        node.meta["val"].shape[0],
                        batch_dim,
                        expected_hint=expected_hint,
                    )

    def test_chunk_batch_recomputes_owned_symbolic_index_tensors(self):
        class IndexBlock(torch.nn.Module):
            def forward(self, x):
                positions = torch.arange(x.shape[0], device=x.device)
                order = torch.flip(positions, (0,))
                return torch.index_select(x, 0, order)

        model, traced, x = self._trace_and_chunk_batch_module(IndexBlock())
        expected = torch.cat((model(x[:2]), model(x[2:])), dim=0)

        actual = run_traced(traced, module=model)(x)
        self.assertEqual(actual, expected)

        index_nodes = [
            node
            for node in traced.gm.graph.nodes
            if node.target is torch.ops.aten.index_select.default
            and node.meta.get("chunked_region_role") == "body"
        ]
        self.assertEqual(len(index_nodes), 2)
        self.assertEqual({node.meta.get("chunk_id") for node in index_nodes}, {0, 1})
        split_inputs = {
            node.args[0].target
            for node in self._nodes_by_target(
                traced.gm, torch.ops.aten.split_with_sizes.default
            )
            if isinstance(node.args[0], torch.fx.Node)
        }
        self.assertNotIn(torch.ops.aten.flip.default, split_inputs)


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


class TestRemoveB2BTransposePass(TestCase):
    """Unit tests for the remove_b2b_transpose_pass graph pass."""

    def _count_t_nodes(self, gm):
        """Count aten.t.default call_function nodes."""
        return sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.t.default
        )

    def test_b2b_transpose_pair_removed(self):
        """``t(t(x))`` collapses: both transpose nodes are removed and the
        consumer reads the original tensor directly."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        t1 = graph.call_function(torch.ops.aten.t.default, args=(x,))
        t2 = graph.call_function(torch.ops.aten.t.default, args=(t1,))
        relu = graph.call_function(torch.ops.aten.relu.default, args=(t2,))
        graph.output(relu)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        self.assertEqual(self._count_t_nodes(gm), 2)

        remove_b2b_transpose_pass(gm)

        self.assertEqual(self._count_t_nodes(gm), 0)
        # relu now consumes the placeholder directly.
        relu_node = next(
            n for n in gm.graph.nodes if n.target is torch.ops.aten.relu.default
        )
        self.assertEqual(relu_node.args[0].op, "placeholder")

        # Numerics preserved: relu(t(t(x))) == relu(x).
        x = torch.randn(3, 4)
        self.assertEqual(gm(x), torch.relu(x))

    def test_single_transpose_preserved(self):
        """A lone transpose is not a back-to-back pair and must be kept."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        t = graph.call_function(torch.ops.aten.t.default, args=(x,))
        graph.output(t)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        remove_b2b_transpose_pass(gm)

        self.assertEqual(self._count_t_nodes(gm), 1)
        x = torch.randn(3, 4)
        self.assertEqual(gm(x), x.t())

    def test_inner_transpose_with_other_user_kept(self):
        """When the inner transpose feeds another consumer, only the outer
        transpose is removed; the inner one stays for its other user."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        t1 = graph.call_function(torch.ops.aten.t.default, args=(x,))
        t2 = graph.call_function(torch.ops.aten.t.default, args=(t1,))
        # t1 also feeds a relu, so it cannot be erased.
        relu = graph.call_function(torch.ops.aten.relu.default, args=(t1,))
        graph.output((t2, relu))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        self.assertEqual(self._count_t_nodes(gm), 2)

        remove_b2b_transpose_pass(gm)

        # Outer transpose removed, inner one kept (still used by relu).
        self.assertEqual(self._count_t_nodes(gm), 1)

        x = torch.randn(3, 4)
        out_t2, out_relu = gm(x)
        self.assertEqual(out_t2, x)  # t(t(x)) == x
        self.assertEqual(out_relu, torch.relu(x.t()))

    def test_chain_of_transposes(self):
        """An odd-length chain ``t(t(t(x)))`` collapses to a single ``t(x)``."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        t1 = graph.call_function(torch.ops.aten.t.default, args=(x,))
        t2 = graph.call_function(torch.ops.aten.t.default, args=(t1,))
        t3 = graph.call_function(torch.ops.aten.t.default, args=(t2,))
        graph.output(t3)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        self.assertEqual(self._count_t_nodes(gm), 3)

        remove_b2b_transpose_pass(gm)

        self.assertEqual(self._count_t_nodes(gm), 1)
        x = torch.randn(3, 4)
        self.assertEqual(gm(x), x.t())

    def test_graph_without_transpose_unchanged(self):
        """Graphs without transpose nodes are returned unchanged."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        neg = graph.call_function(torch.ops.aten.neg.default, args=(relu,))
        graph.output(neg)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        num_nodes_before = len(list(gm.graph.nodes))

        result = remove_b2b_transpose_pass(gm)

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

    def _build_dynamic_slice_gm(
        self,
        *,
        dynamic_arg: str,
    ) -> torch.fx.GraphModule:
        """Build a slice graph where one of start/end/step is a Node."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        # sym_size is just a convenient way to produce a Node — its concrete
        # value is irrelevant, what matters is that the arg is an FX Node.
        sym_node = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 0))
        start = sym_node if dynamic_arg == "start" else 0
        end = sym_node if dynamic_arg == "end" else sys.maxsize
        step = sym_node if dynamic_arg == "step" else 1
        sliced = graph.call_function(
            torch.ops.aten.slice.Tensor, args=(x, 0, start, end, step)
        )
        graph.output(sliced)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_val = fake_mode.from_tensor(torch.empty(8, 16))
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = fake_val
        return gm

    def test_dynamic_end_skipped(self):
        """Slices whose ``end`` is an FX Node (dynamic shape at runtime) must
        be left alone — we can't prove identity at pass time."""
        gm = self._build_dynamic_slice_gm(dynamic_arg="end")
        # Must not raise and must not remove the slice (can't prove identity).
        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 1)

    def test_dynamic_start_skipped(self):
        """Slices whose ``start`` is an FX Node must be left alone."""
        gm = self._build_dynamic_slice_gm(dynamic_arg="start")
        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 1)

    def test_dynamic_step_skipped(self):
        """Slices whose ``step`` is an FX Node must be left alone."""
        gm = self._build_dynamic_slice_gm(dynamic_arg="step")
        remove_identity_slice_pass(gm)
        self.assertEqual(self._count_slice_nodes(gm), 1)


class TestAnnotateModuleFqns(TestCase):
    """Unit tests for annotate_module_fqns and insert_kernel_annotations_pass."""

    def _trace_and_get_fqns(self, model, *args):
        """Trace fwd+bwd via minimal_fx_tracer and return module_fqn annotations."""

        def fwd_step(*inputs):
            pred = model(inputs[0])
            loss = pred.sum()
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params)
            return [loss] + list(grads)

        traced = minimal_fx_tracer(fwd_step, module=model)(*args)
        fqns = set()
        for node in traced.gm.graph.nodes:
            fqn = (node.meta.get("custom") or {}).get(_MODULE_FQN)
            if fqn:
                fqns.add(fqn)
        return fqns

    def test_annotate_transformer_like_model(self):
        """Module FQNs survive minimal_fx_tracer for a transformer-like model
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


class TestNormalizeViewOpsAsReshape(TestCase):
    def test_replaces_view_and_unsafe_view(self):
        aten = torch.ops.aten
        g = torch.fx.Graph()
        x = g.placeholder("x")
        g.output(
            g.call_function(
                aten._unsafe_view.default,
                args=(
                    g.call_function(aten.view.default, args=(x, [4, 4])),
                    [2, 8],
                ),
            )
        )
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        normalize_view_ops_as_reshape(gm)
        for n in gm.graph.nodes:
            self.assertNotIn(n.target, {aten.view.default, aten._unsafe_view.default})


class TestCanonicalizeGraphPass(TestCase):
    """Unit tests for the combined canonicalize_graph_pass entry."""

    def test_runs_all_subpasses(self):
        """A single call drops detach + back-to-back transpose nodes and
        normalizes the surviving view op to reshape."""
        aten = torch.ops.aten
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        d = graph.call_function(aten.detach.default, args=(x,))
        t1 = graph.call_function(aten.t.default, args=(d,))
        t2 = graph.call_function(aten.t.default, args=(t1,))
        # Non-identity view (shape changes), left without fake meta so the
        # identity-view removal skips it and only normalization applies.
        v = graph.call_function(aten.view.default, args=(t2, [2, 8]))
        graph.output(v)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        canonicalize_graph_pass(gm)

        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertNotIn(aten.detach.default, targets)
        self.assertNotIn(aten.t.default, targets)
        self.assertNotIn(aten.view.default, targets)
        self.assertIn(aten.reshape.default, targets)

        # Numerics preserved: reshape(t(t(detach(x))), [2, 8]) == x.reshape(2, 8).
        x = torch.randn(4, 4)
        self.assertEqual(gm(x), x.reshape(2, 8))


class TestAsyncTensorParallelPass(FSDPTest):
    """Verify async_tensor_parallel_pass produces fused ops."""

    @property
    def world_size(self):
        return 2

    def test_ag_mm_becomes_fused_op(self):
        from torch.distributed._symmetric_memory import _test_mode

        from torchtitan.experiments.graph_trainer.passes import (
            async_tensor_parallel_pass,
        )

        pg = torch.distributed.distributed_c10d._get_default_group().group_name
        aten, c10d = torch.ops.aten, torch.ops._c10d_functional

        # shard[2048,4096] -> all_gather -> wait -> mm(w[4096,1024])
        g = torch.fx.Graph()
        s, w = g.placeholder("shard"), g.placeholder("weight")
        ag = g.call_function(c10d.all_gather_into_tensor.default, args=(s, 2, pg))
        wait = g.call_function(c10d.wait_tensor.default, args=(ag,))
        g.output(g.call_function(aten.mm.default, args=(wait, w)))

        # Shapes: shard, weight, ag, wait, mm
        shapes = [(2048, 4096), (4096, 1024), (4096, 4096), (4096, 4096), (4096, 1024)]
        with torch._subclasses.FakeTensorMode():
            for node, shape in zip(g.nodes, shapes):
                node.meta["val"] = torch.randn(shape)

        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        with _test_mode({pg}):
            async_tensor_parallel_pass(gm, ())

        fused = torch.ops.symm_mem.fused_all_gather_matmul.default
        self.assertTrue(any(n.target == fused for n in gm.graph.nodes))

    def test_mm_rs_becomes_fused_op(self):
        from torch.distributed._symmetric_memory import _test_mode

        from torchtitan.experiments.graph_trainer.passes import (
            async_tensor_parallel_pass,
        )

        pg = torch.distributed.distributed_c10d._get_default_group().group_name
        aten, c10d = torch.ops.aten, torch.ops._c10d_functional

        # mm(input[4096,4096], w[4096,1024]) -> reduce_scatter -> wait
        g = torch.fx.Graph()
        x, w = g.placeholder("x"), g.placeholder("w")
        mm = g.call_function(aten.mm.default, args=(x, w))
        rs = g.call_function(
            c10d.reduce_scatter_tensor.default,
            args=(mm, "sum", 2, pg),
        )
        g.output(g.call_function(c10d.wait_tensor.default, args=(rs,)))

        # Shapes: x, w, mm, rs, wait
        shapes = [
            (4096, 4096),
            (4096, 1024),
            (4096, 1024),
            (2048, 1024),
            (2048, 1024),
        ]
        with torch._subclasses.FakeTensorMode():
            for node, shape in zip(g.nodes, shapes):
                node.meta["val"] = torch.randn(shape)

        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        with _test_mode({pg}):
            async_tensor_parallel_pass(gm, ())

        fused = torch.ops.symm_mem.fused_matmul_reduce_scatter.default
        self.assertTrue(any(n.target == fused for n in gm.graph.nodes))


class TestSelectiveActivationRematPass(TestCase):
    """Unit tests for ``selective_activation_remat_pass``."""

    def test_topological_insertion_order(self):
        """
        When multiple independent ``must_recompute`` deps share a downstream
        consumer, duplicates must be inserted in graph (topological) order so
        each dup's args reference upstream dups rather than the originals.
        Without that ordering (e.g. naive DFS or unordered set iteration), a
        downstream dup created before its upstream dup would fall back to the
        original ``must_recompute`` node, defeating recompute.

            a = clone(inp1)        # must_recompute
            b = clone(inp2)        # must_recompute
            d = clone(inp3)        # must_recompute
            c = a + b              # must_recompute
            e = c + d              # must_recompute
            bwd = e + e            # autograd_backward
        """
        from torchtitan.experiments.graph_trainer.selective_activation_remat import (
            selective_activation_remat_pass,
        )

        graph = torch.fx.Graph()
        inp1 = graph.placeholder("inp1")
        inp2 = graph.placeholder("inp2")
        inp3 = graph.placeholder("inp3")
        a = graph.call_function(torch.ops.aten.clone.default, args=(inp1,))
        b = graph.call_function(torch.ops.aten.clone.default, args=(inp2,))
        d = graph.call_function(torch.ops.aten.clone.default, args=(inp3,))
        c = graph.call_function(torch.ops.aten.add.Tensor, args=(a, b))
        e = graph.call_function(torch.ops.aten.add.Tensor, args=(c, d))
        bwd = graph.call_function(torch.ops.aten.add.Tensor, args=(e, e))
        graph.output(bwd)
        for n in (a, b, c, d, e):
            n.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        original_names_in_order = [n.name for n in (a, b, d, c, e)]
        e_name = e.name

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        result = selective_activation_remat_pass(gm)

        nodes = list(result.graph.nodes)
        dups = [n for n in nodes if n.name.endswith("_recomputed")]
        # All 5 must_recompute nodes are transitive deps of bwd.
        self.assertEqual(len(dups), 5)

        # Dup graph order matches the forward order of the originals
        # (a, b, d, c, e).
        self.assertEqual(
            [n.name for n in dups],
            [name + "_recomputed" for name in original_names_in_order],
        )

        # The backward node's must_recompute input was redirected to the dup
        # of e; the original e (now dead) was erased. Use the Python ``bwd``
        # reference rather than searching by ``autograd_backward`` because
        # dups also carry that flag.
        for inp in bwd.all_input_nodes:
            self.assertEqual(inp.name, e_name + "_recomputed")
        self.assertNotIn(e_name, [n.name for n in nodes])

    def test_multiple_backward_regions_recompute_errors(self):
        graph = torch.fx.Graph()
        inp1 = graph.placeholder("inp1")
        inp2 = graph.placeholder("inp2")
        a = graph.call_function(torch.ops.aten.clone.default, args=(inp1,))
        b = graph.call_function(torch.ops.aten.clone.default, args=(inp2,))
        bwd1 = graph.call_function(torch.ops.aten.add.Tensor, args=(a, a))
        sep = graph.call_function(torch.ops.aten.neg.default, args=(inp1,))
        bwd2 = graph.call_function(torch.ops.aten.mul.Tensor, args=(b, b))
        graph.output((bwd1, sep, bwd2))
        for node in (a, b):
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        for node in (bwd1, bwd2):
            node.meta["autograd_backward"] = True

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        with self.assertRaisesRegex(RuntimeError, "disjoint backward regions"):
            selective_activation_remat_pass(gm)

    def test_forward_consumer_keeps_original(self):
        """When a must_recompute node has both forward and backward
        consumers, the original stays (forward needs it) and a dup is
        inserted for the backward consumer. The original is not erased.

            a = clone(inp)              # must_recompute, used by both fwd + bwd
            fwd_use = a + a             # forward consumer
            bwd = a * a                 # autograd_backward consumer
        """
        from torchtitan.experiments.graph_trainer.selective_activation_remat import (
            selective_activation_remat_pass,
        )

        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        a = graph.call_function(torch.ops.aten.clone.default, args=(inp,))
        fwd_use = graph.call_function(torch.ops.aten.add.Tensor, args=(a, a))
        bwd = graph.call_function(torch.ops.aten.mul.Tensor, args=(a, a))
        graph.output((fwd_use, bwd))
        a.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        a_name = a.name

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        result = selective_activation_remat_pass(gm)

        names = [n.name for n in result.graph.nodes]
        # Original kept (forward consumer still needs it) and dup inserted.
        self.assertIn(a_name, names)
        self.assertIn(a_name + "_recomputed", names)

        # bwd's args go to the dup; fwd_use still points to the original.
        bwd_node = next(
            n for n in result.graph.nodes if n.target is torch.ops.aten.mul.Tensor
        )
        for inp_node in bwd_node.all_input_nodes:
            self.assertEqual(inp_node.name, a_name + "_recomputed")
        fwd_use_node = next(
            n for n in result.graph.nodes if n.target is torch.ops.aten.add.Tensor
        )
        for inp_node in fwd_use_node.all_input_nodes:
            self.assertEqual(inp_node.name, a_name)

    def test_subgraph_get_attr_duplicated_for_recompute(self):
        """A recomputed node with a subgraph (GraphModule) get_attr input gets
        a PRIVATE copy of that get_attr, pointing at the same submodule.

        This mirrors how ``flex_attention`` references its score_mod / mask_mod
        subgraphs via get_attr nodes. Without private copies, the later
        regional_inductor pass — which partitions each flex node together with
        its subgraph get_attrs — would place a shared get_attr in only one
        region, leaving the other flex with the subgraph passed as a raw
        GraphModule arg (which fails to compile).

            sub = get_attr("subgraph")     # GraphModule attribute
            a   = some_op(inp, sub)        # must_recompute (HOP-like)
            bwd = a + a                    # autograd_backward consumer

        Plain-tensor get_attrs are left shared (they are never region-annotated),
        so this only fires for GraphModule-valued attributes.
        """
        from torchtitan.experiments.graph_trainer.selective_activation_remat import (
            selective_activation_remat_pass,
        )

        # A trivial GraphModule used as the subgraph attribute, plus a plain
        # tensor constant attribute that must stay shared (not duplicated).
        sub_graph = torch.fx.Graph()
        sub_graph.output(sub_graph.placeholder("x"))
        root = torch.nn.Module()
        root.subgraph = torch.fx.GraphModule(torch.nn.Module(), sub_graph)
        root.const = torch.nn.Buffer(torch.zeros(1))

        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        sub = graph.get_attr("subgraph")
        const = graph.get_attr("const")
        a = graph.call_function(torch.ops.aten.add.Tensor, args=(inp, sub))
        a.kwargs = {"const": const}
        bwd = graph.call_function(torch.ops.aten.add.Tensor, args=(a, a))
        graph.output(bwd)
        a.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        gm = torch.fx.GraphModule(root, graph)
        result = selective_activation_remat_pass(gm)

        dups = [n for n in result.graph.nodes if n.name.endswith("_recomputed")]
        self.assertEqual(len(dups), 1)
        dup = dups[0]

        dup_subgraph_attrs = [
            n
            for n in dup.all_input_nodes
            if n.op == "get_attr" and n.target == "subgraph"
        ]
        self.assertEqual(len(dup_subgraph_attrs), 1)
        # Private copy: a distinct node from the original, same submodule target.
        self.assertIsNot(dup_subgraph_attrs[0], sub)

        # The plain-tensor const get_attr is shared, not duplicated.
        const_attrs = [
            n for n in result.graph.nodes if n.op == "get_attr" and n.target == "const"
        ]
        self.assertEqual(len(const_attrs), 1)

    def test_offload_reload_chain_hoisted(self):
        """Mirrors the graph the CPU-offload pass produces: a forward
        offload chain (``ao.offload`` -> ``ao.wait_tensor``) and a backward
        reload chain (``ao.reload`` -> ``ao.wait_tensor``). When a
        recomputed node references the offloaded forward node F, the dup
        must read from the backward wait_tensor on GPU, not from F's
        freed-GPU storage. The remat pass discovers the offload chain
        through graph structure and hoists the backward reload chain in
        front of the dup's target.

            # Forward (autograd_backward=False)
            F           = clone(inp1)
            offload_op  = ao.offload(F)
            fwd_wait    = ao.wait_tensor(offload_op, F)
            N           = add(F, inp2)             # must_recompute

            # Backward (autograd_backward=True), placed after bwd_use so
            # the hoist actually has work to do:
            bwd_use     = mul(N, N)
            reload_op   = ao.reload(fwd_wait, "cuda")
            bwd_wait    = ao.wait_tensor(reload_op)
            bwd_other   = mul(bwd_wait, bwd_wait)
        """
        # Importing this module registers the ao::offload / ao::reload /
        # ao::wait_tensor ops with torch.ops.
        import torch._functorch._activation_offloading.offload_ops  # noqa: F401

        from torchtitan.experiments.graph_trainer.selective_activation_remat import (
            selective_activation_remat_pass,
        )

        graph = torch.fx.Graph()
        inp1 = graph.placeholder("inp1")
        inp2 = graph.placeholder("inp2")
        f = graph.call_function(torch.ops.aten.clone.default, args=(inp1,))
        offload_op = graph.call_function(torch.ops.ao.offload.default, args=(f,))
        fwd_wait = graph.call_function(
            torch.ops.ao.wait_tensor.default, args=(offload_op, f)
        )
        n = graph.call_function(torch.ops.aten.add.Tensor, args=(f, inp2))
        bwd_use = graph.call_function(torch.ops.aten.mul.Tensor, args=(n, n))
        reload_op = graph.call_function(
            torch.ops.ao.reload.default, args=(fwd_wait, "cuda")
        )
        bwd_wait = graph.call_function(
            torch.ops.ao.wait_tensor.default, args=(reload_op,)
        )
        bwd_other = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(bwd_wait, bwd_wait)
        )
        graph.output((bwd_use, bwd_other))

        n.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        bwd_use.meta["autograd_backward"] = True
        reload_op.meta["autograd_backward"] = True
        bwd_wait.meta["autograd_backward"] = True
        bwd_other.meta["autograd_backward"] = True

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        result = selective_activation_remat_pass(gm)

        nodes = list(result.graph.nodes)

        # Backward reload chain has been moved in front of the dup's target
        # (bwd_use) in topological order (reload_op before bwd_wait).
        reload_idx = nodes.index(reload_op)
        wait_idx = nodes.index(bwd_wait)
        bwd_use_idx = nodes.index(bwd_use)
        self.assertLess(reload_idx, wait_idx)
        self.assertLess(wait_idx, bwd_use_idx)

        # The forward offload chain stayed in forward (no hoist needed).
        offload_idx = nodes.index(offload_op)
        fwd_wait_idx = nodes.index(fwd_wait)
        self.assertLess(offload_idx, fwd_wait_idx)
        # Forward chain is also before the (hoisted) backward chain.
        self.assertLess(fwd_wait_idx, reload_idx)

        # The dup of N references bwd_wait (via the offload chain
        # redirect), not the original offloaded forward node F.
        dup = next(d for d in nodes if d.name.endswith("_recomputed"))
        self.assertIn(bwd_wait, dup.all_input_nodes)
        self.assertNotIn(f, dup.all_input_nodes)
        # The dup itself is positioned after the hoisted chain and before
        # its target.
        dup_idx = nodes.index(dup)
        self.assertLess(wait_idx, dup_idx)
        self.assertLess(dup_idx, bwd_use_idx)

        # bwd_use's args were redirected to the dup.
        for inp in bwd_use.all_input_nodes:
            self.assertIs(inp, dup)

        # bwd_other still consumes the (now hoisted) bwd_wait.
        for inp in bwd_other.all_input_nodes:
            self.assertIs(inp, bwd_wait)

    def test_offload_reload_chain_already_in_front_not_hoisted(self):
        """The CPU offload pass deliberately places ``ao.reload`` well before
        its ``ao.wait_tensor`` (via ``prefetch_reloads``) so the async H2D
        overlaps with backward compute. If the reload chain is already in
        front of the dup that needs it, ``ensure_offload_chain_before`` must
        leave it alone — re-hoisting collapses that prefetch gap and
        serializes the H2D against compute.

            # Forward (autograd_backward=False):
            F           = clone(inp1)
            offload_op  = ao.offload(F)
            fwd_wait    = ao.wait_tensor(offload_op, F)
            N           = add(F, inp2)              # must_recompute

            # Backward (autograd_backward=True), reload chain placed
            # EARLY — before the dup's target — exactly as
            # ``prefetch_reloads`` would arrange it:
            early_bwd   = mul(inp1, inp1)
            reload_op   = ao.reload(fwd_wait, "cuda")
            bwd_wait    = ao.wait_tensor(reload_op)
            middle_bwd  = mul(bwd_wait, bwd_wait)   # uses reload chain too
            bwd_use     = mul(N, N)                 # consumes N (dup target)
        """
        import torch._functorch._activation_offloading.offload_ops  # noqa: F401

        from torchtitan.experiments.graph_trainer.selective_activation_remat import (
            selective_activation_remat_pass,
        )

        graph = torch.fx.Graph()
        inp1 = graph.placeholder("inp1")
        inp2 = graph.placeholder("inp2")
        f = graph.call_function(torch.ops.aten.clone.default, args=(inp1,))
        offload_op = graph.call_function(torch.ops.ao.offload.default, args=(f,))
        fwd_wait = graph.call_function(
            torch.ops.ao.wait_tensor.default, args=(offload_op, f)
        )
        n = graph.call_function(torch.ops.aten.add.Tensor, args=(f, inp2))
        early_bwd = graph.call_function(torch.ops.aten.mul.Tensor, args=(inp1, inp1))
        reload_op = graph.call_function(
            torch.ops.ao.reload.default, args=(fwd_wait, "cuda")
        )
        bwd_wait = graph.call_function(
            torch.ops.ao.wait_tensor.default, args=(reload_op,)
        )
        middle_bwd = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(bwd_wait, bwd_wait)
        )
        bwd_use = graph.call_function(torch.ops.aten.mul.Tensor, args=(n, n))
        graph.output((middle_bwd, bwd_use))

        n.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        early_bwd.meta["autograd_backward"] = True
        reload_op.meta["autograd_backward"] = True
        bwd_wait.meta["autograd_backward"] = True
        middle_bwd.meta["autograd_backward"] = True
        bwd_use.meta["autograd_backward"] = True

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        result = selective_activation_remat_pass(gm)

        nodes = list(result.graph.nodes)
        early_idx = nodes.index(early_bwd)
        reload_idx = nodes.index(reload_op)
        wait_idx = nodes.index(bwd_wait)
        middle_idx = nodes.index(middle_bwd)
        bwd_use_idx = nodes.index(bwd_use)

        # The reload chain stayed at its original position (between early_bwd
        # and middle_bwd), preserving the prefetch gap. If the pass had
        # collapsed it next to bwd_use, reload_op/bwd_wait would land after
        # middle_bwd — which would also be a topology violation since
        # middle_bwd consumes bwd_wait.
        self.assertLess(early_idx, reload_idx)
        self.assertLess(reload_idx, wait_idx)
        self.assertLess(wait_idx, middle_idx)
        self.assertLess(middle_idx, bwd_use_idx)

        # The dup of N references bwd_wait (at its original position) and
        # is itself inserted right before bwd_use.
        dup = next(d for d in nodes if d.name.endswith("_recomputed"))
        self.assertIn(bwd_wait, dup.all_input_nodes)
        dup_idx = nodes.index(dup)
        self.assertLess(wait_idx, dup_idx)
        self.assertLess(dup_idx, bwd_use_idx)

        # middle_bwd still consumes bwd_wait at its original location.
        for inp in middle_bwd.all_input_nodes:
            self.assertIs(inp, bwd_wait)


class TestEliminateDeadCodePass(TestCase):
    """Unit tests for eliminate_dead_code_pass."""

    def test_removes_dead_pure_node_keeps_live(self):
        g = torch.fx.Graph()
        x = g.placeholder("x")
        live = g.call_function(torch.ops.aten.relu.default, (x,))
        g.call_function(torch.ops.aten.add.Tensor, (x, x))  # dead: no users
        g.output(live)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)

        eliminate_dead_code_pass(gm)
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertIn(torch.ops.aten.relu.default, targets)
        self.assertNotIn(torch.ops.aten.add.Tensor, targets)

    def test_removes_unreachable_bad_node(self):
        fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        with fake_mode:
            a_meta = torch.empty(256, 8192, device="cuda")
            b_meta = torch.empty(16384, 512, device="cuda")
            out_meta = torch.empty(1, device="cuda")

        graph = torch.fx.Graph()
        out = graph.placeholder("out")
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        graph.call_function(torch.ops.aten.mm.default, args=(a, b))
        graph.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        out.meta["val"] = out_meta
        a.meta["val"] = a_meta
        b.meta["val"] = b_meta

        eliminate_dead_code_pass(gm)
        self.assertNotIn(torch.ops.aten.mm.default, {n.target for n in gm.graph.nodes})

    def test_keeps_impure_node_with_no_users(self):
        # copy_ mutates its first arg (impure); DCE must keep it even though its
        # own result is unused.
        g = torch.fx.Graph()
        x = g.placeholder("x")
        y = g.placeholder("y")
        g.call_function(torch.ops.aten.copy_.default, (x, y))  # impure, unused result
        out = g.call_function(torch.ops.aten.relu.default, (x,))
        g.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)

        eliminate_dead_code_pass(gm)
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertIn(torch.ops.aten.copy_.default, targets)


class TestIsFullCudagraphable(TestCase):
    """Pure-CPU tests for the per-node cudagraph-safety predicate and the
    whole-graph gate built on it."""

    def test_clean_graph_is_full_cudagraphable(self):
        g = torch.fx.Graph()
        x = g.placeholder("x")
        relu = g.call_function(torch.ops.aten.relu.default, (x,))
        g.output(relu)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        self.assertTrue(is_cudagraphable(relu))
        self.assertTrue(is_full_cudagraphable(gm))

    def test_local_scalar_dense_is_unsafe(self):
        # _local_scalar_dense (.item()/.tolist()) extracts a host scalar a CUDA
        # graph replay can't reproduce -> unsafe, so the graph is not one piece.
        g = torch.fx.Graph()
        x = g.placeholder("x")
        s = g.call_function(torch.ops.aten._local_scalar_dense.default, (x,))
        g.output(s)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        self.assertFalse(is_cudagraphable(s))
        self.assertFalse(is_full_cudagraphable(gm))


class TestEagerChunking(TestCase):
    def _config(
        self,
        *,
        chunk_dim: str = "batch",
        module_fqn: str = "layers.*",
    ) -> GraphTrainerCompileConfig:
        return GraphTrainerCompileConfig(
            enable=True,
            ep_overlap=EpOverlapConfig(
                enabled=True,
                strategy="eager",
                chunk_dim=chunk_dim,
                module_fqn=module_fqn,
            ),
        )

    def test_eager_chunking_is_idempotent(self):
        class Block(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x):
                return self.layers[0](x)

        model = Model()
        config = self._config()
        maybe_apply_ep_overlap_eager_chunking(model, config)
        wrapped_forward = model.layers[0].forward
        maybe_apply_ep_overlap_eager_chunking(model, config)

        self.assertIs(model.layers[0].forward, wrapped_forward)

    def test_eager_chunking_compile_disabled_is_noop(self):
        class Block(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x):
                return self.layers[0](x)

        model = Model()
        forward = model.layers[0].forward
        config = self._config()
        config.enable = False

        maybe_apply_ep_overlap_eager_chunking(model, config)

        self.assertIs(model.layers[0].forward.__func__, forward.__func__)

    def test_eager_chunking_rejects_unsupported_output_type(self):
        class Block(torch.nn.Module):
            def forward(self, x):
                return {"x": x}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x):
                return self.layers[0](x)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(model, self._config())

        with self.assertRaisesRegex(TypeError, "layers.0.*dict"):
            model(torch.randn(4, 3))

    def test_transformer_batch_chunking_splits_positions_by_batch(self):
        seen_positions = []

        class Block(torch.nn.Module):
            def forward(self, x, attention_masks=None, positions=None):
                seen_positions.append(positions)
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x, positions):
                return self.layers[0](x, None, positions)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(model, self._config())
        x = torch.randn(4, 4, 2)
        positions = torch.arange(16).view(4, 4)

        self.assertEqual(model(x, positions), x)
        self.assertEqual([tuple(pos.shape) for pos in seen_positions], [(2, 4), (2, 4)])
        self.assertEqual(seen_positions[0], positions[:2])
        self.assertEqual(seen_positions[1], positions[2:])

    def test_transformer_batch_chunking_rejects_same_extent_tensor_mask(self):
        class Block(torch.nn.Module):
            def forward(self, x, attention_masks):
                return x + attention_masks

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x, attention_masks):
                return self.layers[0](x, attention_masks)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(model, self._config())

        with self.assertRaisesRegex(
            ValueError,
            "attention_masks must be None, BlockMask.*upstream .*TransformerBlock",
        ):
            model(torch.randn(4, 3), torch.randn(4, 3))

    def test_moe_chunking_splits_activation_only(self):
        seen_shapes = []

        class Moe(torch.nn.Module):
            def forward(self, x):
                seen_shapes.append(tuple(x.shape))
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].moe = Moe()

            def forward(self, x):
                return self.layers[0].moe(x)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(
            model,
            self._config(chunk_dim="seq", module_fqn="layers.*.moe"),
        )
        x = torch.randn(2, 4, 3)

        self.assertEqual(model(x), x)
        self.assertEqual(seen_shapes, [(2, 2, 3), (2, 2, 3)])

    def test_moe_chunking_rejects_extra_tensor_input(self):
        class Moe(torch.nn.Module):
            def forward(self, x, aux):
                return x + aux

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].moe = Moe()

            def forward(self, x, aux):
                return self.layers[0].moe(x, aux)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(
            model,
            self._config(module_fqn="layers.*.moe"),
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected exactly one positional activation tensor.*upstream MoE.forward",
        ):
            model(torch.randn(4, 3), torch.randn(4, 3))

    def test_eager_chunking_traces_overlap_metadata(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return torch.relu(self.linear(x))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x):
                return self.layers[0](x)

        model = Model()
        annotate_module_fqns(model)
        maybe_apply_ep_overlap_eager_chunking(model, self._config())

        traced = minimal_fx_tracer(lambda inputs: model(inputs), module=model)(
            torch.randn(4, 3)
        )
        gm = populate_eager_chunk_metadata_pass(traced.gm)

        body_nodes = [
            node
            for node in gm.graph.nodes
            if node.meta.get("chunked_region_role") == "body"
        ]
        roles = {
            node.meta.get("chunked_region_role")
            for node in gm.graph.nodes
            if node.meta.get("chunked_region_fqn") == "layers.0"
        }
        self.assertEqual({node.meta.get("chunk_id") for node in body_nodes}, {0, 1})
        self.assertEqual(
            {node.meta.get("chunked_region_fqn") for node in body_nodes},
            {"layers.0"},
        )
        self.assertIn("split_boundary", roles)
        self.assertIn("materialization", roles)

    def test_eager_chunking_splits_block_mask_batch_metadata(self):
        from torch.nn.attention.flex_attention import create_block_mask

        seen_masks = []

        def mask_mod(b, h, q_idx, kv_idx):
            return (b == 2) & (q_idx >= kv_idx)

        class Block(torch.nn.Module):
            def forward(self, x, attention_masks, positions):
                seen_masks.append(attention_masks)
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x, attention_masks, positions):
                return self.layers[0](x, attention_masks, positions)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(model, self._config())
        block_mask = create_block_mask(
            mask_mod,
            B=4,
            H=None,
            Q_LEN=128,
            KV_LEN=128,
            device="cpu",
        )
        x = torch.randn(4, 128, 8)
        positions = torch.arange(128).repeat(4, 1)

        self.assertEqual(model(x, block_mask, positions).shape, x.shape)
        self.assertEqual(len(seen_masks), 2)
        self.assertEqual([mask.kv_num_blocks.size(0) for mask in seen_masks], [2, 2])

        b = torch.tensor(0)
        h = torch.tensor(0)
        q_idx = torch.tensor(1)
        kv_idx = torch.tensor(0)
        self.assertFalse(seen_masks[0].mask_mod(b, h, q_idx, kv_idx).item())
        self.assertTrue(seen_masks[1].mask_mod(b, h, q_idx, kv_idx).item())


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
