# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import sys
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
    _MODULE_FQN,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.cudagraph import (
    insert_kernel_annotations_pass,
    is_cudagraphable,
    is_full_cudagraphable,
)
from torchtitan.experiments.graph_trainer.ep_process_group_pass import (
    isolate_ep_process_group_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    fsdp_reshard_after_forward_pass,
    reassign_collective_pgs_pass,
)
from torchtitan.experiments.graph_trainer.graph_utils import export_joint
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.memory_policy import (
    _make_default_memory_policy,
    _make_full_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.passes import selective_activation_remat_pass
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
from torchtitan.experiments.graph_trainer.subgraph_regions import (
    extract_common_fsdp_unshards_pass,
    SUBGRAPH_REGION,
    SUBGRAPH_REGION_ROLE,
)
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


class TestFSDPReshardAfterForwardPass(TestCase):
    def _make_graph(self, *, fsdp_all_gather=True, subgraph_region=False):
        aten = torch.ops.aten
        c10d = torch.ops._c10d_functional
        graph = torch.fx.Graph()
        shard = graph.placeholder("param_shard")
        indices = graph.placeholder("indices")
        if fsdp_all_gather:
            ag_input = shard
        else:
            other = graph.placeholder("other")
            ag_input = graph.call_function(aten.add.Tensor, args=(shard, other))
        ag = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(ag_input, 2, "pg")
        )
        if subgraph_region:
            ag.meta["custom"] = {SUBGRAPH_REGION: "loss_head_chunk"}
        wait = graph.call_function(c10d.wait_tensor.default, args=(ag,))
        view = graph.call_function(aten.view.default, args=(wait, [4, 4]))
        embedding = graph.call_function(aten.embedding.default, args=(view, indices))
        graph.output(embedding)
        return torch.fx.GraphModule(torch.nn.Module(), graph), {
            "ag": ag,
            "wait": wait,
            "view": view,
            "embedding": embedding,
        }

    def test_tags_fsdp_unshard_chain_and_mapped_consumer(self):
        gm, nodes = self._make_graph()
        fsdp_reshard_after_forward_pass(
            gm,
            recompute_consumer_arg_indices={torch.ops.aten.embedding.default: (0,)},
        )

        for name in ("ag", "wait", "view", "embedding"):
            self.assertEqual(
                nodes[name].meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
            )

    def test_does_not_tag_non_fsdp_all_gather(self):
        gm, nodes = self._make_graph(fsdp_all_gather=False)
        fsdp_reshard_after_forward_pass(
            gm,
            recompute_consumer_arg_indices={torch.ops.aten.embedding.default: (0,)},
        )

        for name in ("ag", "wait", "view", "embedding"):
            self.assertNotIn("recompute", nodes[name].meta)

    def test_skip_subgraph_regions(self):
        gm, nodes = self._make_graph(subgraph_region=True)
        fsdp_reshard_after_forward_pass(gm)
        self.assertNotIn("recompute", nodes["ag"].meta)

        gm, nodes = self._make_graph(subgraph_region=True)
        fsdp_reshard_after_forward_pass(gm, skip_subgraph_regions=False)
        self.assertEqual(
            nodes["ag"].meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_consumer_mapping_uses_configured_arg_index(self):
        aten = torch.ops.aten
        c10d = torch.ops._c10d_functional
        graph = torch.fx.Graph()
        shard = graph.placeholder("param_shard")
        other = graph.placeholder("other")
        ag = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(shard, 2, "pg")
        )
        wait = graph.call_function(c10d.wait_tensor.default, args=(ag,))
        consumer = graph.call_function(aten.add.Tensor, args=(other, wait))
        graph.output(consumer)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fsdp_reshard_after_forward_pass(
            gm, recompute_consumer_arg_indices={aten.add.Tensor: (0,)}
        )
        self.assertNotIn("recompute", consumer.meta)

        fsdp_reshard_after_forward_pass(
            gm, recompute_consumer_arg_indices={aten.add.Tensor: (1,)}
        )
        self.assertEqual(consumer.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)

    def test_preserves_must_save(self):
        gm, nodes = self._make_graph()
        nodes["ag"].meta["recompute"] = CheckpointPolicy.MUST_SAVE
        fsdp_reshard_after_forward_pass(gm)
        self.assertEqual(nodes["ag"].meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertNotIn("recompute", nodes["wait"].meta)

        gm, nodes = self._make_graph()
        nodes["wait"].meta["recompute"] = CheckpointPolicy.MUST_SAVE
        fsdp_reshard_after_forward_pass(gm)
        self.assertEqual(
            nodes["ag"].meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertEqual(nodes["wait"].meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertNotIn("recompute", nodes["view"].meta)

        gm, nodes = self._make_graph()
        nodes["embedding"].meta["recompute"] = CheckpointPolicy.MUST_SAVE
        fsdp_reshard_after_forward_pass(
            gm,
            recompute_consumer_arg_indices={torch.ops.aten.embedding.default: (0,)},
        )
        self.assertEqual(
            nodes["view"].meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertEqual(
            nodes["embedding"].meta["recompute"], CheckpointPolicy.MUST_SAVE
        )


class TestExtractCommonFSDPUnshardsPass(TestCase):
    def _make_graph(self, *, fsdp_all_gather=True):
        aten = torch.ops.aten
        c10d = torch.ops._c10d_functional
        graph = torch.fx.Graph()
        shard = graph.placeholder("param_shard")
        x = graph.placeholder("x")
        if fsdp_all_gather:
            ag_input = shard
        else:
            other = graph.placeholder("other")
            ag_input = graph.call_function(aten.add.Tensor, args=(shard, other))

        def mark(node, region):
            node.meta["custom"] = {
                SUBGRAPH_REGION: region,
                SUBGRAPH_REGION_ROLE: "loss_head",
            }
            return node

        outputs = []
        for region in ("chunk0", "chunk1"):
            ag = mark(
                graph.call_function(
                    c10d.all_gather_into_tensor.default, args=(ag_input, 2, "pg")
                ),
                region,
            )
            wait = mark(
                graph.call_function(c10d.wait_tensor.default, args=(ag,)), region
            )
            pad = mark(
                graph.call_function(aten.constant_pad_nd.default, args=(wait, [0, 0])),
                region,
            )
            cast = mark(
                graph.call_function(
                    torch.ops.prims.convert_element_type.default,
                    args=(pad, torch.bfloat16),
                ),
                region,
            )
            view = mark(
                graph.call_function(aten.view.default, args=(cast, [4, 4])), region
            )
            outputs.append(
                mark(graph.call_function(aten.add.Tensor, args=(view, x)), region)
            )

        graph.output(tuple(outputs))
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def _nodes_with_target(self, gm, target):
        return [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is target
        ]

    def test_extracts_common_unshard_chain_and_marks_must_save(self):
        gm = self._make_graph()
        extract_common_fsdp_unshards_pass(gm)

        for target in (
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            torch.ops._c10d_functional.wait_tensor.default,
            torch.ops.aten.constant_pad_nd.default,
            torch.ops.prims.convert_element_type.default,
            torch.ops.aten.view.default,
        ):
            nodes = self._nodes_with_target(gm, target)
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].meta["recompute"], CheckpointPolicy.MUST_SAVE)
            self.assertNotIn(SUBGRAPH_REGION, nodes[0].meta.get("custom", {}))
            self.assertNotIn(SUBGRAPH_REGION_ROLE, nodes[0].meta.get("custom", {}))

    def test_default_anchor_ignores_non_fsdp_all_gather(self):
        gm = self._make_graph(fsdp_all_gather=False)
        extract_common_fsdp_unshards_pass(gm)

        nodes = self._nodes_with_target(
            gm, torch.ops._c10d_functional.all_gather_into_tensor.default
        )
        self.assertEqual(len(nodes), 2)
        self.assertTrue(all("recompute" not in node.meta for node in nodes))

    def test_custom_anchor_is_honored(self):
        gm = self._make_graph(fsdp_all_gather=False)
        c10d = torch.ops._c10d_functional

        def is_any_all_gather(node):
            return node.target is c10d.all_gather_into_tensor.default

        extract_common_fsdp_unshards_pass(
            gm,
            is_unshard_anchor=is_any_all_gather,
        )

        nodes = self._nodes_with_target(
            gm, c10d.all_gather_into_tensor.default
        )
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].meta["recompute"], CheckpointPolicy.MUST_SAVE)


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

        # Create a second PG to simulate expert-FSDP
        second_pg = dist.new_group(
            ranks=list(range(self.world_size)),
            use_local_synchronization=True,
        )
        second_pg_name = second_pg.group_name

        # Rewrite half the AG nodes to use the second PG
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

        # Both source PGs should have their own extra PG
        self.assertIn(fsdp_pg_name, _EXTRA_FSDP_PG_REGISTRY)
        self.assertIn(second_pg_name, _EXTRA_FSDP_PG_REGISTRY)
        extra_pg1 = _EXTRA_FSDP_PG_REGISTRY[fsdp_pg_name]
        extra_pg2 = _EXTRA_FSDP_PG_REGISTRY[second_pg_name]
        self.assertNotEqual(
            extra_pg1, extra_pg2, "Each source PG must map to a distinct extra PG"
        )

        # No AG nodes should still use original PGs
        self.assertEqual(self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name), 0)
        self.assertEqual(self._count_ag_nodes_with_pg(bw_gm, second_pg_name), 0)

        # All AG nodes should use their respective extra PGs
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
        a2a.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "dispatch"}
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
        a2a.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "combine"}
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

        Calls minimal_fx_tracer without ``module=`` because parameterless
        models cannot produce gradients via autograd.grad.
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

        def fwd_only(x):
            return model(x)

        traced = minimal_fx_tracer(fwd_only)(torch.randn(4))
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


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
