# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import operator
import sys
from contextlib import contextmanager
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
from torch.fx.traceback import annotate_fn, preserve_node_meta
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase
from torch.utils.checkpoint import checkpoint, CheckpointPolicy

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    _EP_TOKEN_COUNT_EXCHANGE,
    _EP_TOKEN_COUNT_SYNC,
    _EP_TOKEN_EXCHANGE,
    _EP_TOKEN_EXCHANGE_WAIT,
    _MODULE_FQN,
    annotate_module_fqns,
    annotate_moe_ep_regions,
    get_default_transformer_block_buckets,
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
from torchtitan.experiments.graph_trainer.ep_overlap_pass import (
    _apply_schedule,
    _ready_nodes,
    _schedule_ep_overlap_regions,
    _ScheduledRegion,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    CHUNK_SYMBOL_HINTS_META,
    ChunkBody,
    ChunkedRegion,
    ChunkOwner,
    concretize_ep_chunk_symbolic_shapes_pass,
)
from torchtitan.experiments.graph_trainer.ep_process_group_pass import (
    isolate_ep_process_group_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    _FSDP_BUCKET_META,
    deduplicate_fsdp_unshard_chains_pass,
    get_transformer_block_bucket_counts,
    reassign_collective_pgs_pass,
    schedule_fsdp_comms_to_dense_regions_pass,
)
from torchtitan.experiments.graph_trainer.grad_chain_pass import (
    normalize_chunked_grad_collective_chains_pass,
)
from torchtitan.experiments.graph_trainer.graph_utils import export_joint
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _default_memory_policy_pass,
    _make_default_memory_policy,
    _make_full_memory_policy,
    _tag_minimal_async_ep_moe_full_recompute,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.minimal_async_ep_buffer_pass import (
    assign_minimal_async_ep_buffer_sets_pass,
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
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleList


class TestDefaultTransformerBlockBuckets(TestCase):
    def test_compile_time_passes_enable_chunked_loss_bucket_only_when_needed(self):
        from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
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
            compile_time_passes(traced_result, make_config(ChunkedLossWrapper.Config()))

        self.assertEqual(
            [
                call.kwargs["chunked_loss_enabled"]
                for call in mock_bucket_plan.call_args_list
            ],
            [False, True],
        )


class TestFSDPUnshardDedupPass(TestCase):
    def _duplicate_unshard_graph(self) -> torch.fx.GraphModule:
        graph = torch.fx.Graph()
        sharded_param = graph.placeholder("sharded_param")
        x = graph.placeholder("x")

        all_gather_0 = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(sharded_param, 1, "fsdp_pg"),
        )
        wait_0 = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(all_gather_0,),
        )
        unsharded_0 = graph.call_function(
            torch.ops.aten.view.default,
            args=(wait_0, [4]),
        )

        all_gather_1 = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(sharded_param, 1, "fsdp_pg"),
        )
        wait_1 = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(all_gather_1,),
        )
        unsharded_1 = graph.call_function(
            torch.ops.aten.view.default,
            args=(wait_1, [4]),
        )

        params = graph.call_function(
            torch.ops.aten.add.Tensor,
            args=(unsharded_0, unsharded_1),
        )
        out = graph.call_function(torch.ops.aten.add.Tensor, args=(params, x))
        graph.output(out)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        gm.graph.lint()
        gm.recompile()
        return gm

    def test_duplicate_fsdp_unshard_chains_are_canonicalized(self) -> None:
        gm = self._duplicate_unshard_graph()

        self.assertEqual(
            sum(1 for node in gm.graph.nodes if is_all_gather(node)),
            2,
        )

        deduplicate_fsdp_unshard_chains_pass(gm)

        self.assertEqual(
            sum(1 for node in gm.graph.nodes if is_all_gather(node)),
            1,
        )
        gm.graph.lint()


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

    def _count_rs_nodes_with_pg(self, gm, pg_name):
        return sum(
            1
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
            and node.args[3] == pg_name
        )

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


class TestFsdpDenseSchedulerPass(TestCase):
    """Pure FX tests for FSDP dense scheduling; no process group required."""

    def _tag_fsdp_schedule_node(self, node, fqn, *, backward=False):
        node.meta["custom"] = {_MODULE_FQN: fqn}
        if backward:
            node.meta["autograd_backward"] = True
        return node

    def _tag_fsdp_bucket(self, node, plan_fqns, direction):
        node.meta[_FSDP_BUCKET_META] = {
            "plan_fqns": tuple(plan_fqns),
            "direction": direction,
        }
        if direction == "bwd":
            node.meta["autograd_backward"] = True
        return node

    def _node_order(self, gm):
        return {node: i for i, node in enumerate(gm.graph.nodes)}

    def test_transformer_block_bucket_counts_follow_bucket_plan(self):
        counts = get_transformer_block_bucket_counts(
            [
                "tok_embeddings",
                "layers.0",
                [
                    "layers.1.attention_norm",
                    "layers.1.attention",
                    "layers.1.ffn_norm",
                    "layers.1.moe.router",
                    "layers.1.moe.shared_experts",
                ],
                "layers.1.moe.routed_experts.inner_experts",
                ["norm", "lm_head"],
            ],
            n_layers=2,
        )

        self.assertEqual(counts, {0: 1, 1: 2})

    def test_fsdp_dense_scheduler_accepts_expected_bucket_counts(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        fwd_ag0 = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, "pg")
        )
        fwd_ag0_wait = graph.call_function(c10d.wait_tensor.default, args=(fwd_ag0,))
        fwd0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(fwd_ag0_wait,)),
            "layers.0.attention",
        )
        bwd_ag0 = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, "pg")
        )
        bwd_ag0_wait = graph.call_function(c10d.wait_tensor.default, args=(bwd_ag0,))
        bwd0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(bwd_ag0_wait,)),
            "layers.0.attention",
            backward=True,
        )
        rs0 = self._tag_fsdp_schedule_node(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(bwd0, "sum", 0, "pg")
            ),
            "layers.0",
            backward=True,
        )
        rs0_wait = graph.call_function(c10d.wait_tensor.default, args=(rs0,))
        graph.output((fwd0, rs0_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm,
            moe_layer_ids=frozenset(),
            n_layers=1,
            transformer_bucket_counts_by_layer={0: 1},
            strict=True,
        )

        gm.graph.lint()

    def test_fsdp_dense_scheduler_validates_transformer_bucket_counts(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        ag0 = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, "pg")
        )
        ag0_wait = graph.call_function(c10d.wait_tensor.default, args=(ag0,))
        fwd0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(ag0_wait,)),
            "layers.0.attention",
        )
        bwd0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(fwd0,)),
            "layers.0.attention",
            backward=True,
        )
        graph.output(bwd0)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with self.assertRaisesRegex(ValueError, "layer 0"):
            schedule_fsdp_comms_to_dense_regions_pass(
                gm,
                moe_layer_ids=frozenset(),
                n_layers=1,
                transformer_bucket_counts_by_layer={0: 1},
            )

    def test_fsdp_dense_scheduler_ignores_non_transformer_buckets(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        ag = graph.call_function(c10d.all_gather_into_tensor.default, args=(x, 1, "pg"))
        ag_wait = self._tag_fsdp_schedule_node(
            graph.call_function(c10d.wait_tensor.default, args=(ag,)),
            "norm",
        )
        rs = self._tag_fsdp_schedule_node(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(ag_wait, "sum", 0, "pg")
            ),
            "lm_head",
            backward=True,
        )
        rs_wait = graph.call_function(c10d.wait_tensor.default, args=(rs,))
        graph.output(rs_wait)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with self.assertRaisesRegex(ValueError, "layer 0"):
            schedule_fsdp_comms_to_dense_regions_pass(
                gm,
                moe_layer_ids=frozenset(),
                n_layers=1,
                transformer_bucket_counts_by_layer={0: 1},
            )

    def test_fsdp_dense_scheduler_skips_transformer_edge_buckets(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.0.attention",
        )
        fwd_ag0 = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, "pg")
        )
        fwd_ag0_wait = graph.call_function(c10d.wait_tensor.default, args=(fwd_ag0,))
        fwd_ag0_use = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(fwd_ag0_wait,)),
            "layers.0.attention",
        )
        bwd_dense2 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(fwd_ag0_use,)),
            "layers.2.attention",
            backward=True,
        )
        bwd_ag2 = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, "pg")
        )
        bwd_ag2_wait = graph.call_function(c10d.wait_tensor.default, args=(bwd_ag2,))
        bwd_ag2_use = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(bwd_ag2_wait,)),
            "layers.2.attention",
            backward=True,
        )
        rs0 = self._tag_fsdp_schedule_node(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(bwd_ag2_use, "sum", 0, "pg")
            ),
            "layers.0",
            backward=True,
        )
        rs0_wait = graph.call_function(c10d.wait_tensor.default, args=(rs0,))
        graph.output((dense0, bwd_dense2, rs0_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=3, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[dense0], order[fwd_ag0])
        self.assertLess(order[bwd_dense2], order[bwd_ag2])
        self.assertLess(order[bwd_ag2_use], order[rs0])

    def test_fsdp_dense_scheduler_places_top_level_edge_buckets(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.0.attention",
        )
        top_ag = self._tag_fsdp_bucket(
            graph.call_function(c10d.all_gather_into_tensor.default, args=(x, 1, "pg")),
            ["norm", "lm_head"],
            "fwd",
        )
        top_ag_wait = graph.call_function(c10d.wait_tensor.default, args=(top_ag,))
        top_fwd = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(top_ag_wait,)),
            "norm",
        )
        top_bwd = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(top_fwd,)),
            "lm_head",
            backward=True,
        )
        bwd_ag0 = self._tag_fsdp_bucket(
            graph.call_function(c10d.all_gather_into_tensor.default, args=(x, 1, "pg")),
            ["layers.0.attention"],
            "bwd",
        )
        bwd_ag0_wait = graph.call_function(c10d.wait_tensor.default, args=(bwd_ag0,))
        bwd0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(bwd_ag0_wait,)),
            "layers.0.attention",
            backward=True,
        )
        top_rs = self._tag_fsdp_bucket(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(top_bwd, "sum", 0, "pg")
            ),
            ["norm", "lm_head"],
            "bwd",
        )
        top_rs_wait = graph.call_function(c10d.wait_tensor.default, args=(top_rs,))
        graph.output((dense0, bwd0, top_rs_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=1, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[top_ag], order[dense0])
        self.assertLess(order[dense0], order[top_ag_wait])
        self.assertLess(order[bwd_ag0], order[top_bwd])
        self.assertLess(order[top_bwd], order[bwd_ag0_wait])
        self.assertLess(order[top_bwd], order[top_rs])
        self.assertLess(order[top_rs], order[bwd0])
        self.assertGreater(order[top_rs_wait], order[bwd0])

    def test_fsdp_dense_scheduler_places_backward_ag_before_rs(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        ag0 = self._tag_fsdp_schedule_node(
            graph.call_function(c10d.all_gather_into_tensor.default, args=(x, 1, "pg")),
            "",
        )
        dense2 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.2.attention",
            backward=True,
        )
        dense1 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense2,)),
            "layers.1.attention",
            backward=True,
        )
        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense1,)),
            "layers.0.attention",
            backward=True,
        )
        ag0_wait = graph.call_function(c10d.wait_tensor.default, args=(ag0,))
        ag0_wait_user = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(ag0_wait,)),
            "layers.0",
            backward=True,
        )
        rs2 = self._tag_fsdp_schedule_node(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(dense2, "sum", 0, "pg")
            ),
            "layers.2",
            backward=True,
        )
        rs2_wait = graph.call_function(c10d.wait_tensor.default, args=(rs2,))
        graph.output((dense0, ag0_wait_user, rs2_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=3, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[dense2], order[ag0])
        self.assertLess(order[ag0], order[rs2])
        self.assertLess(order[rs2], order[dense1])
        self.assertLess(order[dense1], order[dense0])
        self.assertGreater(order[ag0_wait], order[dense0])
        self.assertGreater(order[rs2_wait], order[dense0])

    def test_fsdp_dense_scheduler_keeps_forward_ag_with_backward_descendants(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.0.attention",
        )
        ag1 = self._tag_fsdp_schedule_node(
            graph.call_function(c10d.all_gather_into_tensor.default, args=(x, 1, "pg")),
            "",
        )
        ag1_wait = graph.call_function(c10d.wait_tensor.default, args=(ag1,))
        fwd_use = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(ag1_wait,)),
            "layers.1.attention",
        )
        bwd2 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(fwd_use,)),
            "layers.2.attention",
            backward=True,
        )
        bwd1 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(bwd2,)),
            "layers.1.attention",
            backward=True,
        )
        graph.output((dense0, fwd_use, bwd1))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=3, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[ag1], order[dense0])
        self.assertLess(order[dense0], order[ag1_wait])

    def test_fsdp_dense_scheduler_uses_compute_not_fsdp_unpack_as_anchor(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        ag0 = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, "pg")
        )
        wait0 = self._tag_fsdp_schedule_node(
            graph.call_function(c10d.wait_tensor.default, args=(ag0,)),
            "layers.0.attention_norm",
        )
        unpack0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.view.default, args=(wait0, [1])),
            "layers.0.attention_norm",
        )
        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(unpack0,)),
            "layers.0.attention",
        )
        ag1 = graph.call_function(
            c10d.all_gather_into_tensor.default, args=(x, 1, "pg")
        )
        ag1_wait = graph.call_function(c10d.wait_tensor.default, args=(ag1,))
        dense1 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(ag1_wait,)),
            "layers.1.attention",
        )
        graph.output((dense0, dense1))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=2, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[wait0], order[ag1])
        self.assertLess(order[unpack0], order[ag1])
        self.assertLess(order[ag1], order[dense0])
        self.assertLess(order[dense0], order[ag1_wait])

    def test_fsdp_dense_scheduler_treats_recomputed_ag_use_as_backward(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        ag0 = self._tag_fsdp_schedule_node(
            graph.call_function(c10d.all_gather_into_tensor.default, args=(x, 1, "pg")),
            "",
        )
        dense2 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.2.attention",
            backward=True,
        )
        dense1 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense2,)),
            "layers.1.attention",
            backward=True,
        )
        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense1,)),
            "layers.0.attention",
            backward=True,
        )
        ag0_wait = graph.call_function(c10d.wait_tensor.default, args=(ag0,))
        view = graph.call_function(torch.ops.aten.view.default, args=(ag0_wait, [1]))
        recomputed = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(view,)),
            "layers.0.ffn_norm",
        )
        recomputed.name = "relu_recomputed"
        graph.output((dense0, recomputed))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=3, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[dense2], order[ag0])
        self.assertLess(order[ag0], order[dense1])
        self.assertLess(order[ag0_wait], order[recomputed])

    def test_fsdp_dense_scheduler_keeps_rs_after_gradient_producer(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        dense2 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.2.attention",
            backward=True,
        )
        dense1a = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense2,)),
            "layers.1.attention",
            backward=True,
        )
        grad_cast = graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(dense1a,),
            kwargs={"dtype": torch.float32},
        )
        dense1b = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(grad_cast,)),
            "layers.1.attention",
            backward=True,
        )
        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense1b,)),
            "layers.0.attention",
            backward=True,
        )
        rs2 = self._tag_fsdp_schedule_node(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(grad_cast, "sum", 0, "pg")
            ),
            "layers.2",
            backward=True,
        )
        rs2_wait = graph.call_function(c10d.wait_tensor.default, args=(rs2,))
        graph.output((dense0, rs2_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=3, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[dense1a], order[grad_cast])
        self.assertLess(order[grad_cast], order[rs2])
        self.assertLess(order[rs2], order[dense1b])

    def test_fsdp_dense_scheduler_sinks_output_only_rs_wait(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        dense2 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.2.attention",
            backward=True,
        )
        rs2 = self._tag_fsdp_schedule_node(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(dense2, "sum", 0, "pg")
            ),
            "layers.2",
            backward=True,
        )
        rs2_wait = graph.call_function(c10d.wait_tensor.default, args=(rs2,))
        dense1 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense2,)),
            "layers.1.attention",
            backward=True,
        )
        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense1,)),
            "layers.0.attention",
            backward=True,
        )
        graph.output((dense0, rs2_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=3, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[rs2], order[dense1])
        self.assertGreater(order[rs2_wait], order[dense0])

    def test_fsdp_dense_scheduler_sinks_unscheduled_rs_wait(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        rs0 = self._tag_fsdp_bucket(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(x, "sum", 0, "pg")
            ),
            ["layers.0"],
            "bwd",
        )
        rs0_wait = graph.call_function(c10d.wait_tensor.default, args=(rs0,))
        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.0.attention",
            backward=True,
        )
        graph.output((dense0, rs0_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=1, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[rs0], order[dense0])
        self.assertGreater(order[rs0_wait], order[dense0])

    def test_fsdp_dense_scheduler_sinks_rs_wait_output_unpack(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        dense2 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.2.attention",
            backward=True,
        )
        rs2 = self._tag_fsdp_schedule_node(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(dense2, "sum", 0, "pg")
            ),
            "layers.2",
            backward=True,
        )
        rs2_wait = graph.call_function(c10d.wait_tensor.default, args=(rs2,))
        split = graph.call_function(
            torch.ops.aten.split_with_sizes.default, args=(rs2_wait, [1])
        )
        shard = graph.call_function(operator.getitem, args=(split, 0))
        dense1 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense2,)),
            "layers.1.attention",
            backward=True,
        )
        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense1,)),
            "layers.0.attention",
            backward=True,
        )
        graph.output((dense0, shard))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=3, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[rs2], order[dense1])
        self.assertGreater(order[rs2_wait], order[dense0])
        self.assertGreater(order[split], order[dense0])
        self.assertGreater(order[shard], order[dense0])

    def test_fsdp_dense_scheduler_sinks_rs_wait_grad_accum_chain(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        dense2 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.2.attention",
            backward=True,
        )
        rs_a = self._tag_fsdp_bucket(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(dense2, "sum", 0, "pg")
            ),
            ["loss"],
            "bwd",
        )
        wait_a = graph.call_function(c10d.wait_tensor.default, args=(rs_a,))
        detached = graph.call_function(torch.ops.aten.detach.default, args=(wait_a,))
        rs_b = self._tag_fsdp_bucket(
            graph.call_function(
                c10d.reduce_scatter_tensor.default, args=(dense2, "sum", 0, "pg")
            ),
            ["loss"],
            "bwd",
        )
        wait_b = graph.call_function(c10d.wait_tensor.default, args=(rs_b,))
        accum = graph.call_function(torch.ops.aten.add_.Tensor, args=(detached, wait_b))
        grad_out = graph.call_function(torch.ops.aten.detach.default, args=(accum,))
        dense1 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense2,)),
            "layers.1.attention",
            backward=True,
        )
        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense1,)),
            "layers.0.attention",
            backward=True,
        )
        graph.output((dense0, grad_out))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset(), n_layers=3, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[dense0], order[wait_a])
        self.assertLess(order[dense0], order[wait_b])
        self.assertLess(order[wait_a], order[detached])
        self.assertLess(order[detached], order[accum])
        self.assertLess(order[wait_b], order[accum])
        self.assertLess(order[accum], order[grad_out])

    def test_fsdp_dense_scheduler_places_rs_after_moe_backward(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        dense2 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.2.attention",
            backward=True,
        )
        layer1_boundary = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(dense2,)),
            "layers.1",
            backward=True,
        )
        ffn_norm = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(layer1_boundary,)),
            "layers.1.ffn_norm",
            backward=True,
        )
        moe_dispatch = self._tag_fsdp_schedule_node(
            graph.call_function(
                c10d.all_to_all_single.default,
                args=(ffn_norm, [], [], "ep_pg"),
            ),
            "layers.1.moe.routed_experts.inner_experts",
            backward=True,
        )
        moe_dispatch_wait = self._tag_fsdp_schedule_node(
            graph.call_function(c10d.wait_tensor.default, args=(moe_dispatch,)),
            "layers.1.moe.routed_experts.inner_experts",
            backward=True,
        )
        dense1_attention = self._tag_fsdp_schedule_node(
            graph.call_function(
                torch.ops.aten.relu.default,
                args=(moe_dispatch_wait,),
            ),
            "layers.1.attention",
            backward=True,
        )
        rs2 = self._tag_fsdp_schedule_node(
            graph.call_function(
                c10d.reduce_scatter_tensor.default,
                args=(dense2, "sum", 0, "pg"),
            ),
            "layers.2",
            backward=True,
        )
        rs2_wait = graph.call_function(c10d.wait_tensor.default, args=(rs2,))
        graph.output((dense1_attention, rs2_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset({1}), n_layers=3, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[layer1_boundary], order[moe_dispatch])
        self.assertLess(order[moe_dispatch], order[moe_dispatch_wait])
        self.assertLess(order[moe_dispatch_wait], order[rs2])
        self.assertLess(order[rs2], order[dense1_attention])

    def test_fsdp_dense_scheduler_excludes_moe_nodes_from_dense_region(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        moe_node = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(x,)),
            "layers.0.moe.router",
        )
        dense0 = self._tag_fsdp_schedule_node(
            graph.call_function(torch.ops.aten.relu.default, args=(moe_node,)),
            "layers.0.attention",
        )
        ag1 = self._tag_fsdp_schedule_node(
            graph.call_function(c10d.all_gather_into_tensor.default, args=(x, 1, "pg")),
            "layers.1",
        )
        ag1_wait = graph.call_function(c10d.wait_tensor.default, args=(ag1,))
        graph.output((dense0, ag1_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        schedule_fsdp_comms_to_dense_regions_pass(
            gm, moe_layer_ids=frozenset({0}), n_layers=2, strict=True
        )

        order = self._node_order(gm)
        self.assertLess(order[moe_node], order[ag1])
        self.assertLess(order[ag1], order[dense0])


class TestOverlapPgIsolationPass(FSDPTest):
    def _setup(self):
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )

    def _get_fsdp_pg_name(self):
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        return fsdp_mesh.get_group().group_name

    def _count_all_ag_nodes(self, gm):
        return sum(1 for node in gm.graph.nodes if is_all_gather(node))

    def _count_ep_a2a_nodes_with_pg(self, gm, pg_name):
        return sum(
            1
            for node in gm.graph.nodes
            if node.op == "call_function"
            and "all_to_all_single" in str(node.target)
            and node.args[3] == pg_name
        )

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

    def test_minimal_async_ep_moe_forces_full_recompute_only_inside_moe(self):
        from torchtitan.models.common.token_dispatcher import (
            MinimalAsyncEPTokenDispatcher,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        moe = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        dense = graph.call_function(torch.ops.aten.neg.default, args=(moe,))
        graph.output(dense)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        moe.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
        dense.meta["custom"] = {_MODULE_FQN: "layers.0.attention"}
        for node in (moe, dense):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE

        config = SimpleNamespace(
            model_spec=SimpleNamespace(
                model=SimpleNamespace(
                    layers=[
                        SimpleNamespace(
                            moe=SimpleNamespace(
                                experts=SimpleNamespace(
                                    token_dispatcher=(
                                        MinimalAsyncEPTokenDispatcher.Config(
                                            num_experts=2,
                                            top_k=1,
                                        )
                                    )
                                )
                            )
                        )
                    ]
                )
            )
        )
        _tag_minimal_async_ep_moe_full_recompute(gm, config=config)

        self.assertEqual(moe.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertEqual(dense.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_default_policy_saves_fsdp_unshard_when_not_resharding(self):
        """Saves the helper-selected FSDP unshard output only when needed."""
        cases = (
            ("never", 1, CheckpointPolicy.MUST_SAVE),
            # Under PP, the default FSDP policy keeps params unsharded across
            # forward/backward, so SAC must save the same unshard boundary.
            ("default", 2, CheckpointPolicy.MUST_SAVE),
            ("always", 1, CheckpointPolicy.PREFER_RECOMPUTE),
        )

        for reshard_after_forward, pp_degree, expected_wait_policy in cases:
            with self.subTest(
                reshard_after_forward=reshard_after_forward,
                pp_degree=pp_degree,
            ):
                (
                    gm,
                    all_gather,
                    wait,
                    view,
                ) = self._fsdp_unshard_test_graph()
                config = SimpleNamespace(
                    parallelism=SimpleNamespace(
                        fsdp_reshard_after_forward=reshard_after_forward,
                        pipeline_parallel_degree=pp_degree,
                    )
                )

                _default_memory_policy_pass(gm, config=config)

                self.assertEqual(
                    all_gather.meta["recompute"],
                    CheckpointPolicy.PREFER_RECOMPUTE,
                )
                self.assertEqual(wait.meta["recompute"], expected_wait_policy)
                self.assertEqual(
                    view.meta["recompute"],
                    CheckpointPolicy.PREFER_RECOMPUTE,
                )

    def _fsdp_unshard_test_graph(self):
        graph = torch.fx.Graph()
        param = graph.placeholder("param")
        x = graph.placeholder("x")
        all_gather = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(param, 1, "0"),
        )
        wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(all_gather,),
        )
        view = graph.call_function(torch.ops.aten.view.default, args=(wait, [4]))
        out = graph.call_function(torch.ops.aten.add.Tensor, args=(view, x))
        graph.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        return gm, all_gather, wait, view

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

    def _node_depends_on(self, node, ancestor):
        seen = set()
        work = [node]
        while work:
            current = work.pop()
            if current is ancestor:
                return True
            if current in seen:
                continue
            seen.add(current)
            work.extend(current.all_input_nodes)
        return False

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

    def _build_chunked_grad_collective_with_output_cast(self, *, producer: str):
        graph = torch.fx.Graph()
        loss = graph.placeholder("loss")
        outputs = []
        for chunk_id in (0, 1):
            grad = graph.placeholder(f"grad_{chunk_id}")
            self._set_fake_tensor_meta(grad, torch.empty(4, 4, dtype=torch.float32))
            self._mark_chunk_body(
                grad,
                chunk_id=chunk_id,
                backward=True,
                producer=producer,
            )
            collective = graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                args=(grad, "sum", 2, "dp"),
            )
            self._set_fake_tensor_meta(
                collective, torch.empty(2, 4, dtype=torch.float32), backward=True
            )
            wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default, args=(collective,)
            )
            self._set_fake_tensor_meta(
                wait, torch.empty(2, 4, dtype=torch.float32), backward=True
            )
            cast = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(wait,),
                kwargs={"dtype": torch.bfloat16},
            )
            outputs.append(
                self._set_fake_tensor_meta(
                    cast, torch.empty(2, 4, dtype=torch.bfloat16), backward=True
                )
            )
        grad_output = graph.call_function(
            torch.ops.aten.add.Tensor, args=tuple(outputs)
        )
        self._set_fake_tensor_meta(
            grad_output, torch.empty(2, 4, dtype=torch.bfloat16), backward=True
        )
        graph.output((loss, grad_output))
        return torch.fx.GraphModule(torch.nn.Module(), graph)

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

    def _build_ep_overlap_schedule_gm(self, *, backward: bool = False):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        outputs = []
        for chunk_id in (1, 0) if backward else (0, 1):
            pre = graph.call_function(torch.ops.aten.relu.default, args=(x,))
            first_launch = graph.call_function(
                c10d.all_to_all_single.default,
                args=(pre, [], [], "ep"),
            )
            first_wait = graph.call_function(
                c10d.wait_tensor.default, args=(first_launch,)
            )
            compute = graph.call_function(
                torch.ops.aten.neg.default, args=(first_wait,)
            )
            second_launch = graph.call_function(
                c10d.all_to_all_single.default,
                args=(compute, [], [], "ep"),
            )
            second_wait = graph.call_function(
                c10d.wait_tensor.default, args=(second_launch,)
            )
            tail = graph.call_function(torch.ops.aten.neg.default, args=(second_wait,))
            outputs.append(tail)

            first_ep = "combine" if backward else "dispatch"
            second_ep = "dispatch" if backward else "combine"
            self._mark_chunk_body(
                pre, chunk_id=chunk_id, backward=backward, ep=first_ep
            )
            self._mark_chunk_body(
                first_launch,
                chunk_id=chunk_id,
                backward=backward,
                ep=first_ep,
                token_exchange=True,
            )
            self._mark_chunk_body(
                first_wait, chunk_id=chunk_id, backward=backward, ep=first_ep
            )
            self._mark_chunk_body(compute, chunk_id=chunk_id, backward=backward)
            self._mark_chunk_body(
                second_launch,
                chunk_id=chunk_id,
                backward=backward,
                ep=second_ep,
                token_exchange=True,
            )
            self._mark_chunk_body(
                second_wait, chunk_id=chunk_id, backward=backward, ep=second_ep
            )
            self._mark_chunk_body(tail, chunk_id=chunk_id, backward=backward)

        graph.output(tuple(outputs))
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    @contextmanager
    def _fake_dp_mesh(self):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.testing._internal.distributed.fake_pg import FakeStore

        already_initialized = dist.is_initialized()
        if not already_initialized:
            dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
        try:
            yield init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        finally:
            if not already_initialized and dist.is_initialized():
                dist.destroy_process_group()

    def _trace_simple_fsdp_moe_grad_collective(
        self,
        *,
        chunk_strategy: str,
        fsdp_mode: str,
    ):
        class Moe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, bias=False)

            def forward(self, x):
                compute = annotate_fn({"EP": "compute"})(self.linear)
                return torch.relu(compute(x))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].moe = Moe()

            def forward(self, x):
                return self.layers[0].moe(x)

        if chunk_strategy not in {"eager", "graph"}:
            raise AssertionError(f"unknown chunk_strategy {chunk_strategy}")

        with self._fake_dp_mesh() as dp_mesh:
            model = Model()
            annotate_module_fqns(model)
            model = data_parallel(model, device_mesh=dp_mesh, mode=fsdp_mode)
            x = torch.randn(4, 2, 4)
            mark_chunk_dynamic_dims(x, mode="batch")

            if chunk_strategy == "eager":
                maybe_apply_ep_overlap_eager_chunking(
                    model,
                    GraphTrainerCompileConfig(
                        enable=True,
                        ep_overlap=EpOverlapConfig(
                            enabled=True,
                            strategy="eager",
                            chunk_dim="batch",
                            module_fqn="layers.*.moe",
                        ),
                    ),
                )

            def step(inputs):
                y = model(inputs)
                loss = y.sum()
                params = [p for p in model.parameters() if p.requires_grad]
                return [loss] + list(torch.autograd.grad(loss, params))

            traced = minimal_fx_tracer(step, module=model)(x)
            gm = traced.gm
            if chunk_strategy == "eager":
                populate_eager_chunk_metadata_pass(gm)
            else:
                populate_chunk_dim_metadata_pass(
                    gm, traced.example_inputs, mode="batch"
                )
                ep_overlap_chunk_pass(
                    gm,
                    mode="batch",
                    module_pattern="layers.*.moe",
                    num_static_inputs=traced.num_static_inputs,
                    optimize_grad_live_out=False,
                    require_token_exchange=False,
                )
        target = (
            torch.ops._c10d_functional.all_reduce.default
            if fsdp_mode == "replicate"
            else torch.ops._c10d_functional.reduce_scatter_tensor.default
        )
        return gm, target

    def _schedule_ep_overlap_and_order(
        self,
        gm,
        *,
        module_pattern: str = "layers.*.moe",
        pair_first_token_exchange: bool = True,
    ):
        _schedule_ep_overlap_regions(
            gm,
            module_pattern=module_pattern,
            require_token_exchange=True,
            pair_first_token_exchange=pair_first_token_exchange,
        )
        return {node: idx for idx, node in enumerate(gm.graph.nodes)}

    def _assert_nodes_in_order(self, order, nodes):
        for before, after in zip(nodes, nodes[1:]):
            self.assertLess(order[before], order[after])

    def _build_ep_sync_copy_schedule_gm(
        self,
        *,
        fqn: str = "layers.0.moe",
        copies_per_chunk: int = 2,
        cpu_destination: bool = True,
    ):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        refs = {}
        for chunk_id in (0, 1):
            router = graph.call_function(torch.ops.aten.relu.default, args=(x,))
            count = graph.call_function(
                c10d.all_to_all_single.default, args=(router, [], [], "ep")
            )
            body_nodes = [router, count]
            copies = []
            consumers = []
            consumer_value = None
            for copy_idx in range(copies_per_chunk):
                producer = count
                if copy_idx:
                    producer = graph.call_function(
                        torch.ops.aten.neg.default, args=(count,)
                    )
                    body_nodes.append(producer)

                copy_kwargs = {"non_blocking": False}
                if cpu_destination:
                    copy_kwargs["device"] = torch.device("cpu")
                copy = graph.call_function(
                    torch.ops.aten._to_copy.default,
                    args=(producer,),
                    kwargs=copy_kwargs,
                )
                if cpu_destination:
                    copy.meta["val"] = torch.empty(2, device="cpu")
                consumer = graph.call_function(
                    torch.ops.aten._local_scalar_dense.default, args=(copy,)
                )
                copies.append(copy)
                consumers.append(consumer)
                body_nodes.extend((copy, consumer))
                consumer_value = (
                    consumer
                    if consumer_value is None
                    else graph.call_function(
                        torch.ops.aten.add.Tensor, args=(consumer_value, consumer)
                    )
                )
                if consumer_value is not consumer:
                    body_nodes.append(consumer_value)

            if consumer_value is None:
                consumer_value = count
            dispatch = graph.call_function(
                c10d.all_to_all_single.default, args=(consumer_value, [], [], "ep")
            )
            wait = graph.call_function(c10d.wait_tensor.default, args=(dispatch,))
            body_nodes.extend((dispatch, wait))
            for node in body_nodes:
                self._mark_chunk_body(
                    node,
                    fqn=fqn,
                    chunk_id=chunk_id,
                    ep="dispatch",
                    token_exchange=node is dispatch,
                )
            for copy in copies:
                copy.meta["custom"][_EP_TOKEN_COUNT_SYNC] = "dispatch"
            refs[chunk_id] = {
                "copies": tuple(copies),
                "consumers": tuple(consumers),
                "dispatch": dispatch,
                "wait": wait,
            }

        graph.output((refs[0]["wait"], refs[1]["wait"]))
        return torch.fx.GraphModule(torch.nn.Module(), graph), refs

    def _build_hidden_boundary_dep_schedule_gm(
        self, *, producer: str, boundary_role: str | None = None
    ):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        outputs = []
        for chunk_id in (0, 1):
            hidden = graph.call_function(torch.ops.aten.relu.default, args=(x,))
            boundary = graph.call_function(torch.ops.aten.clone.default, args=(hidden,))
            if boundary_role is not None:
                boundary.meta["chunked_region_producer"] = "graph"
                boundary.meta["chunked_region_role"] = boundary_role
            launch = graph.call_function(
                c10d.all_to_all_single.default,
                args=(boundary, [], [], "ep"),
            )
            wait = graph.call_function(c10d.wait_tensor.default, args=(launch,))
            outputs.append(wait)

            for node in (hidden, launch, wait):
                self._mark_chunk_body(
                    node,
                    chunk_id=chunk_id,
                    ep="dispatch" if node is launch else None,
                    token_exchange=node is launch,
                    producer=producer,
                )

        graph.output(tuple(outputs))
        return torch.fx.GraphModule(torch.nn.Module(), graph)

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

    def test_chunk_batch_rejects_non_ep_collective_in_body(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        all_reduce = graph.call_function(
            torch.ops._c10d_functional.all_reduce.default,
            args=(relu, "sum", "dp"),
        )
        graph.output(all_reduce)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode, sym_batch = self._symbolic_batch_fake_mode()
        with fake_mode:
            val = torch.empty(sym_batch, 3)
        x.meta["val"] = val
        for node in (relu, all_reduce):
            node.meta["val"] = val
            node.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}

        with self.assertRaisesRegex(ValueError, "only EP collectives"):
            self._chunk_batch(gm, module_patterns=["layers.*.moe"])

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
                enable_fsdp_dense_region_overlap=False,
                precompile_artifact_dir="",
            ),
        )
        return traced_result, config

    def _compile_pass_names(self, traced_result, config):
        def pass_name(pass_fn):
            return (
                pass_fn.func.__name__ if hasattr(pass_fn, "func") else pass_fn.__name__
            )

        return [
            pass_name(pass_fn)
            for pass_fn in compile_time_passes(
                traced_result, config, use_cudagraph=False
            )
        ]

    def test_ep_overlap_pass_pipeline_order(self):
        traced_result, config = self._compile_config_for_ep_overlap_test()
        names = self._compile_pass_names(traced_result, config)
        dead_code_indices = [
            i for i, name in enumerate(names) if name == "eliminate_dead_code_pass"
        ]
        chunk_pass_idx = names.index("ep_overlap_chunk_pass")
        post_chunk_dce = min(i for i in dead_code_indices if i > chunk_pass_idx)

        self.assertLess(
            names.index("canonicalize_graph_pass"),
            names.index("deduplicate_fsdp_unshard_chains_pass"),
        )
        self.assertLess(
            names.index("deduplicate_fsdp_unshard_chains_pass"),
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
        self.assertLess(names.index("populate_chunk_dim_metadata_pass"), chunk_pass_idx)
        self.assertLess(chunk_pass_idx, names.index("isolate_ep_process_group_pass"))
        self.assertLess(
            names.index("isolate_ep_process_group_pass"),
            post_chunk_dce,
        )
        self.assertLess(
            post_chunk_dce,
            names.index("assign_minimal_async_ep_buffer_sets_pass"),
        )
        self.assertLess(
            names.index("assign_minimal_async_ep_buffer_sets_pass"),
            names.index("normalize_chunked_grad_collective_chains_pass"),
        )
        self.assertLess(
            names.index("normalize_chunked_grad_collective_chains_pass"),
            names.index("joint_transformer_block_bucketing_reordering_pass"),
        )
        self.assertLess(
            names.index("joint_transformer_block_bucketing_reordering_pass"),
            names.index("ep_overlap_schedule_pass"),
        )
        self.assertLess(
            names.index("ep_overlap_schedule_pass"),
            names.index("concretize_ep_chunk_symbolic_shapes_pass"),
        )
        self.assertLess(
            names.index("concretize_ep_chunk_symbolic_shapes_pass"),
            names.index("full_inductor_compilation_pass"),
        )

        config.compile.ep_overlap.disable_early_grad_accumulation = True
        disabled_names = self._compile_pass_names(traced_result, config)
        self.assertNotIn(
            "normalize_chunked_grad_collective_chains_pass",
            disabled_names,
        )
        disabled_chunk_pass_idx = disabled_names.index("ep_overlap_chunk_pass")
        disabled_dead_code_indices = [
            i
            for i, name in enumerate(disabled_names)
            if name == "eliminate_dead_code_pass"
        ]
        disabled_post_chunk_dce = min(
            i for i in disabled_dead_code_indices if i > disabled_chunk_pass_idx
        )
        self.assertLess(
            disabled_post_chunk_dce,
            disabled_names.index("joint_transformer_block_bucketing_reordering_pass"),
        )

        config.compile.ep_overlap.strategy = "eager"
        eager_names = self._compile_pass_names(traced_result, config)
        self.assertNotIn("assign_minimal_async_ep_buffer_sets_pass", eager_names)

    def test_graph_ep_chunking_rejects_tensor_parallel(self):
        cases = (
            ("seq", "layers.*.moe"),
            ("batch", "layers.*"),
        )
        for chunk_dim, module_fqn in cases:
            with self.subTest(chunk_dim=chunk_dim):
                traced_result, config = self._compile_config_for_ep_overlap_test()
                config.compile.ep_overlap.chunk_dim = chunk_dim
                config.compile.ep_overlap.module_fqn = module_fqn
                config.parallelism.tensor_parallel_degree = 2

                with self.assertRaisesRegex(
                    ValueError,
                    "Graph EP chunking does not support tensor_parallel_degree > 1",
                ):
                    compile_time_passes(traced_result, config, use_cudagraph=False)

    def test_fsdp_dense_region_scheduler_pass_gating(self):
        def transformer_batch_default(config):
            pass

        def transformer_batch_explicit(config):
            config.compile.enable_fsdp_dense_region_overlap = True

        def moe_ep_default(config):
            config.compile.ep_overlap.module_fqn = "layers.*.moe"

        def moe_ep_explicit(config):
            config.compile.ep_overlap.module_fqn = "layers.*.moe"
            config.compile.enable_fsdp_dense_region_overlap = True

        def fsdp_dense_without_ep(config):
            config.compile.ep_overlap.enabled = False
            config.compile.enable_fsdp_dense_region_overlap = True

        cases = (
            ("transformer_batch_default", transformer_batch_default, True, False, None),
            (
                "transformer_batch_explicit",
                transformer_batch_explicit,
                True,
                False,
                "graph chunking.*layers.*.moe",
            ),
            ("moe_ep_default", moe_ep_default, True, False, None),
            ("moe_ep_explicit", moe_ep_explicit, True, True, None),
            ("fsdp_dense_without_ep", fsdp_dense_without_ep, False, True, None),
        )
        for (
            name,
            configure,
            expects_ep_schedule,
            expects_fsdp_schedule,
            warning,
        ) in cases:
            with self.subTest(name=name):
                traced_result, config = self._compile_config_for_ep_overlap_test()
                configure(config)
                if warning is None:
                    names = self._compile_pass_names(traced_result, config)
                else:
                    with self.assertWarnsRegex(UserWarning, warning):
                        names = self._compile_pass_names(traced_result, config)

                self.assertEqual(
                    "ep_overlap_schedule_pass" in names,
                    expects_ep_schedule,
                )
                self.assertEqual(
                    "schedule_fsdp_comms_to_dense_regions_pass" in names,
                    expects_fsdp_schedule,
                )

    def test_moe_efsdp_bucket_plan_splits_expert_buckets(self):
        buckets = get_default_transformer_block_buckets(
            3,
            moe_layer_ids=frozenset({1}),
            split_moe_expert_buckets=True,
        )

        self.assertIn(
            [
                "layers.1.attention_norm",
                "layers.1.attention",
                "layers.1.ffn_norm",
                "layers.1.moe.router",
                "layers.1.moe.shared_experts",
            ],
            buckets,
        )
        self.assertIn("layers.1.moe.routed_experts.inner_experts", buckets)
        self.assertNotIn("layers.1", buckets)

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
        from torchtitan.components.loss import ChunkedLossWrapper

        torch.manual_seed(42)
        batch, seq_len, dim, vocab_size = 2, 32, 4, 8
        lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        loss_fn = ChunkedLossWrapper(ChunkedLossWrapper.Config(num_chunks=8))
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

    def test_moe_ep_annotations_cover_all_to_all_dispatcher(self):
        from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

        annotate_moe_ep_regions()

        expected_annotations = [
            (AllToAllTokenDispatcher.dispatch, {"EP": "dispatch"}),
            (
                AllToAllTokenDispatcher._token_count_exchange,
                {_EP_TOKEN_COUNT_EXCHANGE: "dispatch"},
            ),
            (
                AllToAllTokenDispatcher._sync_token_count_exchange,
                {_EP_TOKEN_COUNT_SYNC: "dispatch"},
            ),
            (
                AllToAllTokenDispatcher._dispatch_token_exchange,
                {_EP_TOKEN_EXCHANGE: "dispatch"},
            ),
            (
                AllToAllTokenDispatcher._combine_token_exchange,
                {_EP_TOKEN_EXCHANGE: "combine"},
            ),
            (AllToAllTokenDispatcher.combine, {"EP": "combine"}),
        ]
        for method, annotation in expected_annotations:
            self.assertEqual(
                inspect.getclosurevars(method).nonlocals["annotation_dict"],
                annotation,
            )

    def test_ep_overlap_reorders_forward_and_backward_token_exchange_blocks(self):
        for backward, first_chunk in ((False, 0), (True, 1)):
            with self.subTest(backward=backward):
                gm = self._build_ep_overlap_schedule_gm(backward=backward)
                order = self._schedule_ep_overlap_and_order(gm)
                nodes = list(gm.graph.nodes)

                def named(chunk_id, target, ep=None):
                    matches = [
                        node
                        for node in nodes
                        if node.meta.get("chunk_id") == chunk_id
                        and node.target == target
                        and (
                            ep is None
                            or node.meta.get("custom", {}).get(_EP_TOKEN_EXCHANGE) == ep
                            or node.meta.get("custom", {}).get("EP") == ep
                        )
                    ]
                    self.assertEqual(len(matches), 1)
                    return matches[0]

                second_chunk = 1 - first_chunk
                c10d = torch.ops._c10d_functional
                first_dispatch_launch = named(
                    first_chunk,
                    c10d.all_to_all_single.default,
                    ep="dispatch",
                )
                second_dispatch_launch = named(
                    second_chunk,
                    c10d.all_to_all_single.default,
                    ep="dispatch",
                )
                first_dispatch_wait = next(iter(first_dispatch_launch.users))
                first_combine_launch = named(
                    first_chunk,
                    c10d.all_to_all_single.default,
                    ep="combine",
                )
                second_combine_launch = named(
                    second_chunk,
                    c10d.all_to_all_single.default,
                    ep="combine",
                )

                self.assertLess(
                    order[first_dispatch_launch], order[second_dispatch_launch]
                )
                self.assertLess(
                    order[second_dispatch_launch], order[first_dispatch_wait]
                )
                self.assertLess(
                    order[first_combine_launch], order[second_combine_launch]
                )
                if not backward:
                    self.assertLess(
                        order[first_combine_launch],
                        order[next(iter(second_dispatch_launch.users))],
                    )
                else:
                    self.assertLess(
                        order[second_dispatch_launch],
                        order[next(iter(first_dispatch_launch.users))],
                    )

    def test_ep_overlap_schedules_moe_shaped_forward_and_backward_order(self):
        c10d = torch.ops._c10d_functional

        def build_forward_gm():
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            refs = {}
            for chunk_id in (0, 1):
                router = graph.call_function(torch.ops.aten.relu.default, args=(x,))
                count = graph.call_function(
                    c10d.all_to_all_single.default, args=(router, [], [], "ep")
                )
                sync = graph.call_function(
                    torch.ops.aten._local_scalar_dense.default, args=(count,)
                )
                dispatch = graph.call_function(
                    c10d.all_to_all_single.default, args=(sync, [], [], "ep")
                )
                shared = graph.call_function(torch.ops.aten.neg.default, args=(x,))
                dispatch_wait = graph.call_function(
                    c10d.wait_tensor.default, args=(dispatch,)
                )
                grouped_mm = graph.call_function(
                    torch.ops.aten.relu.default, args=(dispatch_wait,)
                )
                combine = graph.call_function(
                    c10d.all_to_all_single.default, args=(grouped_mm, [], [], "ep")
                )
                combine_wait = graph.call_function(
                    c10d.wait_tensor.default, args=(combine,)
                )
                refs[chunk_id] = {
                    "router": router,
                    "count": count,
                    "sync": sync,
                    "dispatch": dispatch,
                    "shared": shared,
                    "dispatch_wait": dispatch_wait,
                    "grouped_mm": grouped_mm,
                    "combine": combine,
                    "combine_wait": combine_wait,
                }
                for node in (router, count, sync, dispatch, dispatch_wait):
                    self._mark_chunk_body(
                        node,
                        chunk_id=chunk_id,
                        ep="dispatch",
                        token_exchange=node is dispatch,
                    )
                for node in (shared, grouped_mm):
                    self._mark_chunk_body(node, chunk_id=chunk_id)
                for node in (combine, combine_wait):
                    self._mark_chunk_body(
                        node,
                        chunk_id=chunk_id,
                        ep="combine",
                        token_exchange=node is combine,
                    )
            graph.output((refs[0]["combine_wait"], refs[1]["combine_wait"]))
            return torch.fx.GraphModule(torch.nn.Module(), graph), refs

        def build_backward_gm():
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            refs = {}
            for chunk_id in (0, 1):
                combine = graph.call_function(
                    c10d.all_to_all_single.default, args=(x, [], [], "ep")
                )
                remat_grouped_mm = graph.call_function(
                    torch.ops.aten.neg.default, args=(x,)
                )
                combine_wait = graph.call_function(
                    c10d.wait_tensor.default, args=(combine,)
                )
                input_grad = graph.call_function(
                    torch.ops.aten.relu.default, args=(combine_wait,)
                )
                dispatch = graph.call_function(
                    c10d.all_to_all_single.default, args=(input_grad, [], [], "ep")
                )
                wgrad = graph.call_function(
                    torch.ops.aten.neg.default, args=(dispatch,)
                )
                dispatch_wait = graph.call_function(
                    c10d.wait_tensor.default, args=(dispatch,)
                )
                refs[chunk_id] = {
                    "combine": combine,
                    "remat_grouped_mm": remat_grouped_mm,
                    "combine_wait": combine_wait,
                    "input_grad": input_grad,
                    "dispatch": dispatch,
                    "wgrad": wgrad,
                    "dispatch_wait": dispatch_wait,
                }
                for node in (combine, combine_wait):
                    self._mark_chunk_body(
                        node,
                        chunk_id=chunk_id,
                        backward=True,
                        ep="combine",
                        token_exchange=node is combine,
                    )
                for node in (remat_grouped_mm, input_grad, wgrad):
                    self._mark_chunk_body(node, chunk_id=chunk_id, backward=True)
                for node in (dispatch, dispatch_wait):
                    self._mark_chunk_body(
                        node,
                        chunk_id=chunk_id,
                        backward=True,
                        ep="dispatch",
                        token_exchange=node is dispatch,
                    )
            graph.output((refs[1]["dispatch_wait"], refs[0]["dispatch_wait"]))
            return torch.fx.GraphModule(torch.nn.Module(), graph), refs

        fwd_gm, fwd = build_forward_gm()
        fwd_order = self._schedule_ep_overlap_and_order(fwd_gm)
        self._assert_nodes_in_order(
            fwd_order,
            [
                fwd[0]["router"],
                fwd[0]["count"],
                fwd[0]["sync"],
                fwd[1]["router"],
                fwd[1]["count"],
                fwd[1]["sync"],
                fwd[0]["dispatch"],
                fwd[1]["dispatch"],
                fwd[0]["shared"],
                fwd[1]["shared"],
                fwd[0]["dispatch_wait"],
                fwd[0]["grouped_mm"],
                fwd[0]["combine"],
                fwd[1]["dispatch_wait"],
                fwd[1]["grouped_mm"],
                fwd[1]["combine"],
                fwd[0]["combine_wait"],
                fwd[1]["combine_wait"],
            ],
        )

        bwd_gm, bwd = build_backward_gm()
        bwd_order = self._schedule_ep_overlap_and_order(bwd_gm)
        self._assert_nodes_in_order(
            bwd_order,
            [
                bwd[1]["combine"],
                bwd[0]["combine"],
                bwd[1]["remat_grouped_mm"],
                bwd[0]["remat_grouped_mm"],
                bwd[1]["combine_wait"],
                bwd[1]["input_grad"],
                bwd[1]["dispatch"],
                bwd[0]["combine_wait"],
                bwd[0]["input_grad"],
                bwd[0]["dispatch"],
                bwd[1]["wgrad"],
                bwd[0]["wgrad"],
                bwd[1]["dispatch_wait"],
                bwd[0]["dispatch_wait"],
            ],
        )

    def test_ep_overlap_schedules_minimal_async_ep_markers(self):
        import torchtitan.distributed.minimal_async_ep  # noqa: F401

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        ranks = graph.placeholder("ranks")
        rows = graph.placeholder("rows")
        valid = graph.placeholder("valid")
        refs = {}
        ops = torch.ops.minimal_async_ep

        for chunk_id in (0, 1):
            pre = graph.call_function(torch.ops.aten.relu.default, args=(x,))
            dispatch = graph.call_function(
                ops.dispatch.default,
                args=(pre, ranks, rows, 8, 2),
            )
            dispatch_hidden = graph.call_function(
                operator.getitem,
                args=(dispatch, 0),
            )
            dispatch_wait = graph.call_function(
                ops.wait_dispatch.default,
                args=(dispatch_hidden, [pre]),
            )
            compute = graph.call_function(
                torch.ops.aten.neg.default, args=(dispatch_wait,)
            )
            combine = graph.call_function(
                ops.combine_data.default,
                args=(compute, ranks, rows, valid, 4),
            )
            combine_wait = graph.call_function(
                ops.wait_combine.default,
                args=(combine, [compute, ranks, rows, valid]),
            )
            tail = graph.call_function(
                torch.ops.aten.relu.default, args=(combine_wait,)
            )
            refs[chunk_id] = {
                "dispatch": dispatch,
                "dispatch_hidden": dispatch_hidden,
                "dispatch_wait": dispatch_wait,
                "compute": compute,
                "combine": combine,
                "combine_wait": combine_wait,
                "tail": tail,
            }

            for node in (pre, dispatch, dispatch_hidden, dispatch_wait):
                self._mark_chunk_body(
                    node,
                    chunk_id=chunk_id,
                    ep="dispatch",
                )
            self._mark_chunk_body(compute, chunk_id=chunk_id)
            for node in (combine, combine_wait, tail):
                self._mark_chunk_body(
                    node,
                    chunk_id=chunk_id,
                    ep="combine" if node is not tail else None,
                )

        graph.output((refs[0]["tail"], refs[1]["tail"]))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        order = self._schedule_ep_overlap_and_order(gm)

        self.assertEqual(
            refs[0]["dispatch_wait"].meta["custom"][_EP_TOKEN_EXCHANGE_WAIT],
            "dispatch",
        )
        self.assertEqual(
            refs[0]["combine_wait"].meta["custom"][_EP_TOKEN_EXCHANGE_WAIT],
            "combine",
        )
        self._assert_nodes_in_order(
            order,
            [
                refs[0]["dispatch"],
                refs[1]["dispatch"],
                refs[0]["dispatch_wait"],
                refs[0]["compute"],
                refs[0]["combine"],
                refs[1]["dispatch_wait"],
                refs[1]["compute"],
                refs[1]["combine"],
                refs[0]["combine_wait"],
                refs[0]["tail"],
                refs[1]["combine_wait"],
                refs[1]["tail"],
            ],
        )

    def test_ep_overlap_deduplicates_peer_chunk_ready_candidates(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        candidate = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        peer = graph.call_function(torch.ops.aten.neg.default, args=(x,))
        graph.output((candidate, peer))

        owners = {
            0: ChunkOwner("layers.0", True, 0),
            1: ChunkOwner("layers.0", True, 1),
        }
        bodies = {
            0: ChunkBody(
                owners[0], (candidate,), frozenset({candidate}), frozenset(), "test"
            ),
            1: ChunkBody(owners[1], (peer,), frozenset({peer}), frozenset(), "test"),
        }
        region = ChunkedRegion("layers.0", True, bodies)

        ready = _ready_nodes(
            candidates_by_chunk={0: {candidate}, 1: {candidate}},
            emitted=set(),
            region=region,
            chunk_order=(1, 0),
            order={node: idx for idx, node in enumerate(graph.nodes)},
            owner_by_node={candidate: owners[0], peer: owners[1]},
            include_waits=True,
        )

        self.assertEqual(ready, (candidate,))

    def test_graph_chunk_assigns_minimal_async_ep_buffer_sets(self):
        import torchtitan.distributed.minimal_async_ep  # noqa: F401

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        ranks = graph.placeholder("ranks")
        rows = graph.placeholder("rows")
        valid = graph.placeholder("valid")
        launches = []
        for chunk_id in (0, 1):
            dispatch = graph.call_function(
                torch.ops.minimal_async_ep.dispatch_data.default,
                args=(x, ranks, rows, 8, 4),
            )
            combine = graph.call_function(
                torch.ops.minimal_async_ep.combine_data.default,
                args=(dispatch, ranks, rows, valid, 4),
            )
            for launch in (dispatch, combine):
                self._mark_chunk_body(launch, chunk_id=chunk_id)
                launches.append((launch, chunk_id))
        graph.output(tuple(launch for launch, _ in launches))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        assign_minimal_async_ep_buffer_sets_pass(gm)

        for launch, chunk_id in launches:
            self.assertEqual(launch.kwargs["buffer_set"], chunk_id)

    def test_ep_overlap_keeps_transformer_batch_first_marker_wait_gated(self):
        c10d = torch.ops._c10d_functional

        def build_forward_gm():
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            refs = {}
            for chunk_id in (0, 1):
                dense_prefix = graph.call_function(
                    torch.ops.aten.relu.default, args=(x,)
                )
                router = graph.call_function(
                    torch.ops.aten.neg.default, args=(dense_prefix,)
                )
                count = graph.call_function(
                    c10d.all_to_all_single.default, args=(router, [], [], "ep")
                )
                sync = graph.call_function(
                    torch.ops.aten._local_scalar_dense.default, args=(count,)
                )
                dispatch = graph.call_function(
                    c10d.all_to_all_single.default, args=(sync, [], [], "ep")
                )
                dispatch_wait = graph.call_function(
                    c10d.wait_tensor.default, args=(dispatch,)
                )
                grouped_mm = graph.call_function(
                    torch.ops.aten.relu.default, args=(dispatch_wait,)
                )
                combine = graph.call_function(
                    c10d.all_to_all_single.default, args=(grouped_mm, [], [], "ep")
                )
                combine_wait = graph.call_function(
                    c10d.wait_tensor.default, args=(combine,)
                )
                refs[chunk_id] = {
                    "dense_prefix": dense_prefix,
                    "router": router,
                    "count": count,
                    "sync": sync,
                    "dispatch": dispatch,
                    "dispatch_wait": dispatch_wait,
                    "grouped_mm": grouped_mm,
                    "combine": combine,
                    "combine_wait": combine_wait,
                }
                for node in (dense_prefix, router, count, sync, dispatch):
                    self._mark_chunk_body(
                        node,
                        fqn="layers.0",
                        chunk_id=chunk_id,
                        ep="dispatch",
                        token_exchange=node is dispatch,
                    )
                for node in (dispatch_wait, grouped_mm, combine, combine_wait):
                    self._mark_chunk_body(
                        node,
                        fqn="layers.0",
                        chunk_id=chunk_id,
                        ep="combine",
                        token_exchange=node is combine,
                    )
            graph.output((refs[0]["combine_wait"], refs[1]["combine_wait"]))
            return torch.fx.GraphModule(torch.nn.Module(), graph), refs

        def build_backward_gm():
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            refs = {}
            for chunk_id in (0, 1):
                remat_prefix = graph.call_function(
                    torch.ops.aten.relu.default, args=(x,)
                )
                combine = graph.call_function(
                    c10d.all_to_all_single.default, args=(remat_prefix, [], [], "ep")
                )
                combine_wait = graph.call_function(
                    c10d.wait_tensor.default, args=(combine,)
                )
                input_grad = graph.call_function(
                    torch.ops.aten.relu.default, args=(combine_wait,)
                )
                dispatch = graph.call_function(
                    c10d.all_to_all_single.default, args=(input_grad, [], [], "ep")
                )
                dispatch_wait = graph.call_function(
                    c10d.wait_tensor.default, args=(dispatch,)
                )
                refs[chunk_id] = {
                    "remat_prefix": remat_prefix,
                    "combine": combine,
                    "combine_wait": combine_wait,
                    "input_grad": input_grad,
                    "dispatch": dispatch,
                    "dispatch_wait": dispatch_wait,
                }
                for node in (remat_prefix, combine, combine_wait):
                    self._mark_chunk_body(
                        node,
                        fqn="layers.0",
                        chunk_id=chunk_id,
                        backward=True,
                        ep="combine",
                        token_exchange=node is combine,
                    )
                for node in (input_grad, dispatch, dispatch_wait):
                    self._mark_chunk_body(
                        node,
                        fqn="layers.0",
                        chunk_id=chunk_id,
                        backward=True,
                        ep="dispatch",
                        token_exchange=node is dispatch,
                    )
            graph.output((refs[1]["dispatch_wait"], refs[0]["dispatch_wait"]))
            return torch.fx.GraphModule(torch.nn.Module(), graph), refs

        fwd_gm, fwd = build_forward_gm()
        fwd_order = self._schedule_ep_overlap_and_order(
            fwd_gm,
            module_pattern="layers.*",
            pair_first_token_exchange=False,
        )
        self._assert_nodes_in_order(
            fwd_order,
            [
                fwd[0]["dense_prefix"],
                fwd[0]["dispatch"],
                fwd[1]["dense_prefix"],
                fwd[1]["dispatch"],
                fwd[0]["dispatch_wait"],
                fwd[0]["combine"],
                fwd[1]["dispatch_wait"],
                fwd[1]["combine"],
            ],
        )

        bwd_gm, bwd = build_backward_gm()
        bwd_order = self._schedule_ep_overlap_and_order(
            bwd_gm,
            module_pattern="layers.*",
            pair_first_token_exchange=False,
        )
        self._assert_nodes_in_order(
            bwd_order,
            [
                bwd[1]["remat_prefix"],
                bwd[1]["combine"],
                bwd[0]["remat_prefix"],
                bwd[0]["combine"],
                bwd[1]["combine_wait"],
                bwd[1]["dispatch"],
                bwd[0]["combine_wait"],
                bwd[0]["dispatch"],
                bwd[1]["dispatch_wait"],
                bwd[0]["dispatch_wait"],
            ],
        )

    def test_ep_overlap_pairs_token_count_sync_cpu_copies_before_consumers(self):
        gm, refs = self._build_ep_sync_copy_schedule_gm()

        order = self._schedule_ep_overlap_and_order(gm)

        copies = refs[0]["copies"] + refs[1]["copies"]
        consumers = refs[0]["consumers"] + refs[1]["consumers"]
        self.assertEqual(
            [copy.kwargs["non_blocking"] for copy in copies],
            [True, True, True, False],
        )
        self.assertLess(
            max(order[copy] for copy in copies),
            min(order[consumer] for consumer in consumers),
        )
        self.assertLess(
            max(order[consumer] for consumer in consumers),
            min(order[refs[chunk_id]["dispatch"]] for chunk_id in (0, 1)),
        )

    def test_ep_overlap_hoists_token_count_sync_cpu_copies_per_chunk(self):
        gm, refs = self._build_ep_sync_copy_schedule_gm(fqn="layers.0")

        order = self._schedule_ep_overlap_and_order(
            gm,
            module_pattern="layers.*",
            pair_first_token_exchange=False,
        )

        for chunk_id in (0, 1):
            self.assertEqual(
                [copy.kwargs["non_blocking"] for copy in refs[chunk_id]["copies"]],
                [True, False],
            )
            self.assertLess(
                max(order[copy] for copy in refs[chunk_id]["copies"]),
                min(order[consumer] for consumer in refs[chunk_id]["consumers"]),
            )
            self.assertLess(
                max(order[consumer] for consumer in refs[chunk_id]["consumers"]),
                order[refs[chunk_id]["dispatch"]],
            )
        self.assertLess(order[refs[0]["dispatch"]], order[refs[1]["copies"][0]])

    def test_ep_overlap_rejects_malformed_token_count_sync_copy_count(self):
        gm, _refs = self._build_ep_sync_copy_schedule_gm(copies_per_chunk=1)

        with self.assertRaisesRegex(ValueError, "exactly two token-count sync"):
            _schedule_ep_overlap_regions(
                gm,
                module_pattern="layers.*.moe",
                require_token_exchange=True,
                pair_first_token_exchange=True,
            )

    def test_ep_overlap_rejects_ambiguous_token_count_sync_copy_destination(self):
        gm, _refs = self._build_ep_sync_copy_schedule_gm(cpu_destination=False)

        with self.assertRaisesRegex(ValueError, "CPU _to_copy destinations"):
            _schedule_ep_overlap_regions(
                gm,
                module_pattern="layers.*.moe",
                require_token_exchange=True,
                pair_first_token_exchange=True,
            )

    def test_ep_overlap_rejects_unannotated_all_to_all_markers(self):
        gm = self._build_ep_overlap_schedule_gm()
        c10d = torch.ops._c10d_functional
        launches = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == c10d.all_to_all_single.default
        ]
        for node in launches:
            custom = dict(node.meta.get("custom", {}))
            custom.pop(_EP_TOKEN_EXCHANGE, None)
            node.meta["custom"] = custom

        with self.assertRaises(ValueError, msg="did not find any chunked EP"):
            _schedule_ep_overlap_regions(
                gm,
                module_pattern="layers.*.moe",
                require_token_exchange=True,
                pair_first_token_exchange=True,
            )

    def test_ep_overlap_rejects_token_exchange_metadata_on_compute(self):
        gm = self._build_ep_overlap_schedule_gm()
        compute = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten.neg.default
            and node.meta.get("chunked_region_role") == "body"
        )
        compute.meta.setdefault("custom", {})[_EP_TOKEN_EXCHANGE] = "dispatch"

        with self.assertRaisesRegex(ValueError, "non-marker node"):
            _schedule_ep_overlap_regions(
                gm,
                module_pattern="layers.*.moe",
                require_token_exchange=True,
                pair_first_token_exchange=True,
            )

    def test_ep_overlap_rejects_mismatched_token_exchange_labels(self):
        gm = self._build_ep_overlap_schedule_gm()
        launch = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops._c10d_functional.all_to_all_single.default
            and node.meta.get("chunk_id") == 0
            and node.meta.get("custom", {}).get(_EP_TOKEN_EXCHANGE) == "dispatch"
        )
        launch.meta["custom"][_EP_TOKEN_EXCHANGE] = "combine"

        with self.assertRaisesRegex(ValueError, "matching token-exchange labels"):
            _schedule_ep_overlap_regions(
                gm,
                module_pattern="layers.*.moe",
                require_token_exchange=True,
                pair_first_token_exchange=True,
            )

    def test_ep_overlap_rejects_wait_before_token_exchange_launch(self):
        gm = self._build_ep_overlap_schedule_gm()
        launch = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops._c10d_functional.all_to_all_single.default
            and node.meta.get("custom", {}).get(_EP_TOKEN_EXCHANGE) == "dispatch"
        )
        wait = next(
            user
            for user in launch.users
            if user.target == torch.ops._c10d_functional.wait_tensor.default
        )
        launch.prepend(wait)

        with self.assertRaisesRegex(ValueError, "wait to appear after its launch"):
            _schedule_ep_overlap_regions(
                gm,
                module_pattern="layers.*.moe",
                require_token_exchange=True,
                pair_first_token_exchange=True,
            )

    def test_ep_overlap_uses_future_closure_prefix_before_wait(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        refs = {}

        # Build the graph in the opposite order from the desired backward
        # schedule. This catches accidental sorting by original graph order when
        # a ready filler phase intentionally contains chunk1 work before chunk0.
        for chunk_id in (0, 1):
            combine_pre = graph.call_function(torch.ops.aten.relu.default, args=(x,))
            combine = graph.call_function(
                c10d.all_to_all_single.default,
                args=(combine_pre, [], [], "ep"),
            )
            dispatch_prefix = graph.call_function(torch.ops.aten.neg.default, args=(x,))
            combine_wait = graph.call_function(
                c10d.wait_tensor.default, args=(combine,)
            )
            dispatch_pre = graph.call_function(
                torch.ops.aten.add.Tensor, args=(dispatch_prefix, combine_wait)
            )
            dispatch = graph.call_function(
                c10d.all_to_all_single.default,
                args=(dispatch_pre, [], [], "ep"),
            )
            dispatch_wait = graph.call_function(
                c10d.wait_tensor.default, args=(dispatch,)
            )
            tail = graph.call_function(
                torch.ops.aten.relu.default, args=(dispatch_wait,)
            )
            refs[chunk_id] = {
                "combine": combine,
                "dispatch_prefix": dispatch_prefix,
                "combine_wait": combine_wait,
                "dispatch_pre": dispatch_pre,
                "dispatch": dispatch,
                "dispatch_wait": dispatch_wait,
                "tail": tail,
            }

            for node in (combine_pre, combine, combine_wait):
                node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "combine"}
            combine.meta["custom"][_EP_TOKEN_EXCHANGE] = "combine"
            dispatch_prefix.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
            dispatch_pre.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
            for node in (dispatch, dispatch_wait):
                node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "dispatch"}
            dispatch.meta["custom"][_EP_TOKEN_EXCHANGE] = "dispatch"
            tail.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
            for node in (
                combine_pre,
                combine,
                dispatch_prefix,
                combine_wait,
                dispatch_pre,
                dispatch,
                dispatch_wait,
                tail,
            ):
                node.meta["chunk_id"] = chunk_id
                node.meta["chunked_region_fqn"] = "layers.0.moe"
                node.meta["chunked_region_role"] = "body"
                node.meta["autograd_backward"] = True

        graph.output((refs[1]["tail"], refs[0]["tail"]))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        order = self._schedule_ep_overlap_and_order(gm)

        self.assertEqual(
            refs[1]["combine_wait"].meta["custom"][_EP_TOKEN_EXCHANGE_WAIT],
            "combine",
        )
        self.assertLess(order[refs[1]["combine"]], order[refs[0]["combine"]])
        self.assertLess(order[refs[0]["combine"]], order[refs[1]["dispatch_prefix"]])
        self.assertLess(
            order[refs[1]["dispatch_prefix"]],
            order[refs[0]["dispatch_prefix"]],
        )
        self.assertLess(
            order[refs[0]["dispatch_prefix"]], order[refs[1]["combine_wait"]]
        )
        self.assertLess(order[refs[1]["combine_wait"]], order[refs[1]["dispatch"]])
        self.assertLess(order[refs[1]["dispatch"]], order[refs[0]["combine_wait"]])
        self.assertLess(order[refs[0]["combine_wait"]], order[refs[0]["dispatch"]])
        self.assertLess(order[refs[0]["dispatch"]], order[refs[1]["dispatch_wait"]])
        self.assertLess(order[refs[1]["dispatch_wait"]], order[refs[1]["tail"]])
        self.assertLess(order[refs[1]["tail"]], order[refs[0]["dispatch_wait"]])
        self.assertLess(order[refs[0]["dispatch_wait"]], order[refs[0]["tail"]])

    def test_ep_overlap_does_not_hoist_non_token_collectives_as_filler(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        refs = {}

        for chunk_id in (0, 1):
            dispatch_pre = graph.call_function(torch.ops.aten.relu.default, args=(x,))
            dispatch = graph.call_function(
                c10d.all_to_all_single.default,
                args=(dispatch_pre, [], [], "ep"),
            )
            dispatch_wait = graph.call_function(
                c10d.wait_tensor.default, args=(dispatch,)
            )
            all_reduce = graph.call_function(
                c10d.all_reduce.default, args=(x, "sum", "dp")
            )
            combine_pre = graph.call_function(
                torch.ops.aten.add.Tensor, args=(dispatch_wait, all_reduce)
            )
            combine = graph.call_function(
                c10d.all_to_all_single.default,
                args=(combine_pre, [], [], "ep"),
            )
            combine_wait = graph.call_function(
                c10d.wait_tensor.default, args=(combine,)
            )
            refs[chunk_id] = {
                "dispatch": dispatch,
                "dispatch_wait": dispatch_wait,
                "all_reduce": all_reduce,
                "combine": combine,
                "combine_wait": combine_wait,
            }

            for node in (dispatch_pre, dispatch, dispatch_wait):
                node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "dispatch"}
            dispatch.meta["custom"][_EP_TOKEN_EXCHANGE] = "dispatch"
            all_reduce.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
            combine_pre.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
            for node in (combine, combine_wait):
                node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "combine"}
            combine.meta["custom"][_EP_TOKEN_EXCHANGE] = "combine"

            for node in (
                dispatch_pre,
                dispatch,
                dispatch_wait,
                all_reduce,
                combine_pre,
                combine,
                combine_wait,
            ):
                node.meta["chunk_id"] = chunk_id
                node.meta["chunked_region_fqn"] = "layers.0.moe"
                node.meta["chunked_region_role"] = "body"

        graph.output((refs[0]["combine_wait"], refs[1]["combine_wait"]))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        order = self._schedule_ep_overlap_and_order(gm)

        self.assertLess(order[refs[0]["dispatch"]], order[refs[1]["dispatch"]])
        self.assertLess(order[refs[1]["dispatch"]], order[refs[0]["dispatch_wait"]])
        for chunk_id in (0, 1):
            self.assertLess(
                order[refs[chunk_id]["dispatch_wait"]],
                order[refs[chunk_id]["all_reduce"]],
            )
            self.assertLess(
                order[refs[chunk_id]["all_reduce"]],
                order[refs[chunk_id]["combine"]],
            )

    def test_ep_overlap_rejects_mismatched_token_exchange_count(self):
        gm = self._build_ep_overlap_schedule_gm(backward=True)
        c10d = torch.ops._c10d_functional
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == c10d.all_to_all_single.default
                and node.meta.get("chunk_id") == 0
            ):
                node.target = c10d.all_reduce.default
                node.args = (node.args[0], "sum", "ep")
                node.meta["custom"].pop(_EP_TOKEN_EXCHANGE, None)
                break
        with self.assertRaisesRegex(ValueError, "matching EP token-exchange counts"):
            _schedule_ep_overlap_regions(
                gm,
                module_pattern="layers.*.moe",
                require_token_exchange=True,
                pair_first_token_exchange=True,
            )

    def test_ep_overlap_schedules_arbitrary_matching_token_exchange_sequence(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        outputs = []
        launches = {}
        exchange_kinds = ("combine", "dispatch", "combine")
        for chunk_id in (1, 0):
            value = x
            body_nodes = []
            for idx, kind in enumerate(exchange_kinds):
                pre = graph.call_function(torch.ops.aten.relu.default, args=(value,))
                launch = graph.call_function(
                    c10d.all_to_all_single.default,
                    args=(pre, [], [], "ep"),
                )
                wait = graph.call_function(c10d.wait_tensor.default, args=(launch,))
                launches[(chunk_id, idx)] = launch
                for node in (pre, launch, wait):
                    node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": kind}
                launch.meta["custom"][_EP_TOKEN_EXCHANGE] = kind
                value = graph.call_function(torch.ops.aten.neg.default, args=(wait,))
                value.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
                body_nodes.extend((pre, launch, wait, value))
            outputs.append(value)
            for node in body_nodes:
                node.meta["chunk_id"] = chunk_id
                node.meta["chunked_region_fqn"] = "layers.0.moe"
                node.meta["chunked_region_role"] = "body"
                node.meta["autograd_backward"] = True

        graph.output(tuple(outputs))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        order = self._schedule_ep_overlap_and_order(gm)
        for idx in range(len(exchange_kinds)):
            self.assertLess(order[launches[(1, idx)]], order[launches[(0, idx)]])
            if idx + 1 < len(exchange_kinds):
                self.assertLess(
                    order[launches[(0, idx)]], order[launches[(1, idx + 1)]]
                )

    def test_ep_overlap_reorders_non_body_setup_dependencies(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional
        outputs = []
        setup_nodes = {}
        first_waits = {}
        for chunk_id in (0, 1):
            setup = graph.call_function(torch.ops.aten.clone.default, args=(x,))
            pre = graph.call_function(torch.ops.aten.relu.default, args=(setup,))
            dispatch = graph.call_function(
                c10d.all_to_all_single.default,
                args=(pre, [], [], "ep"),
            )
            dispatch_wait = graph.call_function(
                c10d.wait_tensor.default, args=(dispatch,)
            )
            compute = graph.call_function(
                torch.ops.aten.neg.default, args=(dispatch_wait,)
            )
            combine = graph.call_function(
                c10d.all_to_all_single.default,
                args=(compute, [], [], "ep"),
            )
            combine_wait = graph.call_function(
                c10d.wait_tensor.default, args=(combine,)
            )
            outputs.append(combine_wait)
            setup_nodes[chunk_id] = setup
            first_waits[chunk_id] = dispatch_wait

            setup.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
            for node in (pre, dispatch, dispatch_wait):
                node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "dispatch"}
            dispatch.meta["custom"][_EP_TOKEN_EXCHANGE] = "dispatch"
            compute.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
            for node in (combine, combine_wait):
                node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "combine"}
            combine.meta["custom"][_EP_TOKEN_EXCHANGE] = "combine"
            for node in (pre, dispatch, dispatch_wait, compute, combine, combine_wait):
                node.meta["chunk_id"] = chunk_id
                node.meta["chunked_region_fqn"] = "layers.0.moe"
                node.meta["chunked_region_role"] = "body"

        graph.output(tuple(outputs))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        order = self._schedule_ep_overlap_and_order(gm)

        self.assertLess(order[setup_nodes[1]], order[first_waits[0]])
        self.assertLess(order[first_waits[0]], order[first_waits[1]])

    def test_ep_overlap_apply_schedule_interleaves_cross_region_boundaries(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        region0_start = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        region1_start = graph.call_function(
            torch.ops.aten.neg.default, args=(region0_start,)
        )
        boundary = graph.call_function(
            torch.ops.aten.clone.default, args=(region1_start,)
        )
        region0_tail = graph.call_function(
            torch.ops.aten.relu.default, args=(boundary,)
        )
        region1_tail = graph.call_function(
            torch.ops.aten.add.Tensor, args=(region1_start, region0_tail)
        )
        graph.output(region1_tail)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        def scheduled_region(
            root_fqn: str,
            phases: tuple[tuple[torch.fx.Node, ...], ...],
        ) -> _ScheduledRegion:
            bodies = {
                idx: ChunkBody(
                    owner=ChunkOwner(root_fqn, False, idx),
                    nodes=(),
                    node_set=frozenset(),
                    live_ins=frozenset(),
                    producer="graph",
                )
                for idx in (0, 1)
            }
            return _ScheduledRegion(
                region=ChunkedRegion(root_fqn, False, bodies),
                phases=phases,
            )

        _apply_schedule(
            gm,
            [
                scheduled_region("layers.0", ((region0_start,), (region0_tail,))),
                scheduled_region("layers.1", ((region1_start,), (region1_tail,))),
            ],
        )

        order = {node: idx for idx, node in enumerate(gm.graph.nodes)}
        self.assertLess(order[region0_start], order[region0_tail])
        self.assertLess(order[region1_start], order[region1_tail])
        self.assertLess(order[region1_start], order[boundary])
        self.assertLess(order[boundary], order[region0_tail])
        self.assertLess(order[region0_tail], order[region1_tail])

    def test_ep_overlap_moves_owned_remat_with_chunk_body(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c10d = torch.ops._c10d_functional

        c0_pre = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        c0_dispatch = graph.call_function(
            c10d.all_to_all_single.default, args=(c0_pre, [], [], "ep")
        )
        c0_dispatch_wait = graph.call_function(
            c10d.wait_tensor.default, args=(c0_dispatch,)
        )
        c0_compute = graph.call_function(
            torch.ops.aten.neg.default, args=(c0_dispatch_wait,)
        )
        c0_combine = graph.call_function(
            c10d.all_to_all_single.default, args=(c0_compute, [], [], "ep")
        )
        c0_combine_wait = graph.call_function(
            c10d.wait_tensor.default, args=(c0_combine,)
        )

        c1_setup = graph.call_function(torch.ops.aten.clone.default, args=(x,))
        c1_setup.meta["autograd_backward"] = True
        c1_setup.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        c1_pre = graph.call_function(torch.ops.aten.relu.default, args=(c1_setup,))
        c1_dispatch = graph.call_function(
            c10d.all_to_all_single.default, args=(c1_pre, [], [], "ep")
        )
        c1_dispatch_wait = graph.call_function(
            c10d.wait_tensor.default, args=(c1_dispatch,)
        )
        c1_compute = graph.call_function(
            torch.ops.aten.neg.default, args=(c1_dispatch_wait,)
        )
        c1_combine = graph.call_function(
            c10d.all_to_all_single.default, args=(c1_compute, [], [], "ep")
        )
        c1_combine_wait = graph.call_function(
            c10d.wait_tensor.default, args=(c1_combine,)
        )
        graph.output((c0_combine_wait, c1_combine_wait))

        for chunk_id, nodes in {
            0: (
                c0_pre,
                c0_dispatch,
                c0_dispatch_wait,
                c0_compute,
                c0_combine,
                c0_combine_wait,
            ),
            1: (
                c1_setup,
                c1_pre,
                c1_dispatch,
                c1_dispatch_wait,
                c1_compute,
                c1_combine,
                c1_combine_wait,
            ),
        }.items():
            for node in nodes:
                node.meta["chunk_id"] = chunk_id
                node.meta["chunked_region_fqn"] = "layers.0.moe"
                node.meta["chunked_region_role"] = "body"
                node.meta["autograd_backward"] = True
            ep_nodes = nodes[1:] if chunk_id == 1 else nodes
            for node in ep_nodes[:3]:
                node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "combine"}
            ep_nodes[1].meta["custom"][_EP_TOKEN_EXCHANGE] = "combine"
            ep_nodes[3].meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
            for node in ep_nodes[4:]:
                node.meta["custom"] = {_MODULE_FQN: "layers.0.moe", "EP": "dispatch"}
            ep_nodes[4].meta["custom"][_EP_TOKEN_EXCHANGE] = "dispatch"

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        order = self._schedule_ep_overlap_and_order(gm)
        self.assertLess(order[c1_setup], order[c1_pre])
        self.assertLess(order[c1_dispatch], order[c0_dispatch_wait])

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

        with self.assertRaisesRegex(ValueError, "No EP token-exchange regions"):
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
            require_token_exchange=False,
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

    def test_normalize_chunked_grad_collective_chains_dedups_eager_fsdp(self):
        for fsdp_mode in ("replicate", "fully_shard"):
            with self.subTest(fsdp_mode=fsdp_mode):
                gm, target = self._trace_simple_fsdp_moe_grad_collective(
                    chunk_strategy="eager",
                    fsdp_mode=fsdp_mode,
                )
                self.assertEqual(len(self._nodes_by_target(gm, target)), 2)

                normalize_chunked_grad_collective_chains_pass(gm)

                collective_nodes = self._nodes_by_target(gm, target)
                sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
                self.assertEqual(len(collective_nodes), 1)
                self.assertEqual(len(sum_nodes), 1)
                self.assertTrue(
                    self._node_depends_on(collective_nodes[0].args[0], sum_nodes[0])
                )
                self.assertNotIn("chunk_id", sum_nodes[0].meta)
                self.assertNotIn("chunk_id", sum_nodes[0].meta.get("custom", {}))

    def test_normalize_chunked_grad_collective_chains_keeps_graph_fsdp_canonical(self):
        for fsdp_mode in ("replicate", "fully_shard"):
            with self.subTest(fsdp_mode=fsdp_mode):
                gm, target = self._trace_simple_fsdp_moe_grad_collective(
                    chunk_strategy="graph",
                    fsdp_mode=fsdp_mode,
                )
                self.assertEqual(len(self._nodes_by_target(gm, target)), 1)

                normalize_chunked_grad_collective_chains_pass(gm)

                collective_nodes = self._nodes_by_target(gm, target)
                sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
                self.assertEqual(len(collective_nodes), 1)
                self.assertEqual(len(sum_nodes), 1)
                self.assertTrue(
                    self._node_depends_on(collective_nodes[0].args[0], sum_nodes[0])
                )
                self.assertNotIn("chunk_id", sum_nodes[0].meta)
                self.assertNotIn("chunk_id", sum_nodes[0].meta.get("custom", {}))

    def test_normalize_chunked_grad_collective_chains_replays_output_cast(self):
        target = torch.ops._c10d_functional.reduce_scatter_tensor.default
        for producer in ("eager", "graph"):
            with self.subTest(producer=producer):
                gm = self._build_chunked_grad_collective_with_output_cast(
                    producer=producer
                )
                self.assertEqual(len(self._nodes_by_target(gm, target)), 2)

                normalize_chunked_grad_collective_chains_pass(gm)

                collective_nodes = self._nodes_by_target(gm, target)
                sum_nodes = self._nodes_by_target(gm, torch.ops.aten.add.Tensor)
                self.assertEqual(len(collective_nodes), 1)
                self.assertEqual(len(sum_nodes), 1)
                self.assertIs(collective_nodes[0].args[0], sum_nodes[0])
                self.assertEqual(sum_nodes[0].meta["val"].dtype, torch.float32)
                output = next(node for node in gm.graph.nodes if node.op == "output")
                grad_output = output.args[0][1]
                self.assertEqual(grad_output.target, torch.ops.aten._to_copy.default)
                self.assertEqual(grad_output.meta["val"].dtype, torch.bfloat16)

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

    def test_loss_region_excluded_from_remat(self):
        # Two disjoint backward regions, each consuming a must_recompute forward
        # node: one is a chunked-loss region (module_fqn 'loss') and one is a
        # model-layer AC region (module_fqn 'layers.0.*'). Only the model-layer
        # region drives remat; the loss region is left untouched (no error).
        graph = torch.fx.Graph()
        inp1 = graph.placeholder("inp1")
        inp2 = graph.placeholder("inp2")
        a = graph.call_function(torch.ops.aten.clone.default, args=(inp1,))
        b = graph.call_function(torch.ops.aten.clone.default, args=(inp2,))
        bwd_loss = graph.call_function(torch.ops.aten.add.Tensor, args=(a, a))
        sep = graph.call_function(torch.ops.aten.neg.default, args=(inp1,))
        bwd_layer = graph.call_function(torch.ops.aten.mul.Tensor, args=(b, b))
        graph.output((bwd_loss, sep, bwd_layer))
        for node in (a, b):
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        bwd_loss.meta["autograd_backward"] = True
        bwd_loss.meta["custom"] = {"module_fqn": "loss"}
        bwd_layer.meta["autograd_backward"] = True
        bwd_layer.meta["custom"] = {"module_fqn": "layers.0.attention"}
        a_name, b_name = a.name, b.name

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        result = selective_activation_remat_pass(gm)

        nodes = list(result.graph.nodes)
        node_names = [n.name for n in nodes]
        dups = {n.name for n in nodes if n.name.endswith("_recomputed")}
        # Only the model-layer region's input is rematerialized.
        self.assertEqual(dups, {b_name + "_recomputed"})
        for inp in bwd_layer.all_input_nodes:
            self.assertEqual(inp.name, b_name + "_recomputed")
        self.assertNotIn(b_name, node_names)
        # The loss region is untouched: its recompute input stays and is still
        # read directly (not remat'd, not erased).
        self.assertIn(a_name, node_names)
        for inp in bwd_loss.all_input_nodes:
            self.assertEqual(inp.name, a_name)

    def test_multiple_model_layer_regions_recompute_errors(self):
        # Two disjoint *model-layer* backward regions that both need remat is
        # still unsupported and must error (remat handles a single such region).
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
        bwd1.meta["autograd_backward"] = True
        bwd1.meta["custom"] = {"module_fqn": "layers.0.attention"}
        bwd2.meta["autograd_backward"] = True
        bwd2.meta["custom"] = {"module_fqn": "layers.1.attention"}

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
