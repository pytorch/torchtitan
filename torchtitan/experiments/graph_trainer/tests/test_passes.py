# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import operator
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._decomp import get_decompositions
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch.cuda._graph_annotations import _is_tools_id_unavailable
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.traceback import preserve_node_meta
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
from graph_trainer.cudagraph import (
    insert_kernel_annotations_pass,
    is_cudagraphable,
    is_full_cudagraphable,
)
from graph_trainer.decompositions import (
    apply_decompositions_pass,
)
from graph_trainer.ep_chunk_pass import (
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
from graph_trainer.ep_eager_chunk import (
    maybe_apply_ep_overlap_eager_chunking,
    populate_eager_chunk_metadata_pass,
)
from graph_trainer.ep_overlap_pass import (
    _apply_schedule,
    _schedule_ep_overlap_regions,
    _ScheduledRegion,
)
from graph_trainer.ep_pass_utils import (
    CHUNK_SYMBOL_HINTS_META,
    ChunkBody,
    ChunkedRegion,
    ChunkOwner,
    concretize_ep_chunk_symbolic_shapes_pass,
)
from graph_trainer.ep_process_group_pass import (
    isolate_ep_process_group_pass,
)
from graph_trainer.fsdp_passes import (
    _FSDP_BUCKET_META,
    deduplicate_fsdp_unshard_chains_pass,
    get_transformer_block_bucket_counts,
    reassign_collective_pgs_pass,
    schedule_fsdp_comms_to_dense_regions_pass,
)
from graph_trainer.graph_utils import export_joint
from graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from graph_trainer.memory_policy import (
    _backward_side_nodes,
    _default_memory_policy_pass,
    _make_default_memory_policy,
    _make_full_memory_policy,
    tag_min_cut_saved_values,
    tag_sac_policy,
    tag_with_memory_policy_pass,
    validate_memory_policy_config,
)
from graph_trainer.passes import (
    compile_time_passes,
    selective_activation_remat_pass,
)
from graph_trainer.remove_noop_passes import (
    canonicalize_graph_pass,
    eliminate_dead_code_pass,
    normalize_view_ops_as_reshape,
    remove_b2b_transpose_pass,
    remove_detach_pass,
    remove_identity_slice_pass,
    remove_identity_view_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from graph_trainer.subgraph_regions import (
    apply_subgraph_region_annotations_pass,
    SUBGRAPH_REGION,
    SUBGRAPH_REGION_ROLE,
)
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleList


class TestDefaultTransformerBlockBuckets(TestCase):
    def test_compile_time_passes_enable_chunked_loss_bucket_only_when_needed(self):
        from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from graph_trainer.passes import compile_time_passes

        def make_config(loss):
            return SimpleNamespace(
                compile=GraphTrainerCompileConfig(inductor_compilation="full"),
                loss=loss,
                model_spec=SimpleNamespace(model=SimpleNamespace(layers=[0, 1])),
                parallelism=SimpleNamespace(),
            )

        traced_result = SimpleNamespace(state_fqns=[])
        with patch(
            "graph_trainer.common_utils."
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
            spmd_backend="partial_dtensor",
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
        from graph_trainer.fsdp_passes import (
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

        from graph_trainer.fsdp_passes import (
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
        from graph_trainer.ep_process_group_pass import (
            _EXTRA_EP_PG_REGISTRY,
        )
        from graph_trainer.fsdp_passes import (
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
            spmd_backend="partial_dtensor",
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

        from graph_trainer.ep_process_group_pass import (
            _EXTRA_EP_PG_REGISTRY,
        )
        from graph_trainer.fsdp_passes import (
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
        from graph_trainer.ep_process_group_pass import (
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
            spmd_backend="partial_dtensor",
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

        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        inputs = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
        labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
        # The dataloader supplies per-document positions, which the trainer
        # requires to build the block-causal FlexAttention mask.
        positions = torch.arange(self.SEQ_LEN, device="cuda", dtype=torch.int32).repeat(
            self.BATCH_SIZE
        )
        global_valid_tokens = torch.tensor(num_tokens, dtype=torch.float, device="cuda")

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

        dim = 0
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
            require_all_to_all=True,
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



















    def _compile_config_for_ep_overlap_test(self):
        from types import SimpleNamespace

        traced_result = SimpleNamespace(num_static_inputs=2, state_fqns=[])
        config = SimpleNamespace(
            model_spec=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
            parallelism=SimpleNamespace(
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






    def test_seq_chunk_marker_traces_chunked_loss_backward(self):
        from torchtitan.components.loss import ChunkedLossWrapper

        torch.manual_seed(42)
        num_tokens, dim, vocab_size = 64, 4, 8
        lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        loss_fn = ChunkedLossWrapper(ChunkedLossWrapper.Config(num_chunks=8))
        loss_fn.lm_head = lm_head

        hidden_states = torch.randn(num_tokens, dim, requires_grad=True)
        labels = torch.randint(0, vocab_size, (num_tokens,))
        mark_chunk_dynamic_dims(hidden_states, mode="seq")
        mark_chunk_dynamic_dims(labels, mode="seq")

        traced = minimal_fx_tracer(lambda h, y: loss_fn(h, y))(hidden_states, labels)

        self.assertGreater(len(list(traced.gm.graph.nodes)), 0)



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


















class TestAsyncTensorParallelPass(FSDPTest):
    """Verify async_tensor_parallel_pass produces fused ops."""

    @property
    def world_size(self):
        return 2

    def test_ag_mm_becomes_fused_op(self):
        from torch.distributed._symmetric_memory import _test_mode

        from graph_trainer.passes import (
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

        from graph_trainer.passes import (
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










if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
