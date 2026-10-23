# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import torch
from graph_trainer.cuda_graph import insert_kernel_annotations_pass
from graph_trainer.ep_process_group_pass import isolate_ep_process_group_pass
from graph_trainer.fsdp_passes import (
    joint_transformer_block_bucketing_reordering_pass,
    reassign_collective_pgs_pass,
    reorder_hsdp_grad_collectives_pass,
)
from graph_trainer.make_fx_tracer import GraphStateSpec, minimal_fx_tracer, run_traced
from graph_trainer.remove_noop_passes import eliminate_dead_code_pass
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase
from torch.utils.checkpoint import checkpoint

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.common_utils import (
    _EP_TOKEN_COUNT_EXCHANGE,
    _EP_TOKEN_COUNT_SYNC,
    _EP_TOKEN_EXCHANGE,
    _MODULE_FQN,
    annotate_module_fqns,
    annotate_moe_ep_regions,
    apply_simple_fsdp,
    get_default_transformer_block_buckets,
    get_simple_fsdp_mesh,
)
from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapConfig,
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.ep_eager_chunk import (
    maybe_apply_ep_overlap_eager_chunking,
    populate_eager_chunk_metadata_pass,
)
from torchtitan.experiments.graph_trainer.passes import compile_time_passes
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    FSDP_PARAM_FQNS_META,
)
from torchtitan.models.base.module import Module, ModuleList
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE


@torch.library.custom_op(
    "torchtitan_graph_trainer_test::ordered_identity", mutates_args=()
)
def _ordered_identity(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_ordered_identity.register_fake
def _ordered_identity_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


_ordered_identity.register_effect(torch.library.EffectType.ORDERED)


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
                model=SimpleNamespace(
                    layers=[SimpleNamespace(feed_forward=None) for _ in range(2)],
                    mtp_layers=[object()],
                ),
                parallelism=SimpleNamespace(),
                training=SimpleNamespace(disable_cuda_graphs=False),
            )

        traced_result = SimpleNamespace(
            state_fqns=[],
            graph_state=GraphStateSpec(),
        )
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
        self.assertEqual(
            [call.kwargs["num_mtp_layers"] for call in mock_bucket_plan.call_args_list],
            [1, 1],
        )

    def test_hsdp_collective_reordering_precedes_bucketing(self) -> None:
        config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                disable_passes=["cuda_graph_pass"],
                inductor_compilation="full",
            ),
            loss=None,
            model=SimpleNamespace(layers=[SimpleNamespace(feed_forward=None)]),
            parallelism=SimpleNamespace(
                data_parallel_replicate_degree=2,
                data_parallel_shard_degree=2,
                fsdp_reshard_after_forward="always",
                pipeline_parallel_degree=1,
            ),
            training=TrainingConfig(disable_cuda_graphs=True),
        )
        traced_result = SimpleNamespace(state_fqns=[])

        def make_mesh(axis, degree):
            return SimpleNamespace(
                size=lambda: degree,
                get_group=lambda: SimpleNamespace(group_name=axis),
            )

        def get_mesh(axis):
            return make_mesh(axis, 2)

        def get_optional_mesh(axis, *, include_singleton_axes=False):
            if axis == ["dp_shard", "cp"]:
                return SimpleNamespace(
                    _flatten=lambda name: make_mesh(name, 2),
                )
            if axis == ["dp_replicate", "edp_shard"]:
                return {
                    "dp_replicate": make_mesh("expert_dp_replicate", 2),
                    "edp_shard": make_mesh("edp_shard", 2),
                }
            return None

        parallelism_context = SimpleNamespace(
            dp_replicate=2,
            dp_shard=2,
            get_mesh=get_mesh,
            get_optional_mesh=get_optional_mesh,
        )
        graph_passes = compile_time_passes(
            traced_result,
            config,
            parallelism_context=parallelism_context,
        )
        pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

        self.assertLess(
            pass_fns.index(reorder_hsdp_grad_collectives_pass),
            pass_fns.index(joint_transformer_block_bucketing_reordering_pass),
        )
        hsdp_pass = next(
            graph_pass
            for graph_pass in graph_passes
            if getattr(graph_pass, "func", None) is reorder_hsdp_grad_collectives_pass
        )
        self.assertEqual(
            hsdp_pass.keywords["hsdp_process_group_pairs"],
            (
                ("dp_replicate", 2, "fsdp"),
                ("expert_dp_replicate", 2, "edp_shard"),
            ),
        )

    def test_compile_time_passes_follow_core_cuda_graph_config(self):
        from graph_trainer.cuda_graph import cuda_graph_pass

        from torchtitan.components.loss import CrossEntropyLoss

        def make_config(*, disable_cuda_graphs):
            return SimpleNamespace(
                compile=GraphTrainerCompileConfig(inductor_compilation="regional"),
                loss=CrossEntropyLoss.Config(),
                model=SimpleNamespace(
                    layers=[SimpleNamespace(feed_forward=None) for _ in range(2)]
                ),
                parallelism=SimpleNamespace(),
                training=SimpleNamespace(disable_cuda_graphs=disable_cuda_graphs),
            )

        traced_result = SimpleNamespace(
            state_fqns=[],
            graph_state=GraphStateSpec(),
        )
        with patch(
            "torchtitan.experiments.graph_trainer.common_utils."
            "get_default_transformer_block_buckets",
            return_value=[],
        ):
            with_cuda_graph = compile_time_passes(
                traced_result,
                make_config(disable_cuda_graphs=False),
            )
            without_cuda_graph = compile_time_passes(
                traced_result,
                make_config(disable_cuda_graphs=True),
            )

        self.assertEqual(with_cuda_graph.count(insert_kernel_annotations_pass), 1)
        self.assertNotIn(insert_kernel_annotations_pass, without_cuda_graph)
        self.assertNotIn(cuda_graph_pass, with_cuda_graph)


class ToyModel(Module):
    """A small toy model with multiple linear layers and activation
    checkpointing so that the backward graph recomputes the forward
    all-gathers."""

    def __init__(self, dim=16, n_layers=3):
        super().__init__()

        def _make_linear():
            # Traced under SimpleFSDP: no in-place WGRAD accumulation.
            cfg = Linear.Config(
                in_features=dim,
                out_features=dim,
                bias=True,
                inplace_wgrad_accum=False,
            )
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


class TestHSDPGradCollectiveReorderingPass(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_reorders_real_simple_fsdp_joint_trace(self) -> None:
        parallelism_context = ParallelismContext(
            dp_shard=2,
            dp_replicate=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        model = ToyModel(dim=16, n_layers=2).cuda()
        model = apply_simple_fsdp(
            model,
            parallelism_context=parallelism_context,
            training=TrainingConfig(mixed_precision_param="float32"),
        )
        torch.manual_seed(42 + self.rank)
        inputs = torch.randn(4, 16, device="cuda")

        def fwd_bwd(inp):
            loss = model(inp).sum()
            return list(torch.autograd.grad(loss, tuple(model.parameters())))

        traced = minimal_fx_tracer(fwd_bwd, module=model)(inputs)
        c10d = torch.ops._c10d_functional
        all_reduces = [
            node
            for node in traced.gm.graph.nodes
            if node.op == "call_function" and node.target is c10d.all_reduce.default
        ]
        reduce_scatters = [
            node
            for node in traced.gm.graph.nodes
            if node.op == "call_function"
            and node.target is c10d.reduce_scatter_tensor.default
        ]
        self.assertGreater(len(all_reduces), 0)
        self.assertEqual(len(all_reduces), len(reduce_scatters))
        self.assertTrue(
            all(
                node.meta.get("custom", {}).get(FSDP_PARAM_FQNS_META)
                for node in all_reduces
            )
        )
        self.assertTrue(
            all(
                next(iter(all_reduce.users)).args[0] is all_reduce
                for all_reduce in all_reduces
            )
        )

        expected = fwd_bwd(inputs)
        reorder_hsdp_grad_collectives_pass(
            traced.gm,
            traced.example_inputs,
            hsdp_process_group_pairs=(
                (
                    parallelism_context.get_mesh("dp_replicate").get_group().group_name,
                    get_simple_fsdp_mesh(parallelism_context).size(),
                    get_simple_fsdp_mesh(parallelism_context).get_group().group_name,
                ),
            ),
        )
        actual = run_traced(traced, module=model)(inputs)

        for all_reduce in all_reduces:
            reduce_scatter_wait = all_reduce.args[0]
            self.assertIsInstance(reduce_scatter_wait, torch.fx.Node)
            self.assertEqual(reduce_scatter_wait.target, c10d.wait_tensor.default)
            reduce_scatter = reduce_scatter_wait.args[0]
            self.assertIsInstance(reduce_scatter, torch.fx.Node)
            self.assertEqual(
                reduce_scatter.target,
                c10d.reduce_scatter_tensor.default,
            )
            self.assertEqual(
                all_reduce.meta["val"].shape,
                reduce_scatter.meta["val"].shape,
            )
        for expected_grad, actual_grad in zip(expected, actual, strict=True):
            torch.testing.assert_close(
                expected_grad.to_local(),
                actual_grad.to_local(),
            )

        for node in traced.gm.graph.nodes:
            if node.target in (
                c10d.all_reduce.default,
                c10d.reduce_scatter_tensor.default,
            ):
                node.meta.setdefault("custom", {})[_MODULE_FQN] = "layers.0"
        joint_transformer_block_bucketing_reordering_pass(
            traced.gm,
            module_bucket_plans=["layers.0"],
        )
        bucketed_all_reduces = traced.gm.graph.find_nodes(
            op="call_function", target=c10d.all_reduce.default
        )
        bucketed_reduce_scatters = traced.gm.graph.find_nodes(
            op="call_function", target=c10d.reduce_scatter_tensor.default
        )
        self.assertEqual(len(bucketed_all_reduces), 1)
        self.assertEqual(len(bucketed_reduce_scatters), 1)
        self.assertTrue(
            bucketed_all_reduces[0].meta.get("custom", {}).get(FSDP_PARAM_FQNS_META)
        )
        (bucketed_all_reduce_wait,) = tuple(
            node
            for node in bucketed_all_reduces[0].users
            if node.target is c10d.wait_tensor.default
        )
        self.assertTrue(
            bucketed_all_reduce_wait.meta.get("custom", {}).get(FSDP_PARAM_FQNS_META)
        )
        bucketed_actual = run_traced(traced, module=model)(inputs)
        for expected_grad, actual_grad in zip(expected, bucketed_actual, strict=True):
            torch.testing.assert_close(
                expected_grad.to_local(),
                actual_grad.to_local(),
            )
        traced.gm.graph.lint()


class TestReassignCollectivePgsPass(FSDPTest):
    """Integration tests: toy model + simple_fsdp + minimal_fx_tracer + reassign_collective_pgs_pass."""

    def _setup(self):
        """Set up ParallelismContext and device mesh for FSDP."""
        self.parallelism_context = ParallelismContext(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )

    def _make_fsdp_model(self, dim=16, n_layers=3):
        """Create a toy model and apply simple_fsdp data_parallel."""
        model = ToyModel(dim, n_layers).cuda()
        from torchtitan.experiments.graph_trainer.common_utils import (
            get_simple_fsdp_mesh,
        )

        fsdp_mesh = get_simple_fsdp_mesh(self.parallelism_context)
        model = data_parallel(model, device_mesh=fsdp_mesh, mode="fully_shard")
        return model

    def _get_fsdp_pg_name(self):
        """Get the FSDP process group name from the mesh."""
        from torchtitan.experiments.graph_trainer.common_utils import (
            get_simple_fsdp_mesh,
        )

        fsdp_mesh = get_simple_fsdp_mesh(self.parallelism_context)
        return fsdp_mesh.get_group().group_name

    def _trace_joint_graph(self, model, inputs):
        """Trace the joint fwd+bwd graph with minimal_fx_tracer."""

        def fwd_bwd_step(x):
            loss = model(x).sum()
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params)
            return [loss, *grads]

        traced = minimal_fx_tracer(fwd_bwd_step, module=model)(inputs)
        return traced.gm, traced.example_inputs

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
        """Apply reassign_collective_pgs_pass on the traced joint graph and verify
        that FSDP AG nodes are rewritten to the auto-created extra PG."""
        from graph_trainer.fsdp_passes import _EXTRA_FSDP_PG_REGISTRY

        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        gm, example_inputs = self._trace_joint_graph(model, inputs)

        # Before: all AG nodes should use the FSDP PG
        ag_before = self._count_ag_nodes_with_pg(gm, fsdp_pg_name)
        self.assertGreater(ag_before, 0, "Expected AG nodes with FSDP PG name")

        _EXTRA_FSDP_PG_REGISTRY.pop(fsdp_pg_name, None)
        reassign_collective_pgs_pass(gm, example_inputs)

        extra_pg_name = _EXTRA_FSDP_PG_REGISTRY[fsdp_pg_name]
        ag_with_old = self._count_ag_nodes_with_pg(gm, fsdp_pg_name)
        ag_with_new = self._count_ag_nodes_with_pg(gm, extra_pg_name)

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

        gm, example_inputs = self._trace_joint_graph(model, inputs)

        total_before = self._count_all_ag_nodes(gm)
        reassign_collective_pgs_pass(gm, example_inputs)
        total_after = self._count_all_ag_nodes(gm)

        self.assertEqual(total_before, total_after)

    def test_overlap_rewrites_multiple_pgs(self):
        """When the graph has AG nodes from multiple FSDP PGs (e.g. FSDP +
        expert-FSDP), each source PG should be mapped to its own extra PG."""
        import torch.distributed as dist
        from graph_trainer.fsdp_passes import _EXTRA_FSDP_PG_REGISTRY

        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        gm, example_inputs = self._trace_joint_graph(model, inputs)

        # Create a second PG to simulate expert-FSDP.
        second_pg = dist.new_group(
            ranks=list(range(self.world_size)),
            use_local_synchronization=True,
        )
        second_pg_name = second_pg.group_name

        # Rewrite half the AG nodes to use the second PG.
        ag_nodes = [n for n in gm.graph.nodes if is_all_gather(n)]
        self.assertGreater(len(ag_nodes), 1)
        half = len(ag_nodes) // 2
        for node in ag_nodes[:half]:
            node.args = (node.args[0], node.args[1], second_pg_name)

        ag_pg1_before = self._count_ag_nodes_with_pg(gm, fsdp_pg_name)
        ag_pg2_before = self._count_ag_nodes_with_pg(gm, second_pg_name)
        self.assertGreater(ag_pg1_before, 0)
        self.assertGreater(ag_pg2_before, 0)

        _EXTRA_FSDP_PG_REGISTRY.pop(fsdp_pg_name, None)
        _EXTRA_FSDP_PG_REGISTRY.pop(second_pg_name, None)
        reassign_collective_pgs_pass(gm, example_inputs)

        # Both source PGs should have their own extra PG.
        self.assertIn(fsdp_pg_name, _EXTRA_FSDP_PG_REGISTRY)
        self.assertIn(second_pg_name, _EXTRA_FSDP_PG_REGISTRY)
        extra_pg1 = _EXTRA_FSDP_PG_REGISTRY[fsdp_pg_name]
        extra_pg2 = _EXTRA_FSDP_PG_REGISTRY[second_pg_name]
        self.assertNotEqual(
            extra_pg1, extra_pg2, "Each source PG must map to a distinct extra PG"
        )

        # No AG nodes should still use original PGs.
        self.assertEqual(self._count_ag_nodes_with_pg(gm, fsdp_pg_name), 0)
        self.assertEqual(self._count_ag_nodes_with_pg(gm, second_pg_name), 0)

        # All AG nodes should use their respective extra PGs.
        self.assertEqual(self._count_ag_nodes_with_pg(gm, extra_pg1), ag_pg1_before)
        self.assertEqual(self._count_ag_nodes_with_pg(gm, extra_pg2), ag_pg2_before)

    def test_overlap_rewrites_ep_a2a_on_fsdp_pg_to_separate_pg(self):
        from graph_trainer.ep_process_group_pass import _EXTRA_EP_PG_REGISTRY
        from graph_trainer.fsdp_passes import _EXTRA_FSDP_PG_REGISTRY

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
            _MODULE_FQN: "layers.0.feed_forward",
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
        from torchtitan.experiments.graph_trainer.common_utils import (
            annotate_graph_trainer_model,
        )
        from torchtitan.experiments.graph_trainer.llama3 import (
            build_model_config as build_llama3_model_config,
        )
        from torchtitan.experiments.graph_trainer.simple_fsdp import (
            data_parallel,
            MixedPrecisionPolicy,
        )
        from torchtitan.experiments.graph_trainer.tests._trainer_test_utils import (
            build_minimal_trainer,
        )
        from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

        parallelism_context = ParallelismContext(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )

        model_config = build_llama3_model_config(
            "debugmodel", parallelism=ParallelismConfig()
        )
        vocab_size = model_config.vocab_size

        with torch.device("meta"):
            model = model_config.build()

        annotate_graph_trainer_model(model)
        from torchtitan.experiments.graph_trainer.common_utils import (
            get_simple_fsdp_mesh,
        )

        fsdp_mesh = get_simple_fsdp_mesh(parallelism_context)
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
            parallelism_context=parallelism_context,
        )

        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        inputs = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
        labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
        # The dataloader supplies per-document positions, which the trainer
        # requires to build the block-causal FlexInnerAttention mask.
        positions = torch.arange(self.SEQ_LEN, device="cuda", dtype=torch.int32).repeat(
            self.BATCH_SIZE
        )
        global_loss_token_counts = torch.tensor(
            num_tokens, dtype=torch.float, device="cuda"
        )

        # One accumulation step traces the model and applies all graph passes.
        trainer.engine.forward_backward(
            microbatch_groups=[
                [
                    TokenizedTrainingMicrobatch(
                        input=inputs,
                        positions=positions,
                        labels=labels,
                        padding_mask=torch.zeros_like(labels, dtype=torch.bool),
                        loss_token_counts=torch.tensor(labels.numel()),
                        routing_token_counts=torch.tensor([labels.numel()]),
                    )
                ]
            ],
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_loss_token_counts.unsqueeze(0),
        )

        layer_ids = self._get_bucketed_ag_layer_order(trainer.engine._traced_step.gm)
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
    def _compile_config_for_ep_overlap_test(self):
        from types import SimpleNamespace

        traced_result = SimpleNamespace(
            num_static_inputs=2,
            state_fqns=[],
            graph_state=GraphStateSpec(),
        )
        config = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(feed_forward=None)]),
            parallelism=SimpleNamespace(
                expert_parallel_degree=1,
                fsdp_reshard_after_forward="default",
                pipeline_parallel_degree=1,
            ),
            training=SimpleNamespace(disable_cuda_graphs=False),
            compile=GraphTrainerCompileConfig(
                ep_overlap=EpOverlapConfig(
                    enabled=True,
                    chunk_dim="batch",
                    module_fqn="layers.*",
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
            pass_name(pass_fn) for pass_fn in compile_time_passes(traced_result, config)
        ]

    def test_ep_overlap_pass_pipeline_order(self):
        traced_result, config = self._compile_config_for_ep_overlap_test()
        names = self._compile_pass_names(traced_result, config)
        dead_code_indices = [
            i for i, name in enumerate(names) if name == "eliminate_dead_code_pass"
        ]
        populate_idx = names.index("populate_eager_chunk_metadata_pass")
        post_chunk_dce = min(i for i in dead_code_indices if i > populate_idx)

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
        self.assertLess(names.index("selective_activation_remat_pass"), populate_idx)
        self.assertLess(populate_idx, names.index("isolate_ep_process_group_pass"))
        self.assertLess(
            names.index("isolate_ep_process_group_pass"),
            post_chunk_dce,
        )
        self.assertLess(
            post_chunk_dce,
            names.index("joint_transformer_block_bucketing_reordering_pass"),
        )
        self.assertLess(
            names.index("joint_transformer_block_bucketing_reordering_pass"),
            names.index("ep_overlap_schedule_pass"),
        )
        self.assertLess(
            names.index("ep_overlap_schedule_pass"),
            names.index("full_inductor_compilation_pass"),
        )

    def test_fsdp_dense_region_scheduler_pass_gating(self):
        def transformer_batch_default(config):
            pass

        def transformer_batch_explicit(config):
            config.compile.enable_fsdp_dense_region_overlap = True

        def moe_ep_default(config):
            config.compile.ep_overlap.module_fqn = "layers.*.feed_forward"

        def moe_ep_explicit(config):
            config.compile.ep_overlap.module_fqn = "layers.*.feed_forward"
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
                "ignored when compile.ep_overlap.enabled is set",
            ),
            ("moe_ep_default", moe_ep_default, True, False, None),
            (
                "moe_ep_explicit",
                moe_ep_explicit,
                True,
                False,
                "ignored when compile.ep_overlap.enabled is set",
            ),
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

    def test_moe_edp_shard_bucket_plan_splits_expert_buckets(self):
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
                "layers.1.feed_forward.router",
                "layers.1.feed_forward.shared_experts",
            ],
            buckets,
        )
        self.assertIn("layers.1.feed_forward.routed_experts", buckets)
        self.assertNotIn("layers.1", buckets)

    def test_mtp_layers_use_parent_scope_buckets(self):
        buckets = get_default_transformer_block_buckets(
            2,
            num_mtp_layers=2,
        )

        self.assertEqual(
            buckets,
            [
                "tok_embeddings",
                "layers.0",
                "layers.1",
                "mtp_layers.0",
                "mtp_layers.1",
                ["norm", "lm_head"],
            ],
        )

    def test_moe_ep_annotations_cover_all_to_all_dispatcher(self):
        from torchtitan.models.common.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

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
                ep_overlap=EpOverlapConfig(
                    enabled=True,
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


class TestEagerChunking(TestCase):
    def _config(
        self,
        *,
        chunk_dim: str = "batch",
        module_fqn: str = "layers.*",
    ) -> GraphTrainerCompileConfig:
        return GraphTrainerCompileConfig(
            ep_overlap=EpOverlapConfig(
                enabled=True,
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
        config.ep_overlap.enabled = False

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

    def test_transformer_batch_chunking_splits_token_metadata_by_batch(self):
        seen_positions = []
        seen_padding_masks = []

        class Block(torch.nn.Module):
            def forward(
                self,
                x,
                attention_metadata=None,
                positions=None,
                *,
                padding_mask=None,
            ):
                seen_positions.append(positions)
                seen_padding_masks.append(padding_mask)
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x, positions, padding_mask):
                return self.layers[0](x, None, positions, padding_mask=padding_mask)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(model, self._config())
        x = torch.randn(4, 4, 2)
        positions = torch.arange(16).view(4, 4)
        padding_mask = torch.tensor(
            [
                [False, False, False, True],
                [False, False, True, True],
                [False, False, False, False],
                [False, True, True, True],
            ]
        )

        self.assertEqual(model(x, positions, padding_mask), x)
        self.assertEqual([tuple(pos.shape) for pos in seen_positions], [(2, 4), (2, 4)])
        self.assertEqual(seen_positions[0], positions[:2])
        self.assertEqual(seen_positions[1], positions[2:])
        self.assertEqual(seen_padding_masks[0], padding_mask[:2])
        self.assertEqual(seen_padding_masks[1], padding_mask[2:])

    def test_transformer_batch_chunking_rejects_same_extent_tensor_mask(self):
        class Block(torch.nn.Module):
            def forward(self, x, attention_metadata):
                return x + attention_metadata

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x, attention_metadata):
                return self.layers[0](x, attention_metadata)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(model, self._config())

        with self.assertRaisesRegex(
            ValueError,
            "attention_metadata must be None, BlockMask.*upstream .*TransformerBlock",
        ):
            model(torch.randn(4, 3), torch.randn(4, 3))

    def test_moe_chunking_splits_activation_only(self):
        seen_shapes = []

        class Moe(MoE):
            def __init__(self):
                torch.nn.Module.__init__(self)

            def forward(self, x):
                seen_shapes.append(tuple(x.shape))
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].feed_forward = Moe()

            def forward(self, x):
                return self.layers[0].feed_forward(x)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(
            model,
            self._config(chunk_dim="seq", module_fqn="layers.*.feed_forward"),
        )
        x = torch.randn(8, 3)

        self.assertEqual(model(x), x)
        self.assertEqual(seen_shapes, [(4, 3), (4, 3)])

    def test_moe_chunking_splits_padding_mask_with_activation(self):
        seen_inputs = []

        class Moe(MoE):
            def __init__(self):
                torch.nn.Module.__init__(self)

            def forward(self, x, *, padding_mask_T=None):
                seen_inputs.append((x, padding_mask_T))
                return x.masked_fill(padding_mask_T.unsqueeze(-1), 0)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].feed_forward = Moe()

            def forward(self, x, padding_mask_T):
                return self.layers[0].feed_forward(x, padding_mask_T=padding_mask_T)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(
            model,
            self._config(chunk_dim="seq", module_fqn="layers.*.feed_forward"),
        )
        x = torch.randn(8, 3)
        padding_mask_T = torch.tensor(
            [False, True, False, True, True, False, True, False]
        )

        self.assertEqual(
            model(x, padding_mask_T),
            x.masked_fill(padding_mask_T.unsqueeze(-1), 0),
        )
        self.assertEqual(
            [tuple(chunk_x.shape) for chunk_x, _ in seen_inputs],
            [(4, 3), (4, 3)],
        )
        self.assertEqual(seen_inputs[0][1], padding_mask_T[:4])
        self.assertEqual(seen_inputs[1][1], padding_mask_T[4:])

    def test_moe_chunking_shares_aux_loss_denominator(self):
        seen_denominators = []

        class Moe(MoE):
            def __init__(self):
                torch.nn.Module.__init__(self)

            def forward(self, x, *, aux_loss_denominator):
                seen_denominators.append(aux_loss_denominator)
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].feed_forward = Moe()

            def forward(self, x, aux_loss_denominator):
                return self.layers[0].feed_forward(
                    x, aux_loss_denominator=aux_loss_denominator
                )

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(
            model,
            self._config(chunk_dim="seq", module_fqn="layers.*.feed_forward"),
        )
        x = torch.randn(8, 3)
        aux_loss_denominator = torch.tensor(8)

        self.assertEqual(model(x, aux_loss_denominator), x)
        self.assertEqual(seen_denominators, [aux_loss_denominator] * 2)

    def test_moe_chunking_rejects_extra_tensor_input(self):
        class Moe(MoE):
            def __init__(self):
                torch.nn.Module.__init__(self)

            def forward(self, x, aux):
                return x + aux

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].feed_forward = Moe()

            def forward(self, x, aux):
                return self.layers[0].feed_forward(x, aux)

        model = Model()
        maybe_apply_ep_overlap_eager_chunking(
            model,
            self._config(module_fqn="layers.*.feed_forward"),
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
            def forward(self, x, attention_metadata, positions):
                seen_masks.append(attention_metadata)
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])

            def forward(self, x, attention_metadata, positions):
                return self.layers[0](x, attention_metadata, positions)

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
