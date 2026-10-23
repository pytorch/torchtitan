# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import types
import unittest
from typing import Any, cast
from unittest import mock

import torch
import torch.utils._pytree as pytree
from graph_trainer.graph_pp import multiplex_fw_bw_graph
from graph_trainer.graph_pp.runner import (
    _grad_reduction_runs_in_backward,
    BACKWARD,
    BACKWARD_WEIGHT_WITH_REDUCE_GRAD,
    BACKWARD_WITH_REDUCE_GRAD,
    FORWARD_BACKWARD,
    FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
    FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
    FORWARD_BACKWARD_NOGRADACCUM,
    FULL_FORWARD_BACKWARD,
)
from graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced
from torch import fx, nn
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    OVERLAP_F_B,
    RESHARD,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    ensure_boxed_graph_module,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    SPMDGradientAccumulationConfig,
)
from torchtitan.experiments.graph_trainer.graph_builder import (
    GraphExecutionPlan,
    GraphTrainerStageGraphProvider,
)
from torchtitan.experiments.graph_trainer.graph_builder_utils import (
    _compile_graph_pp_module,
    _execute_graph_module,
)
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
    _make_spmd_graph_runtime,
    _make_spmd_runtime_schedule,
    _set_graph_backward_actions,
    _validate_graph_pp_config,
    make_graph_runtime,
    resolve_graph_execution_plan,
)
from torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder import (
    _build_graph_pp_overlap_graphs,
    _build_stage_graphs,
    _dist_moe_forward_slot_arguments,
    _rewrite_dist_moe_activation_slot_input,
    GraphTrainerStageGraphs,
)
from torchtitan.experiments.graph_trainer.spmd_gradient_accumulation_graph_builder import (
    _build_gradient_accumulation_fwd_bwd_graphs,
    GraphTrainerScheduledFwdBwdStageGraphs,
)
from torchtitan.experiments.graph_trainer.spmd_graph_builder import (
    _build_fwd_bwd_graphs,
    GraphTrainerJointStageGraphs,
)


def _boxed_run(gm: fx.GraphModule, args: list[object]):
    return fx.Interpreter(gm).boxed_run(args)


def _make_runtime_schedule_mock() -> mock.Mock:
    schedule = mock.Mock()

    def prepare_schedule(actions, *, format):
        schedule.pipeline_order_with_comms = {
            rank: list(rank_actions) for rank, rank_actions in actions.items()
        }

    schedule._prepare_schedule_with_comms.side_effect = prepare_schedule
    return schedule


def _make_test_spmd_plan(
    *,
    num_microbatches: int,
    fsdp_enabled: bool,
    compile_config: GraphTrainerCompileConfig | None = None,
) -> GraphExecutionPlan:
    return resolve_graph_execution_plan(
        compile_config or GraphTrainerCompileConfig(),
        num_microbatches=num_microbatches,
        parallelism=ParallelismConfig(),
        pp_enabled=False,
        fsdp_enabled=fsdp_enabled,
    )


def _make_test_pp_plan(*, fsdp_enabled: bool = True) -> GraphExecutionPlan:
    return resolve_graph_execution_plan(
        GraphTrainerCompileConfig(),
        num_microbatches=2,
        parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        pp_enabled=True,
        fsdp_enabled=fsdp_enabled,
    )


def _build_test_stage_graphs(
    stage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    compile_graphs: bool = True,
    extract_fsdp_param_unshard: bool = True,
    extract_fsdp_grad_reduction: bool = True,
    gradient_accumulation: bool = False,
    fuse_wgrad_accumulation: bool = False,
) -> None:
    _build_stage_graphs(
        stage,
        args,
        kwargs,
        target,
        loss_kwargs,
        loss_fn=stage.loss_fn,
        config=types.SimpleNamespace(
            compile=stage.compile_config,
            training=types.SimpleNamespace(disable_cuda_graphs=False),
            parallelism=None,
            model=None,
        ),
        compile_graphs=compile_graphs,
        extract_fsdp_param_unshard=extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
        gradient_accumulation=gradient_accumulation,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    )


def _make_test_stage(
    submod: nn.Module,
    *,
    is_last: bool,
    loss_fn=None,
    stage_index: int = 0,
    output_grads=None,
    compile_config: GraphTrainerCompileConfig | None = None,
    runtime_validate: bool = False,
):
    stage = types.SimpleNamespace(
        submod=submod,
        device=next(
            (parameter.device for parameter in submod.parameters()),
            torch.device("cpu"),
        ),
        is_last=is_last,
        loss_fn=loss_fn,
        stage_index=stage_index,
        compile_config=compile_config or GraphTrainerCompileConfig(enable_passes=False),
        _runtime_validate=runtime_validate,
    )
    if not is_last:
        if output_grads is None:
            raise ValueError("non-last test stages must provide output_grads metadata")
        stage._stage_meta = types.SimpleNamespace(
            outputs=pytree.tree_leaves(output_grads),
            output_grads=pytree.tree_leaves(output_grads),
        )
        stage._to_tensor = lambda meta: meta
    return stage


def _split_batch_offset_block_masks() -> list[Any]:
    from torch.distributed.pipelining.microbatch import _split_block_mask
    from torch.nn.attention.flex_attention import create_block_mask

    maybe_register_blockmask_pytree_node()

    def mask_mod(b, h, q_idx, kv_idx):
        return (b == 1) & (q_idx >= kv_idx)

    block_mask = create_block_mask(
        mask_mod,
        B=2,
        H=None,
        Q_LEN=128,
        KV_LEN=128,
        device="cpu",
    )
    return _split_block_mask(block_mask, 2)


def _trace_mask_mod_replay(mask0: Any, mask1: Any) -> tuple[bool, bool]:
    def forward(mask):
        b = torch.tensor(0)
        h = torch.tensor(0)
        q_idx = torch.tensor(1)
        kv_idx = torch.tensor(0)
        return mask.mask_mod(b, h, q_idx, kv_idx)

    traced = minimal_fx_tracer(forward)(mask0)
    replay = run_traced(traced)
    return replay(mask0).item(), replay(mask1).item()


class GraphRuntimeTraceTest(unittest.TestCase):
    def test_non_last_graph_build_does_not_run_real_pretrace_forward(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensor

        class FakeOnlyStage(nn.Module):
            def forward(self, x):
                if not isinstance(x, FakeTensor):
                    raise RuntimeError("GraphPP ran a real pre-trace forward")
                return x.sin()

        x = torch.randn(2, 4, requires_grad=True)
        stage = _make_test_stage(
            FakeOnlyStage(),
            is_last=False,
            output_grads=torch.empty(2, 4),
        )

        _build_test_stage_graphs(stage, (x,), {}, None, {})

    def test_last_graph_build_does_not_run_real_pretrace_forward_or_loss(
        self,
    ) -> None:
        from torch._subclasses.fake_tensor import FakeTensor

        class FakeOnlyStage(nn.Module):
            def forward(self, x):
                if not isinstance(x, FakeTensor):
                    raise RuntimeError("GraphPP ran a real pre-trace forward")
                return x * 2

        def loss_fn(pred, target):
            if not isinstance(pred, FakeTensor) or not isinstance(target, FakeTensor):
                raise RuntimeError("GraphPP ran a real pre-trace loss")
            return ((pred - target) ** 2).sum()

        x = torch.randn(2, 4, requires_grad=True)
        target = torch.randn(2, 4)
        stage = _make_test_stage(FakeOnlyStage(), is_last=True, loss_fn=loss_fn)

        _build_test_stage_graphs(stage, (x,), {}, target, {})

    def test_provider_builds_one_graph_with_dist_moe_slot_input(self) -> None:
        """One stage graph traces with a representative Dist-MoE slot view."""
        graph = object()
        stage = types.SimpleNamespace(
            stage_index=0,
            graphs=None,
            is_first=True,
            is_last=False,
        )
        schedule = types.SimpleNamespace(
            _n_microbatches=4,
            _stages=[stage],
            rank=0,
            pipeline_order_with_comms={0: []},
        )
        ctx = _PipelineContext(schedule, [()] * 4, [{} for _ in range(4)], None, [])
        provider = GraphTrainerStageGraphProvider(
            loss_fn=lambda pred, target: pred.sum(),
            config=types.SimpleNamespace(
                compile=GraphTrainerCompileConfig(),
                training=types.SimpleNamespace(disable_cuda_graphs=False),
            ),
            plan=GraphExecutionPlan(
                pp_enabled=True,
                num_microbatches=4,
                unshard="schedule",
                reduce_grad="schedule",
                fuse_wgrad_accumulation=False,
            ),
        )

        slot = torch.tensor([0])
        forward_context = mock.Mock()
        forward_context.resolve_activation_slot.return_value = slot

        def build_graphs(stage, *_args, **_kwargs):
            stage.graphs = graph

        with (
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_builder."
                "_build_stage_graphs",
                side_effect=build_graphs,
            ) as build,
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_builder."
                "_compile_stage_graphs",
            ) as compile_graphs,
        ):
            provider.prepare_graphs(
                schedule,
                ctx,
                loss_kwargs={},
                dist_moe_forward_context=forward_context,
            )

        self.assertEqual(build.call_count, 1)
        self.assertEqual(compile_graphs.call_count, 1)
        self.assertIs(stage.graphs, graph)
        self.assertEqual(build.call_args.kwargs["activation_slot_id_1"], slot)
        info = forward_context.resolve_activation_slot.call_args.args[0]
        self.assertEqual((info.stage_index, info.microbatch_index), (0, 0))

    def test_dist_moe_slot_is_an_explicit_stage_graph_input(self) -> None:
        """The exact BF16 and MXFP8 forward schemas consume one slot input."""
        for op, slot_index in _dist_moe_forward_slot_arguments():
            with self.subTest(op=op):
                self.assertEqual(
                    op._schema.arguments[slot_index].name,
                    "activation_slot_id_1",
                )
                root = nn.Module()
                root.register_buffer(
                    "captured_slot", torch.tensor([0], dtype=torch.int64)
                )
                graph = fx.Graph()
                tensor = graph.placeholder("tensor")
                slot_input = graph.placeholder("activation_slot_id_1")
                graph.placeholder("backward_grad")
                captured_slot = graph.get_attr("captured_slot")
                captured_slot.meta["val"] = root.captured_slot
                args = [tensor] * (slot_index + 1)
                args[slot_index] = captured_slot
                forward = graph.call_function(op, tuple(args))
                graph.output(forward)
                traced = types.SimpleNamespace(gm=fx.GraphModule(root, graph))

                _rewrite_dist_moe_activation_slot_input(traced, input_index=1)

                self.assertEqual(
                    [
                        node.name
                        for node in traced.gm.graph.find_nodes(op="placeholder")
                    ],
                    ["tensor", "activation_slot_id_1", "backward_grad"],
                )
                self.assertIs(forward.args[slot_index], slot_input)
                self.assertNotIn(captured_slot, traced.gm.graph.nodes)

    def test_existing_stage_graphs_normalize_split_block_masks_in_place(self) -> None:
        arg_mbs = [(), ()]
        kwarg_mbs = [
            {"attention_metadata": mask} for mask in _split_batch_offset_block_masks()
        ]
        stage = types.SimpleNamespace(graphs=object())
        schedule = types.SimpleNamespace(
            _stages=[stage],
            rank=0,
            pipeline_order_with_comms={0: []},
        )
        ctx = _PipelineContext(schedule, arg_mbs, kwarg_mbs, None, [])
        provider = GraphTrainerStageGraphProvider(
            loss_fn=lambda pred, target: pred.sum(),
            config=types.SimpleNamespace(
                compile=GraphTrainerCompileConfig(),
                training=types.SimpleNamespace(disable_cuda_graphs=False),
            ),
            plan=_make_test_pp_plan(),
        )

        with (
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_builder."
                "_build_graph_pp_overlap_graphs",
                return_value={},
            ),
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_builder."
                "_compile_stage_graphs",
            ),
        ):
            provider.prepare_graphs(schedule, ctx, loss_kwargs={})

        self.assertIs(ctx.arg_mbs, arg_mbs)
        self.assertIs(ctx.kwarg_mbs, kwarg_mbs)
        mask0 = ctx.kwarg_mbs[0]["attention_metadata"]
        mask1 = ctx.kwarg_mbs[1]["attention_metadata"]

        self.assertEqual(_trace_mask_mod_replay(mask0, mask1), (False, True))

    def test_single_stage_schedule_hard_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime PP schedule"):
            _validate_graph_pp_config(
                compile_config=GraphTrainerCompileConfig(),
                parallelism=ParallelismConfig(pipeline_parallel_schedule="1F1B"),
            )

    def test_runtime_schedule_validation_accepts_interleaved(self) -> None:
        _validate_graph_pp_config(
            compile_config=GraphTrainerCompileConfig(),
            parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        )

    def test_split_fsdp_boundaries_reject_dense_region_overlap(self) -> None:
        compile_config = GraphTrainerCompileConfig(
            enable_fsdp_dense_region_overlap=True
        )
        with self.assertRaisesRegex(ValueError, "all-gathers and gradient reductions"):
            _make_test_spmd_plan(
                num_microbatches=2,
                fsdp_enabled=True,
                compile_config=compile_config,
            )

        with self.assertRaisesRegex(ValueError, "all-gathers and gradient reductions"):
            resolve_graph_execution_plan(
                compile_config,
                num_microbatches=2,
                parallelism=ParallelismConfig(
                    pipeline_parallel_schedule="Interleaved1F1B"
                ),
                pp_enabled=True,
                fsdp_enabled=True,
            )

        plan = _make_test_spmd_plan(
            num_microbatches=1,
            fsdp_enabled=True,
            compile_config=compile_config,
        )
        self.assertFalse(plan.has_gradient_accumulation)

    def test_precompile_rejects_gradient_accumulation(self) -> None:
        parallelism_context = types.SimpleNamespace(
            pp_enabled=False, fsdp_enabled=False
        )
        with self.assertRaisesRegex(
            ValueError,
            "SPMD with gradient accumulation does not support "
            "compile.precompile_artifact_dir",
        ):
            make_graph_runtime(
                [mock.Mock()],
                num_microbatches=2,
                parallelism_context=parallelism_context,
                config=types.SimpleNamespace(
                    compile=GraphTrainerCompileConfig(
                        precompile_artifact_dir="artifacts"
                    ),
                    parallelism=ParallelismConfig(),
                ),
                loss_fn=mock.Mock(),
            )

    def test_joint_stage_graphs_bind_runtime_meshes(self) -> None:
        traced = mock.Mock()
        module = nn.Linear(2, 2)
        runtime_meshes = [mock.Mock()]
        with mock.patch(
            "torchtitan.experiments.graph_trainer.spmd_graph_builder.run_traced"
        ) as run_traced_mock:
            GraphTrainerJointStageGraphs(
                traced=traced,
                module=module,
                num_param_grads=2,
                runtime_meshes=runtime_meshes,
            )

        run_traced_mock.assert_called_once_with(
            traced,
            module=module,
            precompile_meshes=runtime_meshes,
        )

    def test_simple_spmd_builds_joint_graph_without_partitioning(self) -> None:
        model = nn.Linear(3, 2)
        compile_config = GraphTrainerCompileConfig(enable_passes=False)
        stage = types.SimpleNamespace(
            submod=model,
            device=torch.device("cpu"),
            is_first=True,
            is_last=True,
            stage_index=0,
        )
        trainer_config = types.SimpleNamespace(
            compile=compile_config,
            training=types.SimpleNamespace(disable_cuda_graphs=True),
            model=None,
            parallelism=ParallelismConfig(),
        )
        x = torch.randn(4, 3)
        target = torch.randn(4, 2)

        def loss_fn(prediction, labels, **kwargs):
            return ((prediction - labels) ** 2).sum(), {}

        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder."
            "partition_joint_graph",
            side_effect=AssertionError("SPMD must not partition its joint graph"),
        ):
            _build_fwd_bwd_graphs(
                stage,
                (x,),
                {},
                target,
                {"global_loss_token_counts": torch.tensor(8)},
                loss_fn=loss_fn,
                trainer_config=trainer_config,
                parallelism_context=types.SimpleNamespace(),
            )

        self.assertIsInstance(stage.graphs, GraphTrainerJointStageGraphs)

    def test_scheduled_spmd_builds_and_reuses_joint_graph_without_partitioning(
        self,
    ) -> None:
        model = nn.Linear(3, 2)
        compile_config = GraphTrainerCompileConfig(enable_passes=False)
        stage = types.SimpleNamespace(
            submod=model,
            device=torch.device("cpu"),
            is_first=True,
            is_last=True,
            stage_index=0,
        )
        trainer_config = types.SimpleNamespace(
            compile=compile_config,
            training=types.SimpleNamespace(disable_cuda_graphs=True),
            model=None,
            parallelism=ParallelismConfig(),
        )
        x = torch.randn(4, 3)
        target = torch.randn(4, 2)

        def loss_fn(prediction, labels, **kwargs):
            return ((prediction - labels) ** 2).sum(), {}

        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder."
            "partition_joint_graph",
            side_effect=AssertionError("SPMD must not partition its joint graph"),
        ):
            _build_gradient_accumulation_fwd_bwd_graphs(
                stage,
                (x,),
                {},
                target,
                {"global_loss_token_counts": torch.tensor(8)},
                loss_fn=loss_fn,
                trainer_config=trainer_config,
                parallelism_context=types.SimpleNamespace(),
                plan=_make_test_spmd_plan(num_microbatches=2, fsdp_enabled=False),
            )

        self.assertIsInstance(stage.graphs, GraphTrainerScheduledFwdBwdStageGraphs)
        graphs = stage.graphs
        flat_params = list(model.parameters())
        flat_buffers = list(model.buffers())
        unsharded_params = graphs.unshard_params(flat_params)
        _, accumulators = graphs.forward_backward_nogradaccum(
            (x,),
            {},
            target,
            {"global_loss_token_counts": torch.tensor(8)},
            unsharded_param_values=unsharded_params,
            buffer_values=flat_buffers,
            runtime_validate=True,
        )
        _, accumulated_grads = graphs.forward_backward(
            (x,),
            {},
            target,
            {"global_loss_token_counts": torch.tensor(8)},
            unsharded_param_values=unsharded_params,
            buffer_values=flat_buffers,
            grad_accumulators=accumulators,
            runtime_validate=True,
        )

        expected_loss = ((model(x) - target) ** 2).sum()
        expected_grads = torch.autograd.grad(expected_loss, tuple(model.parameters()))
        for actual, accumulated, expected in zip(
            accumulators,
            accumulated_grads,
            expected_grads,
            strict=True,
        ):
            self.assertIs(actual, accumulated)
            torch.testing.assert_close(actual, 2 * expected)

    def test_spmd_auto_fsdp_collective_placement(self) -> None:
        single_microbatch = _make_test_spmd_plan(
            num_microbatches=1,
            fsdp_enabled=True,
        )
        gradient_accumulation = _make_test_spmd_plan(
            num_microbatches=3,
            fsdp_enabled=True,
        )
        self.assertEqual(
            (single_microbatch.unshard, single_microbatch.reduce_grad),
            ("every_microbatch", "every_microbatch"),
        )
        self.assertEqual(
            (gradient_accumulation.unshard, gradient_accumulation.reduce_grad),
            ("first_microbatch", "last_microbatch"),
        )

        schedule = _make_runtime_schedule_mock()
        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline."
            "_new_spmd_runtime_schedule",
            return_value=schedule,
        ):
            _make_spmd_runtime_schedule(
                mock.Mock(),
                loss_fn=mock.Mock(),
                plan=gradient_accumulation,
            )

        actions = schedule.pipeline_order_with_comms[0]
        self.assertEqual(
            [action.computation_type for action in actions],
            [
                FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
                FORWARD_BACKWARD,
                FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
                RESHARD,
            ],
        )

    def test_spmd_fsdp_boundaries_in_edge_microbatches(self) -> None:
        # The default reshard policy does not block parameter reuse.
        plan = _make_test_spmd_plan(num_microbatches=2, fsdp_enabled=True)

        self.assertEqual(plan.unshard, "first_microbatch")
        self.assertEqual(plan.reduce_grad, "last_microbatch")
        self.assertTrue(plan.reuse_unsharded_parameters)

    def test_spmd_parameter_reuse_disables_reshard_after_forward(self) -> None:
        @dataclasses.dataclass
        class TrainerConfig:
            compile: GraphTrainerCompileConfig
            parallelism: ParallelismConfig

        trainer_config = TrainerConfig(
            compile=GraphTrainerCompileConfig(),
            parallelism=ParallelismConfig(fsdp_reshard_after_forward="always"),
        )
        for num_microbatches, expected_policy in ((1, "always"), (2, "never")):
            with self.subTest(num_microbatches=num_microbatches):
                with (
                    mock.patch(
                        "torchtitan.experiments.graph_trainer.graph_pp.pipeline."
                        "_make_spmd_runtime_schedule",
                    ),
                    mock.patch(
                        "torchtitan.experiments.graph_trainer.graph_pp.pipeline."
                        "_register_graph_runtime",
                    ) as register_mock,
                ):
                    _make_spmd_graph_runtime(
                        mock.Mock(),
                        plan=_make_test_spmd_plan(
                            num_microbatches=num_microbatches, fsdp_enabled=True
                        ),
                        trainer_config=cast(Any, trainer_config),
                        loss_fn=mock.Mock(),
                        parallelism_context=mock.Mock(),
                    )
                config = register_mock.call_args.kwargs["config"]
                self.assertEqual(
                    config.parallelism.fsdp_reshard_after_forward, expected_policy
                )
        self.assertEqual(
            trainer_config.parallelism.fsdp_reshard_after_forward, "always"
        )

    def test_spmd_without_gradient_accumulation_ignores_fsdp_modes(self) -> None:
        with self.assertLogs(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline", "WARNING"
        ):
            plan = _make_test_spmd_plan(
                num_microbatches=1,
                fsdp_enabled=True,
                compile_config=GraphTrainerCompileConfig(
                    spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                        fsdp_param_unshard_mode="every_microbatch",
                        fsdp_grad_reduce_mode="every_microbatch",
                    ),
                ),
            )

        self.assertEqual(
            (plan.unshard, plan.reduce_grad), ("every_microbatch", "every_microbatch")
        )
        self.assertFalse(plan.has_gradient_accumulation)

    def test_pp_fsdp_boundaries_run_in_schedule(self) -> None:
        plan = _make_test_pp_plan()
        self.assertEqual((plan.unshard, plan.reduce_grad), ("schedule", "schedule"))
        self.assertFalse(plan.reuse_unsharded_parameters)
        self.assertFalse(plan.fuse_wgrad_accumulation)

        no_fsdp = _make_test_pp_plan(fsdp_enabled=False)
        self.assertTrue(no_fsdp.extract_fsdp_param_unshard)
        self.assertFalse(no_fsdp.extract_fsdp_grad_reduction)

        with self.assertLogs(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline", "WARNING"
        ):
            plan = resolve_graph_execution_plan(
                GraphTrainerCompileConfig(
                    spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                        fsdp_param_unshard_mode="every_microbatch",
                    ),
                ),
                num_microbatches=2,
                parallelism=ParallelismConfig(
                    pipeline_parallel_schedule="Interleaved1F1B"
                ),
                pp_enabled=True,
                fsdp_enabled=True,
            )
        self.assertEqual((plan.unshard, plan.reduce_grad), ("schedule", "schedule"))

    def test_gradient_accumulation_policy_auto(self) -> None:
        single_microbatch = _make_test_spmd_plan(num_microbatches=1, fsdp_enabled=False)
        multiple_microbatches = _make_test_spmd_plan(
            num_microbatches=2, fsdp_enabled=False
        )

        self.assertFalse(single_microbatch.fuse_wgrad_accumulation)
        self.assertFalse(multiple_microbatches.fuse_wgrad_accumulation)

        fsdp_reduction_in_every_graph = _make_test_spmd_plan(
            num_microbatches=2,
            fsdp_enabled=True,
            compile_config=GraphTrainerCompileConfig(
                numerics_changing_optim=True,
                spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                    fsdp_param_unshard_mode="every_microbatch",
                    fsdp_grad_reduce_mode="every_microbatch",
                ),
            ),
        )
        self.assertFalse(fsdp_reduction_in_every_graph.fuse_wgrad_accumulation)

        optimized = _make_test_spmd_plan(
            num_microbatches=2,
            fsdp_enabled=False,
            compile_config=GraphTrainerCompileConfig(numerics_changing_optim=True),
        )
        self.assertTrue(optimized.fuse_wgrad_accumulation)

    def test_spmd_fwd_bwd_plan_agrees_with_schedule_and_graph_boundaries(self) -> None:
        cases = (
            (
                GraphTrainerCompileConfig(),
                1,
                False,
                None,
                None,
                (FULL_FORWARD_BACKWARD,),
                False,
            ),
            (
                GraphTrainerCompileConfig(),
                2,
                False,
                None,
                None,
                (FORWARD_BACKWARD_NOGRADACCUM, FULL_FORWARD_BACKWARD),
                True,
            ),
            (
                GraphTrainerCompileConfig(),
                3,
                True,
                "first_microbatch",
                "last_microbatch",
                (
                    FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
                    FORWARD_BACKWARD,
                    FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
                    RESHARD,
                ),
                True,
            ),
            (
                GraphTrainerCompileConfig(
                    spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                        fsdp_param_unshard_mode="first_microbatch",
                        fsdp_grad_reduce_mode="last_microbatch",
                    ),
                ),
                3,
                True,
                "first_microbatch",
                "last_microbatch",
                (
                    FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
                    FORWARD_BACKWARD,
                    FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
                    RESHARD,
                ),
                True,
            ),
            (
                GraphTrainerCompileConfig(
                    spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                        fsdp_grad_reduce_mode="last_microbatch"
                    )
                ),
                2,
                True,
                "first_microbatch",
                "last_microbatch",
                (
                    FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
                    FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
                    RESHARD,
                ),
                True,
            ),
            (
                GraphTrainerCompileConfig(
                    spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                        fsdp_param_unshard_mode="every_microbatch",
                        fsdp_grad_reduce_mode="every_microbatch",
                    ),
                ),
                2,
                True,
                "every_microbatch",
                "every_microbatch",
                (FORWARD_BACKWARD_NOGRADACCUM, FULL_FORWARD_BACKWARD),
                True,
            ),
        )
        for (
            config,
            num_microbatches,
            fsdp_enabled,
            unshard,
            reduce_grad,
            expected,
            has_gradient_accumulation,
        ) in cases:
            with self.subTest(config=config, num_microbatches=num_microbatches):
                plan = _make_test_spmd_plan(
                    num_microbatches=num_microbatches,
                    fsdp_enabled=fsdp_enabled,
                    compile_config=config,
                )
                self.assertEqual(
                    (plan.unshard, plan.reduce_grad), (unshard, reduce_grad)
                )
                self.assertEqual(
                    plan.has_gradient_accumulation, has_gradient_accumulation
                )
                schedule = _make_runtime_schedule_mock()
                with mock.patch(
                    "torchtitan.experiments.graph_trainer.graph_pp.pipeline."
                    "_new_spmd_runtime_schedule",
                    return_value=schedule,
                ):
                    _make_spmd_runtime_schedule(
                        mock.Mock(), loss_fn=mock.Mock(), plan=plan
                    )
                self.assertEqual(
                    tuple(
                        action.computation_type
                        for action in schedule.pipeline_order_with_comms[0]
                    ),
                    expected,
                )

    def test_gradient_accumulation_does_not_require_wgrad_fusion(self) -> None:
        plan = _make_test_spmd_plan(
            num_microbatches=2,
            fsdp_enabled=False,
            compile_config=GraphTrainerCompileConfig(
                spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                    fuse_wgrad_accumulation="disabled",
                ),
            ),
        )

        self.assertFalse(plan.fuse_wgrad_accumulation)

    def test_wgrad_fusion_ignored_without_gradient_accumulation(self) -> None:
        compile_config = GraphTrainerCompileConfig(
            spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                fuse_wgrad_accumulation="enabled",
            ),
        )
        with self.assertLogs(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline", "WARNING"
        ):
            plan = _make_test_spmd_plan(
                num_microbatches=1,
                fsdp_enabled=False,
                compile_config=compile_config,
            )
        self.assertFalse(plan.fuse_wgrad_accumulation)

        with self.assertNoLogs(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline", "WARNING"
        ):
            plan = resolve_graph_execution_plan(
                compile_config,
                num_microbatches=2,
                parallelism=ParallelismConfig(
                    pipeline_parallel_schedule="Interleaved1F1B"
                ),
                pp_enabled=True,
                fsdp_enabled=True,
            )
        self.assertTrue(plan.fuse_wgrad_accumulation)

    def test_wgrad_fusion_without_fsdp_does_not_warn(self) -> None:
        with self.assertNoLogs(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline", "WARNING"
        ):
            plan = _make_test_spmd_plan(
                num_microbatches=2,
                fsdp_enabled=False,
                compile_config=GraphTrainerCompileConfig(
                    spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                        fuse_wgrad_accumulation="enabled",
                    ),
                ),
            )
        self.assertTrue(plan.fuse_wgrad_accumulation)

    def test_wgrad_fusion_with_fsdp_requires_split_grad_reduction(self) -> None:
        with self.assertRaisesRegex(ValueError, "'last_microbatch'"):
            _make_test_spmd_plan(
                num_microbatches=2,
                fsdp_enabled=True,
                compile_config=GraphTrainerCompileConfig(
                    spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                        fsdp_param_unshard_mode="every_microbatch",
                        fsdp_grad_reduce_mode="every_microbatch",
                        fuse_wgrad_accumulation="enabled",
                    ),
                ),
            )

    def test_pipeline_parallel_fuses_schedule_gradient_accumulation(self) -> None:
        plan = resolve_graph_execution_plan(
            GraphTrainerCompileConfig(numerics_changing_optim=True),
            num_microbatches=2,
            parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
            pp_enabled=True,
            fsdp_enabled=True,
        )

        self.assertTrue(plan.fuse_wgrad_accumulation)

    def test_spmd_schedule_has_one_joint_action_per_microbatch(self) -> None:
        schedule = _make_runtime_schedule_mock()
        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline."
            "_new_spmd_runtime_schedule",
            return_value=schedule,
        ):
            _make_spmd_runtime_schedule(
                mock.Mock(),
                loss_fn=mock.Mock(),
                plan=_make_test_spmd_plan(num_microbatches=1, fsdp_enabled=False),
            )

        actions = schedule.pipeline_order_with_comms[0]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].computation_type, FULL_FORWARD_BACKWARD)
        self.assertEqual(actions[0].stage_index, 0)
        self.assertEqual(actions[0].microbatch_index, 0)
        self.assertIsNone(actions[0].sub_actions)

    def test_spmd_schedule_fuses_fsdp_boundaries_into_edge_microbatches(
        self,
    ) -> None:
        schedule = _make_runtime_schedule_mock()
        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline."
            "_new_spmd_runtime_schedule",
            return_value=schedule,
        ):
            _make_spmd_runtime_schedule(
                mock.Mock(),
                loss_fn=mock.Mock(),
                plan=_make_test_spmd_plan(
                    num_microbatches=3,
                    fsdp_enabled=True,
                    compile_config=GraphTrainerCompileConfig(
                        spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                            fsdp_param_unshard_mode="first_microbatch",
                            fsdp_grad_reduce_mode="last_microbatch",
                        ),
                    ),
                ),
            )

        actions = schedule.pipeline_order_with_comms[0]
        self.assertEqual(
            [action.computation_type for action in actions],
            [
                FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
                FORWARD_BACKWARD,
                FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
                RESHARD,
            ],
        )

    def test_graph_schedule_backward_actions_encode_gradient_reduction(self) -> None:
        def make_schedule():
            return types.SimpleNamespace(
                pipeline_order_with_comms={
                    0: [
                        _Action(0, FULL_BACKWARD, 0),
                        _Action(0, BACKWARD_WEIGHT, 0),
                        _Action(
                            -1,
                            OVERLAP_F_B,
                            None,
                            (
                                _Action(0, FORWARD, 1),
                                _Action(1, FULL_BACKWARD, 0),
                            ),
                        ),
                    ]
                }
            )

        deferred_schedule = make_schedule()
        _set_graph_backward_actions(
            deferred_schedule,
            extract_fsdp_grad_reduction=True,
        )
        deferred_actions = deferred_schedule.pipeline_order_with_comms[0]
        self.assertEqual(deferred_actions[0].computation_type, BACKWARD)
        self.assertEqual(deferred_actions[1].computation_type, BACKWARD_WEIGHT)
        self.assertEqual(
            deferred_actions[2].sub_actions[1].computation_type,
            BACKWARD,
        )

        fused_reduction_schedule = make_schedule()
        _set_graph_backward_actions(
            fused_reduction_schedule,
            extract_fsdp_grad_reduction=False,
        )
        fused_reduction_actions = fused_reduction_schedule.pipeline_order_with_comms[0]
        self.assertEqual(
            fused_reduction_actions[0].computation_type,
            BACKWARD_WITH_REDUCE_GRAD,
        )
        self.assertEqual(
            fused_reduction_actions[1].computation_type,
            BACKWARD_WEIGHT_WITH_REDUCE_GRAD,
        )
        self.assertEqual(
            fused_reduction_actions[2].sub_actions[1].computation_type,
            BACKWARD_WITH_REDUCE_GRAD,
        )

        self.assertFalse(_grad_reduction_runs_in_backward(deferred_actions[0]))
        self.assertTrue(_grad_reduction_runs_in_backward(fused_reduction_actions[0]))
        self.assertFalse(_grad_reduction_runs_in_backward(deferred_actions[1]))
        self.assertTrue(_grad_reduction_runs_in_backward(fused_reduction_actions[1]))

    def test_graph_pp_accepts_zero_two_fsdp_reshard_policies(self) -> None:
        for policy in ("default", "never"):
            _validate_graph_pp_config(
                compile_config=GraphTrainerCompileConfig(),
                parallelism=ParallelismConfig(
                    pipeline_parallel_schedule="Interleaved1F1B",
                    fsdp_reshard_after_forward=policy,
                ),
            )

        with self.assertRaisesRegex(ValueError, "fsdp_reshard_after_forward"):
            _validate_graph_pp_config(
                compile_config=GraphTrainerCompileConfig(),
                parallelism=ParallelismConfig(
                    pipeline_parallel_schedule="Interleaved1F1B",
                    fsdp_reshard_after_forward="always",
                ),
            )

    def test_graph_pp_compile_uses_inductor_compilation_with_default_backend(
        self,
    ) -> None:
        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        for node in gm.graph.find_nodes(op="placeholder"):
            node.meta["val"] = torch.randn(2)
        compile_config = GraphTrainerCompileConfig()

        def boxed_apply_graph_passes(gm, example_inputs, passes, compile_config):
            return ensure_boxed_graph_module(gm)

        with (
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_builder_utils."
                "final_inductor_compile_passes",
                return_value=[],
            ) as final_inductor_passes,
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_builder_utils."
                "apply_graph_passes",
                side_effect=boxed_apply_graph_passes,
            ) as apply_graph_passes,
        ):
            compiled = _compile_graph_pp_module(
                gm,
                compile_config=compile_config,
                graph_name="test_graph",
                use_cuda_graph=True,
            )

        self.assertIs(compiled, gm)
        final_inductor_passes.assert_called_once_with(
            compile_config,
            use_cuda_graph=True,
            boxed_codegen=True,
        )
        apply_graph_passes.assert_called_once()

    def test_graph_pp_graph_execution_uses_mutable_boxed_args(self) -> None:
        gm = torch.fx.symbolic_trace(lambda x, y: x + y)
        ensure_boxed_graph_module(gm)
        x = torch.randn(2)
        y = torch.randn(2)
        args = [x, y]

        (out,) = _execute_graph_module(gm, args)

        self.assertEqual(args, [])
        self.assertTrue(torch.equal(out, x + y))

    def test_full_inductor_overlap_builds_multiplexed_graph(self) -> None:
        full_compile = GraphTrainerCompileConfig(
            enable_passes=True,
            inductor_compilation="full",
        )
        torch.manual_seed(0)
        x = torch.randn(2, 4, requires_grad=True)
        stage0_mod = nn.Linear(4, 3)
        stage1_mod = nn.Linear(4, 3)
        stage0 = _make_test_stage(
            stage0_mod,
            is_last=False,
            loss_fn=None,
            stage_index=0,
            output_grads=torch.empty_like(stage0_mod(x)),
        )
        stage1 = _make_test_stage(
            stage1_mod,
            is_last=False,
            loss_fn=None,
            stage_index=1,
            output_grads=torch.empty_like(stage1_mod(x)),
        )
        _build_test_stage_graphs(stage0, (x,), {}, None, {}, compile_graphs=False)
        _build_test_stage_graphs(stage1, (x,), {}, None, {}, compile_graphs=False)
        stage0.compile_config = full_compile
        stage1.compile_config = full_compile
        schedule = types.SimpleNamespace(
            _stages=[stage0, stage1],
            rank=0,
            pipeline_order_with_comms={
                0: [
                    _Action(
                        -1,
                        OVERLAP_F_B,
                        None,
                        (
                            _Action(0, FORWARD, 0, None),
                            _Action(1, BACKWARD, 0, None),
                        ),
                    )
                ]
            },
        )

        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder."
            "_compile_graph_pp_module",
            side_effect=lambda gm, *, compile_config, graph_name, use_cuda_graph: gm,
        ) as compile_graph:
            overlap_graphs = _build_graph_pp_overlap_graphs(
                schedule,
                compile_config=full_compile,
                use_cuda_graph=True,
            )

        self.assertIn((0, 1), overlap_graphs)
        compile_graph.assert_called_once()
        self.assertIs(compile_graph.call_args.kwargs["compile_config"], full_compile)

    def test_overlap_backward_input_sub_action_errors(self) -> None:
        schedule = types.SimpleNamespace(
            _stages=[],
            rank=0,
            pipeline_order_with_comms={
                0: [
                    _Action(
                        -1,
                        OVERLAP_F_B,
                        None,
                        (
                            _Action(0, FORWARD, 0, None),
                            _Action(1, BACKWARD_INPUT, 0, None),
                        ),
                    )
                ]
            },
        )

        with self.assertRaisesRegex(NotImplementedError, "BACKWARD_INPUT"):
            _build_graph_pp_overlap_graphs(
                schedule,
                compile_config=GraphTrainerCompileConfig(enable_passes=False),
                use_cuda_graph=True,
            )

    def test_intermediate_stage_graphs_match_eager_grads(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3)
        x = torch.randn(2, 4, requires_grad=True)
        output_grad = torch.randn(2, 3)
        stage = _make_test_stage(
            model,
            is_last=False,
            loss_fn=None,
            stage_index=0,
            output_grads=output_grad,
        )

        _build_test_stage_graphs(stage, (x,), {}, None, {})

        state = list(model.parameters())
        output, saved = stage.graphs.forward(
            (x,),
            {},
            None,
            {},
            unsharded_param_values=state,
            buffer_values=[],
        )
        self.assertTrue(torch.allclose(output, model(x)))

        input_grads, param_grads = stage.graphs.full_backward(
            saved,
            (output_grad,),
        )
        expected_grads = torch.autograd.grad(
            model(x),
            [*model.parameters(), x],
            grad_outputs=output_grad,
        )
        for actual, expected in zip(
            param_grads + input_grads, expected_grads, strict=True
        ):
            self.assertTrue(torch.allclose(actual, expected))

        di_grads, dw_inputs = stage.graphs.backward_input(
            saved,
            (output_grad,),
        )
        dw_grads = stage.graphs.backward_weight(dw_inputs)
        for actual, expected in zip(dw_grads + di_grads, expected_grads, strict=True):
            self.assertTrue(torch.allclose(actual, expected))

    def test_pp_backward_graph_fuses_repeated_wgrad_accumulation(self) -> None:
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(4, 5, bias=False, dtype=torch.bfloat16),
            nn.Linear(5, 3, bias=False, dtype=torch.bfloat16),
        )
        x0 = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
        x1 = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
        output_grad0 = torch.randn(2, 3, dtype=torch.bfloat16)
        output_grad1 = torch.randn(2, 3, dtype=torch.bfloat16)
        stage = _make_test_stage(
            model,
            is_last=False,
            loss_fn=None,
            stage_index=0,
            output_grads=output_grad0,
        )
        _build_test_stage_graphs(
            stage,
            (x0,),
            {},
            None,
            {},
            gradient_accumulation=True,
            fuse_wgrad_accumulation=True,
        )

        graphs = cast(GraphTrainerStageGraphs, stage.graphs)
        self.assertEqual(len(graphs.full_bw_initial_grad_accumulators), 2)
        self.assertEqual(graphs.full_bw_grad_accumulator_indices, (0, 1))
        self.assertEqual(
            sum(
                node.target == torch.ops.aten.addmm_.default
                for node in graphs.modules.full_bw.graph.nodes
            ),
            2,
        )

        state = list(model.parameters())
        _, saved0 = graphs.forward(
            (x0,),
            {},
            None,
            {},
            unsharded_param_values=state,
            buffer_values=[],
        )
        _, param_grads0 = graphs.full_backward(saved0, (output_grad0,))
        _, saved1 = graphs.forward(
            (x1,),
            {},
            None,
            {},
            unsharded_param_values=state,
            buffer_values=[],
        )
        _, param_grads1 = graphs.full_backward(
            saved1,
            (output_grad1,),
            grad_accumulators=param_grads0,
        )

        weight0, weight1 = model.parameters()
        hidden0 = x0 @ weight0.t()
        hidden_grad0 = output_grad0 @ weight1
        expected = [hidden_grad0.t() @ x0, output_grad0.t() @ hidden0]
        expected_first = [grad.clone() for grad in expected]
        hidden1 = x1 @ weight0.t()
        hidden_grad1 = output_grad1 @ weight1
        expected[0].addmm_(hidden_grad1.t(), x1)
        expected[1].addmm_(output_grad1.t(), hidden1)
        for index in range(2):
            self.assertIs(param_grads1[index], param_grads0[index])
            torch.testing.assert_close(param_grads1[index], expected[index])

        _, saved0 = graphs.forward(
            (x0,),
            {},
            None,
            {},
            unsharded_param_values=state,
            buffer_values=[],
        )
        _, reset_param_grads = graphs.full_backward(saved0, (output_grad0,))
        for index in range(2):
            self.assertIs(reset_param_grads[index], param_grads0[index])
            torch.testing.assert_close(reset_param_grads[index], expected_first[index])

        _, saved0 = graphs.forward(
            (x0,),
            {},
            None,
            {},
            unsharded_param_values=state,
            buffer_values=[],
        )
        _, dw_inputs0 = graphs.backward_input(saved0, (output_grad0,))
        param_grads0 = graphs.backward_weight(dw_inputs0)
        for full_bw_accumulator, bw_dw_accumulator in zip(
            graphs.full_bw_initial_grad_accumulators,
            graphs.bw_dw_initial_grad_accumulators,
            strict=True,
        ):
            self.assertIs(full_bw_accumulator, bw_dw_accumulator)
        _, saved1 = graphs.forward(
            (x1,),
            {},
            None,
            {},
            unsharded_param_values=state,
            buffer_values=[],
        )
        _, dw_inputs1 = graphs.backward_input(saved1, (output_grad1,))
        param_grads1 = graphs.backward_weight(
            dw_inputs1,
            grad_accumulators=param_grads0,
        )
        self.assertEqual(graphs.bw_dw_grad_accumulator_indices, (0, 1))
        for index in range(2):
            self.assertIs(param_grads1[index], param_grads0[index])
            torch.testing.assert_close(param_grads1[index], expected[index])

    def test_unshard_params_validates_exact_flat_param_count(self) -> None:
        model = nn.Linear(4, 3)
        x = torch.randn(2, 4, requires_grad=True)
        stage = _make_test_stage(
            model,
            is_last=False,
            loss_fn=None,
            stage_index=0,
            output_grads=torch.empty_like(model(x)),
        )
        _build_test_stage_graphs(stage, (x,), {}, None, {})

        with self.assertRaisesRegex(ValueError, "one runtime value per flat param"):
            stage.graphs.unshard_params(
                [*model.parameters(), object()],
                runtime_validate=True,
            )

    def test_unshard_params_skips_repeated_count_validation_by_default(self) -> None:
        model = nn.Linear(4, 3)
        x = torch.randn(2, 4, requires_grad=True)
        stage = _make_test_stage(
            model,
            is_last=False,
            loss_fn=None,
            stage_index=0,
            output_grads=torch.empty_like(model(x)),
        )
        _build_test_stage_graphs(stage, (x,), {}, None, {})

        extra_value = object()
        expected_values = [*model.parameters(), extra_value]
        actual_values = stage.graphs.unshard_params(expected_values)
        self.assertEqual(len(actual_values), len(expected_values))
        for actual, expected in zip(actual_values, expected_values, strict=True):
            self.assertIs(actual, expected)

    def test_forward_remaps_inputs_after_unshard_collapses_flat_params(self) -> None:
        graphs = GraphTrainerStageGraphs(
            modules=types.SimpleNamespace(unshard=object()),
            meta=types.SimpleNamespace(
                num_sharded_param_values=2,
                num_fw_param_inputs=1,
                fwd_input_names=("unsharded_weight", "x"),
                fwd_flat_input_indices=(2,),
                uses_dist_moe_activation_slot=False,
                is_last_stage=False,
            ),
        )
        unsharded_weight = object()
        x = torch.randn(2, 4)

        forward_args = graphs._forward_args(
            (x,),
            {},
            None,
            {},
            unsharded_param_values=[unsharded_weight],
            buffer_values=[],
            runtime_validate=True,
        )

        self.assertIs(forward_args[0], unsharded_weight)
        self.assertIs(forward_args[1], x)

    def test_stage_trace_preserves_buffers_and_forward_keeps_mutations(self) -> None:
        class BufferCountingStage(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4))
                self.register_buffer("tokens", torch.tensor(5.0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    self.tokens.add_(x.detach().sum())
                return x * self.weight

        torch.manual_seed(0)
        model = BufferCountingStage()
        x = torch.randn(2, 4, requires_grad=True)
        stage = _make_test_stage(
            model,
            is_last=False,
            loss_fn=None,
            stage_index=0,
            output_grads=torch.empty_like(model(x)),
        )
        initial_tokens = model.tokens.clone()

        _build_test_stage_graphs(stage, (x,), {}, None, {})

        self.assertTrue(torch.equal(model.tokens, initial_tokens))
        self.assertGreater(stage.graphs.meta.partition.num_fwd_side_effect_outputs, 0)

        stage.graphs.forward(
            (x,),
            {},
            None,
            {},
            unsharded_param_values=[model.weight],
            buffer_values=[model.tokens],
        )

        self.assertTrue(torch.equal(model.tokens, initial_tokens + x.detach().sum()))

    def test_multiplexed_graph_returns_backward_then_forward_outputs(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3)
        x = torch.randn(2, 4, requires_grad=True)
        output_grad = torch.randn(2, 3)
        stage = _make_test_stage(
            model,
            is_last=False,
            loss_fn=None,
            stage_index=0,
            output_grads=output_grad,
        )
        _build_test_stage_graphs(stage, (x,), {}, None, {})

        state = list(model.parameters())
        fw_outputs = _boxed_run(
            stage.graphs.modules.fw,
            [*state, x],
        )
        bw_outputs = _boxed_run(
            stage.graphs.modules.full_bw,
            [*fw_outputs[1:], output_grad],
        )
        multiplexed = multiplex_fw_bw_graph(
            stage.graphs.modules.fw,
            stage.graphs.modules.full_bw,
        )

        multiplexed_outputs = _boxed_run(
            multiplexed,
            [*fw_outputs[1:], output_grad, *state, x],
        )

        expected_outputs = [*bw_outputs, *fw_outputs]
        self.assertEqual(len(multiplexed_outputs), len(expected_outputs))
        for actual, expected in zip(
            multiplexed_outputs,
            expected_outputs,
            strict=True,
        ):
            self.assertTrue(torch.allclose(actual, expected))

    def test_multiplexed_graph_is_prebuilt_for_overlap_action(self) -> None:
        torch.manual_seed(0)
        compile_config = GraphTrainerCompileConfig(enable_passes=False)
        x = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
        stage0_mod = nn.Linear(4, 3, dtype=torch.bfloat16)
        stage1_mod = nn.Linear(4, 3, dtype=torch.bfloat16)
        output_grad = torch.randn_like(stage1_mod(x))
        stage0 = _make_test_stage(
            stage0_mod,
            is_last=False,
            loss_fn=None,
            stage_index=0,
            compile_config=compile_config,
            output_grads=torch.empty_like(stage0_mod(x)),
        )
        stage1 = _make_test_stage(
            stage1_mod,
            is_last=False,
            loss_fn=None,
            stage_index=1,
            compile_config=compile_config,
            output_grads=output_grad,
        )
        _build_test_stage_graphs(stage0, (x,), {}, None, {}, compile_graphs=False)
        _build_test_stage_graphs(
            stage1,
            (x,),
            {},
            None,
            {},
            compile_graphs=False,
            gradient_accumulation=True,
            fuse_wgrad_accumulation=True,
        )
        schedule = types.SimpleNamespace(
            _stages=[stage0, stage1],
            rank=0,
            pipeline_order_with_comms={
                0: [
                    _Action(
                        -1,
                        OVERLAP_F_B,
                        None,
                        (
                            _Action(0, FORWARD, 0, None),
                            _Action(1, BACKWARD_WITH_REDUCE_GRAD, 0, None),
                        ),
                    )
                ]
            },
        )

        stage1_graphs = cast(GraphTrainerStageGraphs, stage1.graphs)
        _, backward_saved = stage1_graphs._split_forward_outputs(
            _boxed_run(
                stage1_graphs.modules.fw,
                [*stage1_mod.parameters(), x],
            )
        )

        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder."
            "_compile_graph_pp_module",
            side_effect=lambda gm, **kwargs: ensure_boxed_graph_module(gm),
        ) as compile_graph_module:
            overlap_graphs = _build_graph_pp_overlap_graphs(
                schedule,
                compile_config=compile_config,
                use_cuda_graph=True,
            )

        compile_graph_module.assert_called_once()
        self.assertIn((0, 1), overlap_graphs)
        overlap_graph = overlap_graphs[(0, 1)]
        (_, param_grads0, forward_output, _,) = overlap_graph.forward_backward(
            backward_saved_values_for_backward=backward_saved,
            output_grads_from_next=(output_grad,),
            forward_args=(x,),
            forward_kwargs={},
            forward_target=None,
            forward_loss_kwargs={},
            forward_unsharded_param_values=list(stage0_mod.parameters()),
            forward_buffer_values=[],
        )

        expected_weight_grad = torch.zeros_like(stage1_mod.weight)
        expected_weight_grad.addmm_(output_grad.t(), x)
        expected_bias_grad = output_grad.sum(dim=0)
        torch.testing.assert_close(param_grads0[0], expected_weight_grad)
        torch.testing.assert_close(param_grads0[1], expected_bias_grad)
        torch.testing.assert_close(forward_output, stage0_mod(x))

        (_, param_grads1, _, _,) = overlap_graph.forward_backward(
            backward_saved_values_for_backward=backward_saved,
            output_grads_from_next=(output_grad,),
            forward_args=(x,),
            forward_kwargs={},
            forward_target=None,
            forward_loss_kwargs={},
            forward_unsharded_param_values=list(stage0_mod.parameters()),
            forward_buffer_values=[],
            backward_grad_accumulators=param_grads0,
        )
        self.assertIs(param_grads1[0], param_grads0[0])
        self.assertIs(param_grads1[1], param_grads0[1])
        torch.testing.assert_close(
            param_grads1[0],
            expected_weight_grad + expected_weight_grad,
        )
        torch.testing.assert_close(
            param_grads1[1],
            expected_bias_grad + expected_bias_grad,
        )

    def test_last_stage_graphs_return_loss_and_input_grad(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3)

        def loss_fn(pred, target, global_loss_token_counts):
            return ((pred - target) ** 2).sum() / global_loss_token_counts

        stage = _make_test_stage(
            model,
            is_last=True,
            loss_fn=loss_fn,
            stage_index=1,
        )
        x = torch.randn(2, 4, requires_grad=True)
        target = torch.randn(2, 3)
        global_loss_token_counts = torch.tensor(2.0)

        _build_test_stage_graphs(
            stage,
            (x,),
            {},
            target,
            {"global_loss_token_counts": global_loss_token_counts},
        )
        self.assertEqual(stage.graphs.meta.partition.backward_grad_input_names, ())
        self.assertEqual(stage.graphs.meta.partition.backward_grad_input_indices, ())

        state = list(model.parameters())
        loss, saved = stage.graphs.forward(
            (x,),
            {},
            target,
            {"global_loss_token_counts": global_loss_token_counts},
            unsharded_param_values=state,
            buffer_values=[],
        )
        expected_loss = loss_fn(model(x), target, global_loss_token_counts)
        self.assertTrue(torch.allclose(loss, expected_loss))

        input_grads, param_grads = stage.graphs.full_backward(
            saved,
            (),
        )
        expected_grads = torch.autograd.grad(
            expected_loss,
            [*model.parameters(), x],
        )
        for actual, expected in zip(
            param_grads + input_grads, expected_grads, strict=True
        ):
            self.assertTrue(torch.allclose(actual, expected))

    def test_last_stage_chunked_loss_preserves_hidden_grad_accumulator(self) -> None:
        class LastStage(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(16, 16),
                    nn.ReLU(),
                    nn.Linear(16, 16),
                )
                self.lm_head = nn.Linear(16, 33, bias=False)
                self._skip_lm_head = True

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                hidden_states = self.block(x)
                if self._skip_lm_head:
                    return hidden_states
                return self.lm_head(hidden_states)

        torch.manual_seed(0)
        model = LastStage()
        loss_fn = ChunkedLossWrapperWithParamGrads(
            ChunkedLossWrapperWithParamGrads.Config(num_chunks=4)
        )
        loss_fn.set_lm_head(model.lm_head)
        stage = _make_test_stage(
            model,
            is_last=True,
            loss_fn=loss_fn,
            stage_index=1,
        )
        x = torch.randn(16, 16, requires_grad=True)
        labels = torch.randint(0, 33, (16,))
        global_loss_token_counts = torch.tensor(float(labels.numel()))

        _build_test_stage_graphs(
            stage,
            (x,),
            {},
            labels,
            {"global_loss_token_counts": global_loss_token_counts},
        )

        state = list(model.parameters())
        loss, saved = stage.graphs.forward(
            (x,),
            {},
            labels,
            {"global_loss_token_counts": global_loss_token_counts},
            unsharded_param_values=state,
            buffer_values=[],
        )
        input_grads, param_grads = stage.graphs.full_backward(
            saved,
            (),
        )

        expected_loss, _ = loss_fn(model(x), labels, global_loss_token_counts)
        expected_grads = torch.autograd.grad(
            expected_loss,
            [*model.parameters(), x],
        )
        self.assertTrue(torch.equal(loss, expected_loss))
        for actual, expected in zip(
            param_grads + input_grads, expected_grads, strict=True
        ):
            self.assertTrue(torch.equal(actual, expected))
        self.assertGreater(torch.linalg.vector_norm(input_grads[0]).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
