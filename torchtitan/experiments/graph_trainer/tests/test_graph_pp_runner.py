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
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    OVERLAP_F_B,
    REDUCE_GRAD,
    RESHARD,
    WAIT_REDUCE_GRAD,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    compute_annotated_loss,
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
from torchtitan.experiments.graph_trainer.graph_pp import multiplex_fw_bw_graph
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
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    _grad_reduction_runs_in_backward,
    _post_fwd_common,
    _prepare_fwd_user_args,
    BACKWARD,
    BACKWARD_WEIGHT_WITH_REDUCE_GRAD,
    BACKWARD_WITH_REDUCE_GRAD,
    FORWARD_BACKWARD,
    FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
    FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
    FORWARD_BACKWARD_NOGRADACCUM,
    FULL_FORWARD_BACKWARD,
    GraphRuntime,
    register_graph_schedule,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import (
    GraphPipelineStage,
    GraphPPStageRuntimeState,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    normalize_graph_pp_microbatch_inputs,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
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
    def test_spmd_runtime_skips_upstream_stage_initialization(self) -> None:
        stage = mock.Mock(spec=GraphPipelineStage)
        stage.num_stages = 1
        stage.group_size = 1
        stage.group_rank = 0
        stage.stage_index = 0
        schedule = _PipelineScheduleRuntime(
            [stage],
            n_microbatches=1,
            loss_fn=mock.Mock(),
            backward_requires_autograd=False,
        )
        schedule.pipeline_order_with_comms = {0: []}
        runtime = GraphRuntime(schedule, is_spmd=True)

        runtime._skip_spmd_stage_initialization(has_backward=True)
        schedule._initialize_stages(
            (torch.ones(1),),
            {},
            target=torch.ones(1),
        )

        stage._prepare_forward_infra.assert_not_called()
        stage._prepare_backward_infra.assert_not_called()
        stage._pre_metadata_inference_backup.assert_not_called()
        self.assertFalse(schedule._p2p_initialized)

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

    def test_compute_annotated_loss_uses_loss_kwargs_and_unwraps_metrics(self) -> None:
        def loss_fn(pred, target, *, global_loss_token_counts):
            return ((pred - target) ** 2).sum() / global_loss_token_counts, {
                "ignored": pred.sum()
            }

        pred = torch.randn(2, 4)
        target = torch.randn(2, 4)
        global_loss_token_counts = torch.tensor(2.0)

        loss = compute_annotated_loss(
            loss_fn,
            pred,
            target,
            {"global_loss_token_counts": global_loss_token_counts},
        )

        self.assertEqual(loss, ((pred - target) ** 2).sum() / global_loss_token_counts)

    def test_prepare_fwd_user_args_allows_absent_args_and_kwargs(self) -> None:
        stage = types.SimpleNamespace(is_first=True, is_last=False)
        ctx = _PipelineContext(types.SimpleNamespace(), None, None, None, [])

        args, kwargs, target = _prepare_fwd_user_args(stage, 0, ctx)

        self.assertEqual(args, ())
        self.assertEqual(kwargs, {})
        self.assertIsNone(target)

    def test_graph_pp_accumulates_grads_only_for_trainable_params(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for param in model[0].parameters():
            param.requires_grad_(False)

        stage = types.SimpleNamespace(
            submod=model,
            state=GraphPPStageRuntimeState(),
        )
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.is_spmd = True
        runner._populate_stage_states(stage)

        self.assertEqual(len(stage.state.sharded_param_values), 4)
        self.assertEqual(len(stage.state.trainable_params), 2)
        self.assertEqual(stage.state.unsharded_param_grads, [])

    def test_joint_handler_threads_returned_grad_handles_between_graphs(self) -> None:
        first_grads: list[torch.Tensor] = []
        repeated_grads: list[torch.Tensor] = []

        def forward_backward_nogradaccum(*args, **kwargs):
            grad = torch.ones(2)
            first_grads.append(grad)
            return torch.tensor(1.0), [grad]

        def forward_backward(*args, grad_accumulators, **kwargs):
            expected_input = repeated_grads[-1] if repeated_grads else first_grads[-1]
            self.assertIs(grad_accumulators[0], expected_input)
            grad_accumulators[0].add_(2)
            returned_grad = grad_accumulators[0].view_as(grad_accumulators[0])
            repeated_grads.append(returned_grad)
            return torch.tensor(2.0), [returned_grad]

        graphs = types.SimpleNamespace(
            forward_backward_nogradaccum=forward_backward_nogradaccum,
            forward_backward=forward_backward,
        )
        stage = types.SimpleNamespace(
            stage_index=0,
            is_first=True,
            is_last=True,
            has_backward=True,
            _runtime_validate=False,
            state=GraphPPStageRuntimeState(
                unsharded_param_values=[object()],
                buffer_values=[],
            ),
            output_chunks=[],
        )
        schedule = types.SimpleNamespace(
            _stages=[stage],
            backward_counter={0: 0},
            _internal_losses=[],
        )
        ctx = types.SimpleNamespace(
            arg_mbs=[(object(),), (object(),)],
            kwarg_mbs=[{}, {}],
            target_mbs=[object(), object()],
        )
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = schedule
        runner.stage_graphs = {0: graphs}
        runner.loss_kwargs = {}
        runner._joint_gradient_accumulation_stage_indices = {0}
        runner.ensure_ready = mock.Mock()

        runner._handle_forward_backward(
            _Action(0, FORWARD_BACKWARD_NOGRADACCUM, 0),
            ctx,
        )
        accumulator = stage.state.unsharded_param_grads[0]
        runner._handle_forward_backward(
            _Action(0, FORWARD_BACKWARD, 1),
            ctx,
        )

        self.assertIs(stage.state.unsharded_param_grads[0], repeated_grads[-1])
        self.assertEqual(
            stage.state.unsharded_param_grads[0].data_ptr(),
            accumulator.data_ptr(),
        )
        runner._handle_forward_backward(
            _Action(0, FORWARD_BACKWARD, 1),
            ctx,
        )
        self.assertIs(stage.state.unsharded_param_grads[0], repeated_grads[-1])
        self.assertIsNot(repeated_grads[-1], repeated_grads[-2])
        self.assertEqual(accumulator.tolist(), [5.0, 5.0])
        self.assertEqual(schedule.backward_counter[0], 3)
        self.assertEqual(len(schedule._internal_losses), 3)
        self.assertEqual(len(stage.output_chunks), 3)

        stage.state.unsharded_param_grads = []
        runner._handle_forward_backward(
            _Action(0, FORWARD_BACKWARD_NOGRADACCUM, 0),
            ctx,
        )
        self.assertIsNot(stage.state.unsharded_param_grads[0], accumulator)

    def test_joint_handler_publishes_single_microbatch_gradients(self) -> None:
        param = nn.Parameter(torch.zeros(2))
        param_grad = torch.ones(2)
        graphs = types.SimpleNamespace(
            forward_backward=lambda *args, **kwargs: (
                torch.tensor(1.0),
                [param_grad],
            ),
            reduce_grads=lambda grads, **kwargs: grads,
            param_grads_for_accumulation=lambda grads: grads,
        )
        stage = types.SimpleNamespace(
            stage_index=0,
            is_first=True,
            is_last=True,
            has_backward=True,
            _runtime_validate=False,
            state=GraphPPStageRuntimeState(
                unsharded_param_values=[object()],
                buffer_values=[],
                trainable_params=[param],
            ),
            output_chunks=[],
        )
        schedule = types.SimpleNamespace(
            _stages=[stage],
            _n_microbatches=1,
            scale_grads=False,
            backward_counter={0: 0},
            _internal_losses=[],
            pipeline_order_with_comms={0: [_Action(0, FULL_FORWARD_BACKWARD, 0)]},
        )
        ctx = types.SimpleNamespace(
            arg_mbs=[(object(),)],
            kwarg_mbs=[{}],
            target_mbs=[object()],
        )
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = schedule
        runner.stage_graphs = {0: graphs}
        runner.loss_kwargs = {}
        runner._joint_gradient_accumulation_stage_indices = set()
        runner.ensure_ready = mock.Mock()

        runner._handle_forward_backward(
            _Action(0, FULL_FORWARD_BACKWARD, 0),
            ctx,
        )

        torch.testing.assert_close(param.grad, param_grad)
        self.assertIsNot(param.grad, param_grad)

    def test_backward_action_controls_gradient_accumulation_mode(self) -> None:
        param = nn.Parameter(torch.zeros(2))
        stage = types.SimpleNamespace(
            state=GraphPPStageRuntimeState(trainable_params=[param]),
            _runtime_validate=False,
        )
        graphs = types.SimpleNamespace(
            param_grads_for_accumulation=lambda grads: grads,
        )
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = types.SimpleNamespace(_n_microbatches=1, scale_grads=False)

        direct_grad = torch.ones(2)
        runner._accumulate_split_stage_backward_grads(
            stage,
            graphs,
            [direct_grad],
            grad_reduction_in_backward=True,
        )
        self.assertTrue(torch.equal(param.grad, direct_grad))
        self.assertIsNot(param.grad, direct_grad)
        self.assertEqual(stage.state.unsharded_param_grads, [])

        param.grad = None
        runner.schedule._n_microbatches = 2
        first_unsharded_grad = torch.ones(2)
        runner._accumulate_split_stage_backward_grads(
            stage,
            graphs,
            [first_unsharded_grad],
            grad_reduction_in_backward=False,
        )
        latest_unsharded_grad = first_unsharded_grad.view_as(first_unsharded_grad)
        latest_unsharded_grad.add_(2)
        runner._accumulate_split_stage_backward_grads(
            stage,
            graphs,
            [latest_unsharded_grad],
            grad_reduction_in_backward=False,
        )
        self.assertIs(stage.state.unsharded_param_grads[0], latest_unsharded_grad)
        self.assertTrue(torch.equal(first_unsharded_grad, torch.full((2,), 3.0)))
        self.assertIsNone(param.grad)

    def test_pp_graph_accumulation_reduces_and_commits_once(
        self,
    ) -> None:
        class OneParameterModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2))

        class Graphs:
            def __init__(self) -> None:
                self.num_reductions = 0
                self.reduced_inputs: list[Any] = []

            def reduce_grads(
                self,
                grads: list[Any],
                *,
                runtime_validate: bool = False,
            ) -> list[Any]:
                self.num_reductions += 1
                self.reduced_inputs.append(grads[0])
                return [grads[0].clone()]

            def param_grads_for_accumulation(self, grads: list[Any]) -> list[Any]:
                return grads

        module = OneParameterModule()
        module.weight.grad = torch.full_like(module.weight, 10)
        graphs = Graphs()
        stage = types.SimpleNamespace(
            stage_index=0,
            submod=module,
            graphs=graphs,
            state=GraphPPStageRuntimeState(),
            _runtime_validate=True,
            _graph_pp_grads_scaled=False,
            clear_runtime_states=mock.Mock(),
        )
        schedule = types.SimpleNamespace(
            _stages=[stage],
            _n_microbatches=2,
            scale_grads=True,
        )
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = schedule
        runner.graph_provider = None
        runner.is_spmd = False
        runner.overlap_graphs = {}
        runner.stage_graphs = {}
        runner._graph_pp_ready = False
        runner.loss_kwargs = {}
        runner._dist_moe_forward_context = None
        contributions_by_step = ((2.0, 4.0), (4.0, 6.0))
        returned_aliases: list[torch.Tensor] = []
        step_index = 0

        def schedule_step(*args, **kwargs) -> None:
            nonlocal step_index
            del args, kwargs
            runner.ensure_ready(cast(Any, object()))
            self.assertEqual(stage.state.unsharded_param_grads, [])
            accumulator = torch.full_like(
                module.weight, contributions_by_step[step_index][0]
            )
            runner._accumulate_split_stage_backward_grads(
                cast(Any, stage),
                cast(Any, graphs),
                [accumulator],
                grad_reduction_in_backward=False,
            )
            for contribution in contributions_by_step[step_index][1:]:
                accumulator.add_(contribution)
                returned_alias = accumulator.view_as(accumulator)
                returned_aliases.append(returned_alias)
                runner._accumulate_split_stage_backward_grads(
                    cast(Any, stage),
                    cast(Any, graphs),
                    [returned_alias],
                    grad_reduction_in_backward=False,
                )
            if len(contributions_by_step[step_index]) == 1:
                returned_aliases.append(accumulator)
            runner._handle_reduce_grad(
                _Action(0, REDUCE_GRAD, None),
                cast(Any, object()),
            )
            step_index += 1

        schedule.step = schedule_step

        runner.step()
        torch.testing.assert_close(module.weight.grad, torch.full((2,), 13.0))
        self.assertEqual(graphs.num_reductions, 1)
        self.assertEqual(stage.state.unsharded_param_grads, [])

        runner.step()
        torch.testing.assert_close(module.weight.grad, torch.full((2,), 18.0))
        self.assertEqual(graphs.num_reductions, 2)
        self.assertIs(graphs.reduced_inputs[0], returned_aliases[0])
        self.assertIs(graphs.reduced_inputs[1], returned_aliases[1])
        self.assertEqual(stage.clear_runtime_states.call_count, 2)

    def test_split_block_mask_batch_offset_is_dynamic_for_replay(self) -> None:
        _, kwargs_mbs = normalize_graph_pp_microbatch_inputs(
            [(), ()],
            [
                {"attention_metadata": mask}
                for mask in _split_batch_offset_block_masks()
            ],
        )
        mask0 = kwargs_mbs[0]["attention_metadata"]
        mask1 = kwargs_mbs[1]["attention_metadata"]

        self.assertEqual(_trace_mask_mod_replay(mask0, mask1), (False, True))

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
            config=types.SimpleNamespace(compile=GraphTrainerCompileConfig()),
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
                backward_grad = graph.placeholder("backward_grad")
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
            config=types.SimpleNamespace(compile=GraphTrainerCompileConfig()),
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

    def test_spmd_step_skips_stage_initialization(self) -> None:
        original_split_inputs = object()
        stage = types.SimpleNamespace(
            state=GraphPPStageRuntimeState(),
            graphs=None,
            clear_runtime_states=mock.Mock(),
        )

        class FakeSchedule:
            def __init__(self) -> None:
                self._stages = [stage]
                self._split_inputs = original_split_inputs
                self._stages_forward_initialized = False
                self._stages_backward_initialized = False
                self.step_called = False

            def step(self, *args, **kwargs) -> None:
                self.step_called = True
                if not self._stages_forward_initialized:
                    raise AssertionError("SPMD forward stage was not pre-initialized")
                if not self._stages_backward_initialized:
                    raise AssertionError("SPMD backward stage was not pre-initialized")
                if self._split_inputs is not original_split_inputs:
                    raise AssertionError("GraphPP replaced upstream split inputs")

        schedule = FakeSchedule()
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = schedule
        runner.overlap_graphs = {}
        runner.stage_graphs = {}
        runner.loss_kwargs = {}
        runner._graph_pp_ready = False
        runner.is_spmd = True

        runner.step(torch.ones(2))

        self.assertTrue(schedule.step_called)
        self.assertTrue(schedule._stages_forward_initialized)
        self.assertTrue(schedule._stages_backward_initialized)
        self.assertIs(schedule._split_inputs, original_split_inputs)

    def test_step_failure_clears_transient_gradient_references(self) -> None:
        stage = types.SimpleNamespace(
            state=GraphPPStageRuntimeState(
                unsharded_param_grads=[torch.ones(1)],
            ),
            clear_runtime_states=mock.Mock(),
        )

        class FakeSchedule:
            _stages = [stage]

            def step(self, *args, **kwargs) -> None:
                raise RuntimeError("step failed")

        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = FakeSchedule()
        runner.overlap_graphs = {}
        runner.stage_graphs = {0: object()}
        runner.loss_kwargs = {"stale": object()}
        runner._graph_pp_ready = True
        runner.is_spmd = False

        with self.assertRaisesRegex(RuntimeError, "step failed"):
            runner.step()

        self.assertEqual(stage.state, GraphPPStageRuntimeState())
        stage.clear_runtime_states.assert_called_once_with()
        self.assertEqual(runner.stage_graphs, {})
        self.assertEqual(runner.loss_kwargs, {})
        self.assertFalse(runner._graph_pp_ready)

    def test_eval_forwards_to_schedule_and_clears_runtime_state(self) -> None:
        stage = types.SimpleNamespace(
            state=GraphPPStageRuntimeState(
                sharded_param_values=[object()],
                buffer_values=[object()],
                unsharded_param_values=[object()],
                unsharded_param_grads=[torch.ones(1)],
                sharded_param_grads=[torch.ones(1)],
                trainable_params=[torch.nn.Parameter(torch.ones(1))],
            ),
            clear_runtime_states=mock.Mock(),
        )

        class FakeSchedule:
            def __init__(self) -> None:
                self._stages = [stage]
                self.args = None
                self.kwargs = None

            def eval(self, *args, **kwargs) -> str:
                self.args = args
                self.kwargs = kwargs
                return "eval-result"

        schedule = FakeSchedule()
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = schedule
        runner.overlap_graphs = {}
        runner.stage_graphs = {}
        runner.loss_kwargs = {"stale": object()}
        runner._graph_pp_ready = True
        runner.is_spmd = False

        result = runner.eval(
            torch.ones(2),
            target=torch.ones(2),
            loss_kwargs={"global_loss_token_counts": torch.tensor(2.0)},
        )

        self.assertEqual(result, "eval-result")
        self.assertEqual(len(schedule.args), 1)
        self.assertIn("loss_kwargs", schedule.kwargs)
        self.assertEqual(stage.state, GraphPPStageRuntimeState())
        stage.clear_runtime_states.assert_called_once_with()
        self.assertEqual(runner.loss_kwargs, {})
        self.assertFalse(runner._graph_pp_ready)

    def test_ensure_ready_invokes_provider_before_state_population(self) -> None:
        model = nn.Linear(4, 2)
        stage = types.SimpleNamespace(
            submod=model,
            stage_index=0,
            graphs=None,
            state=GraphPPStageRuntimeState(),
            _graph_pp_grads_scaled=False,
        )
        ctx = types.SimpleNamespace()

        class Provider:
            def __init__(self) -> None:
                self.ctx = None

            def prepare_graphs(
                self,
                schedule,
                provider_ctx,
                *,
                loss_kwargs,
                dist_moe_forward_context,
            ) -> dict[tuple[int, int], object]:
                self.ctx = provider_ctx
                self.dist_moe_forward_context = dist_moe_forward_context
                stage.graphs = types.SimpleNamespace()
                return {}

        provider = Provider()
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = types.SimpleNamespace(_stages=[stage])
        runner.graph_provider = provider
        runner.loss_kwargs = {}
        runner.overlap_graphs = {}
        runner.stage_graphs = {}
        runner._dist_moe_forward_context = mock.Mock()
        runner._graph_pp_ready = False
        runner.is_spmd = False

        runner.ensure_ready(ctx)

        self.assertIs(provider.ctx, ctx)
        self.assertIs(
            provider.dist_moe_forward_context,
            runner._dist_moe_forward_context,
        )
        self.assertTrue(runner._graph_pp_ready)
        self.assertEqual(len(stage.state.sharded_param_values), 2)
        self.assertEqual(stage.state.unsharded_param_grads, [])

    def test_last_stage_forward_leaves_losses_to_upstream_update(self) -> None:
        loss = torch.tensor(1.0)
        stage = types.SimpleNamespace(
            is_last=True,
            stage_index=0,
            output_chunks=[],
            _record_graph_forward=mock.Mock(),
        )
        schedule = types.SimpleNamespace(_internal_losses=[])
        ctx = types.SimpleNamespace(losses=[])

        _post_fwd_common(
            stage,
            0,
            loss,
            (),
            schedule,
            {},
            False,
        )

        self.assertEqual(ctx.losses, [])
        self.assertEqual(schedule._internal_losses, [loss])
        stage._record_graph_forward.assert_called_once_with(0, (loss,), ())

    def test_graph_forward_state_releases_only_transport_ownership(self) -> None:
        """Forward-send release preserves explicit backward graph values."""
        stage = GraphPipelineStage.__new__(GraphPipelineStage)
        stage_base = cast(Any, stage)
        stage.stage_index = 0
        stage.num_stages = 2
        stage.submod = nn.Identity()
        stage.dw_builder = None
        stage.has_backward = True
        stage._retained_output_reason = None
        stage._gradient_reduction_handle = None
        stage_base._forward_chunk_states = {}
        stage.output_chunks = []
        stage._saved_values_for_backward = {}
        stage.saved_values_for_backward_weight_cache = {}

        output = torch.ones(2)
        saved = (torch.full((2,), 2.0),)
        stage._record_graph_forward(0, (output,), saved)
        stage_base._forward_chunk_states[0].send_pending[0] = True
        stage.release_fwd_send_outputs(0)

        self.assertIsNone(stage_base._forward_chunk_states[0].live_outputs[0])
        self.assertIs(stage._saved_values_for_backward[0], saved)
        self.assertIs(stage._take_graph_backward_values(0), saved)
        self.assertEqual(stage_base._forward_chunk_states, {})

        stage._record_graph_forward(1, (output,), saved)
        stage.saved_values_for_backward_weight_cache[1] = saved
        stage.clear_runtime_states()
        self.assertEqual(stage._saved_values_for_backward, {})
        self.assertEqual(stage.saved_values_for_backward_weight_cache, {})

    def test_graph_pp_registers_noop_reduce_grad_wait(self) -> None:
        """Graph reductions do not wait on an eager FSDP reduction handle."""
        stage = GraphPipelineStage.__new__(GraphPipelineStage)
        stage.stage_index = 0
        schedule = types.SimpleNamespace(
            _stages=[stage],
            _n_microbatches=1,
            pipeline_order_with_comms={0: []},
            _comp_type_to_function_map={},
            register_custom_function=mock.Mock(),
        )

        runtime = register_graph_schedule(schedule)

        handlers = {
            call.args[0]: call.args[1]
            for call in schedule.register_custom_function.call_args_list
        }
        self.assertIs(handlers[WAIT_REDUCE_GRAD].__self__, runtime)
        handlers[WAIT_REDUCE_GRAD](mock.Mock(), mock.Mock())

    def test_graph_pp_warns_when_cuda_graph_pass_is_enabled(self) -> None:
        provider = GraphTrainerStageGraphProvider(
            loss_fn=lambda pred, target: (pred.sum(), {}),
            config=types.SimpleNamespace(
                compile=GraphTrainerCompileConfig(enable_passes=True)
            ),
            plan=_make_test_pp_plan(),
        )

        with self.assertWarnsRegex(UserWarning, "use_cuda_graph=False"):
            provider._warn_if_cuda_graph_pass_requested()

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
        self.assertFalse(no_fsdp.extract_fsdp_param_unshard)
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
            )

        self.assertIs(compiled, gm)
        final_inductor_passes.assert_called_once_with(
            compile_config,
            use_cuda_graph=False,
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
            side_effect=lambda gm, *, compile_config, graph_name: gm,
        ) as compile_graph:
            overlap_graphs = _build_graph_pp_overlap_graphs(
                schedule,
                compile_config=full_compile,
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
            )

    def test_multiplexed_graph_copies_backward_meta_to_forward_fake_mode(
        self,
    ) -> None:
        # This is intentionally a focused synthetic metadata test. Real model
        # traces only expose the final Inductor failure; this constructs the
        # exact GraphPP multiplexing case where foreign unbacked base symbols
        # must be copied before derived SymInt expressions.
        import sympy
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        bw_mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
        fw_mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
        bw_shape_sym = bw_mode.shape_env.create_unbacked_symint()
        bw_shape_sym_2 = bw_mode.shape_env.create_unbacked_symint()
        bw_raw_sym = bw_mode.shape_env.create_unbacked_symint()
        bw_raw_sym_2 = bw_mode.shape_env.create_unbacked_symint()
        fw_sym = fw_mode.shape_env.create_unbacked_symint()
        with bw_mode:
            bw_fake = bw_mode.from_tensor(
                torch.empty((bw_shape_sym + bw_shape_sym_2, 4), device="meta")
            )
        with fw_mode:
            fw_fake = fw_mode.from_tensor(torch.empty((fw_sym, 4), device="meta"))

        bw_gm = torch.fx.symbolic_trace(lambda x: (x * 2,))
        fw_gm = torch.fx.symbolic_trace(lambda x: (x + 1,))
        bw_gm.graph.find_nodes(op="placeholder")[0].meta["val"] = bw_fake
        fw_gm.graph.find_nodes(op="placeholder")[0].meta["val"] = fw_fake
        bw_call_node = next(
            node for node in bw_gm.graph.nodes if node.op == "call_function"
        )
        bw_call_node.meta["raw_symints"] = (
            bw_raw_sym + 2 * bw_raw_sym_2,
            bw_raw_sym,
            bw_raw_sym_2,
        )
        bw_call_node.meta["unbacked_bindings"] = {
            bw_raw_sym.node.expr: ("lhs",),
            bw_raw_sym_2.node.expr: ("rhs",),
        }

        multiplexed = multiplex_fw_bw_graph(fw_gm, bw_gm)
        placeholders = multiplexed.graph.find_nodes(op="placeholder")
        bw_meta = placeholders[0].meta["val"]
        fw_meta = placeholders[1].meta["val"]
        multiplexed_call_node = next(
            node for node in multiplexed.graph.nodes if "raw_symints" in node.meta
        )

        self.assertIsInstance(bw_meta, FakeTensor)
        self.assertIsInstance(fw_meta, FakeTensor)
        self.assertIs(bw_meta.fake_mode, fw_meta.fake_mode)
        self.assertIs(
            bw_meta.size()[0].node.shape_env,
            fw_meta.fake_mode.shape_env,
        )
        for symint in multiplexed_call_node.meta["raw_symints"]:
            self.assertIs(symint.node.shape_env, fw_meta.fake_mode.shape_env)
        raw_derived, raw_lhs, raw_rhs = multiplexed_call_node.meta["raw_symints"]
        self.assertEqual(
            len(bw_meta.size()[0].node.expr.free_symbols),
            2,
        )
        self.assertEqual(
            sympy.simplify(
                raw_derived.node.expr - (raw_lhs.node.expr + 2 * raw_rhs.node.expr)
            ),
            0,
        )
        self.assertEqual(
            set(multiplexed_call_node.meta["unbacked_bindings"].keys()),
            {raw_lhs.node.expr, raw_rhs.node.expr},
        )

    def test_multiplexed_graph_errors_on_unsupported_symbolic_meta(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        bw_mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
        fw_mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
        bw_sym = bw_mode.shape_env.create_unbacked_symint()
        fw_sym = fw_mode.shape_env.create_unbacked_symint()
        with fw_mode:
            fw_fake = fw_mode.from_tensor(torch.empty((fw_sym, 4), device="meta"))
        bw_gm = torch.fx.symbolic_trace(lambda x: (x * 2,))
        fw_gm = torch.fx.symbolic_trace(lambda x: (x + 1,))
        fw_gm.graph.find_nodes(op="placeholder")[0].meta["val"] = fw_fake
        bw_call_node = next(
            node for node in bw_gm.graph.nodes if node.op == "call_function"
        )
        bw_call_node.meta["unsupported_symbool"] = bw_sym > 0

        with self.assertRaisesRegex(RuntimeError, "unsupported symbolic metadata"):
            multiplex_fw_bw_graph(fw_gm, bw_gm)

    def test_multiplexed_graph_errors_on_missing_backward_get_attr_remap(self) -> None:
        class BackwardGraphWithAttr(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.ones(2))

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
                return (x + self.scale,)

        fw_gm = torch.fx.symbolic_trace(lambda x: (x + 1,))
        bw_gm = torch.fx.symbolic_trace(BackwardGraphWithAttr())

        with (
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_pp.graph_multiplex."
                "_copy_prefixed_get_attrs",
                return_value={},
            ),
            self.assertRaisesRegex(ValueError, "missing copied get_attr target"),
        ):
            multiplex_fw_bw_graph(fw_gm, bw_gm)

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
        self.assertIsNotNone(graphs.modules.full_bw_first)
        self.assertEqual(graphs.full_bw_grad_accumulator_indices, (0, 1))
        self.assertEqual(
            sum(
                node.target == torch.ops.aten.addmm_.default
                for node in graphs.modules.full_bw_repeat.graph.nodes
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
        _, dw_inputs0 = graphs.backward_input(saved0, (output_grad0,))
        param_grads0 = graphs.backward_weight(dw_inputs0)
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
            stage.graphs.modules.full_bw_repeat,
            [*fw_outputs[1:], output_grad],
        )
        multiplexed = multiplex_fw_bw_graph(
            stage.graphs.modules.fw,
            stage.graphs.modules.full_bw_repeat,
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
        ):
            overlap_graphs = _build_graph_pp_overlap_graphs(
                schedule,
                compile_config=compile_config,
            )

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
