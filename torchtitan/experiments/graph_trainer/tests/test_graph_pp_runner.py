# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest
from typing import Any
from unittest import mock

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    OVERLAP_F_B,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
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
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_builder import (
    _build_graph_pp_overlap_graphs,
    _build_joint_stage_graph,
    _build_stage_graphs,
    _compile_graph_pp_module,
    _execute_graph_module,
    GraphTrainerJointStageGraphs,
    GraphTrainerScheduledJointStageGraphs,
    GraphTrainerStageGraphProvider,
    GraphTrainerStageGraphs,
)
from torchtitan.experiments.graph_trainer.graph_pp import multiplex_fw_bw_graph
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
    _make_spmd_runtime_schedule,
    _set_graph_backward_actions,
    _validate_graph_pp_config,
    make_graph_runtime,
    resolve_graph_runtime_fsdp_policy,
    resolve_graph_runtime_gradient_accumulation_policy,
)

from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    _grad_reduction_runs_in_backward,
    _post_fwd_common,
    _prepare_fwd_user_args,
    BACKWARD,
    BACKWARD_WEIGHT_WITH_REDUCE_GRAD,
    BACKWARD_WITH_REDUCE_GRAD,
    FULL_FORWARD_BACKWARD,
    GraphRuntime,
    ZERO_GRAD_ACCUMS,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import GraphPPStageRuntimeState
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    normalize_graph_pp_microbatch_inputs,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
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
    accumulate_gradients_in_graph: bool = False,
    fuse_wgrad_accumulation: bool = False,
) -> None:
    _build_stage_graphs(
        stage,
        args,
        kwargs,
        target,
        loss_kwargs,
        loss_fn=stage.loss_fn,
        compile_config=stage.compile_config,
        model_config=stage.model_config,
        parallelism=stage.parallelism,
        compile_graphs=compile_graphs,
        extract_fsdp_param_unshard=extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
        accumulate_gradients_in_graph=accumulate_gradients_in_graph,
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
        model_config=None,
        parallelism=None,
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

    def test_compute_annotated_loss_uses_loss_kwargs_and_unwraps_metrics(self) -> None:
        def loss_fn(pred, target, *, global_valid_tokens):
            return ((pred - target) ** 2).sum() / global_valid_tokens, {
                "ignored": pred.sum()
            }

        pred = torch.randn(2, 4)
        target = torch.randn(2, 4)
        global_valid_tokens = torch.tensor(2.0)

        loss = compute_annotated_loss(
            loss_fn,
            pred,
            target,
            {"global_valid_tokens": global_valid_tokens},
        )

        self.assertEqual(loss, ((pred - target) ** 2).sum() / global_valid_tokens)

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
            stage_index=0,
            graphs=types.SimpleNamespace(zero_grad_=lambda: []),
            state=GraphPPStageRuntimeState(),
        )
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.stage_graphs = {0: stage.graphs}
        runner._populate_stage_states(stage)
        runner._initialize_split_grad_accumulators(stage, [object(), object()])

        self.assertEqual(len(stage.state.flat_param_values), 4)
        self.assertEqual(len(stage.state.trainable_params), 2)
        self.assertEqual(len(stage.state.unsharded_param_grads), 2)

    def test_zero_grad_accums_action_resets_graph_owned_buffers(self) -> None:
        model = nn.Linear(2, 2)
        accumulator = torch.ones_like(model.weight)

        def zero_grad_() -> list[torch.Tensor]:
            accumulator.zero_()
            return [accumulator]

        stage = types.SimpleNamespace(
            submod=model,
            stage_index=0,
            graphs=types.SimpleNamespace(zero_grad_=zero_grad_),
            state=GraphPPStageRuntimeState(),
            _graph_pp_grads_scaled=False,
        )
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = types.SimpleNamespace(_stages=[stage])
        runner.graph_provider = None
        runner.overlap_graphs = {}
        runner.stage_graphs = {}
        runner.loss_kwargs = {}
        runner._graph_pp_ready = False

        runner._handle_zero_grad_accums(
            _Action(0, ZERO_GRAD_ACCUMS),
            types.SimpleNamespace(),
        )

        self.assertIs(stage.state.unsharded_param_grads[0], accumulator)
        self.assertEqual(torch.count_nonzero(accumulator), 0)

    def test_backward_action_controls_gradient_accumulation_mode(self) -> None:
        param = nn.Parameter(torch.zeros(2))
        stage = types.SimpleNamespace(
            state=GraphPPStageRuntimeState(trainable_params=[param]),
            _runtime_validate=False,
        )
        graphs = types.SimpleNamespace(
            accumulates_gradients_in_graph=False,
            param_grads_for_accumulation=lambda grads: grads,
        )
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = types.SimpleNamespace(_n_microbatches=2, scale_grads=False)

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
        first_unsharded_grad = torch.ones(2)
        runner._accumulate_split_stage_backward_grads(
            stage,
            graphs,
            [first_unsharded_grad],
            grad_reduction_in_backward=False,
        )
        runner._accumulate_split_stage_backward_grads(
            stage,
            graphs,
            [torch.full((2,), 2.0)],
            grad_reduction_in_backward=False,
        )
        self.assertIs(stage.state.unsharded_param_grads[0], first_unsharded_grad)
        self.assertTrue(
            torch.equal(stage.state.unsharded_param_grads[0], torch.full((2,), 3.0))
        )
        self.assertIsNone(param.grad)

    def test_split_block_mask_batch_offset_is_dynamic_for_replay(self) -> None:
        _, kwargs_mbs = normalize_graph_pp_microbatch_inputs(
            [(), ()],
            [{"attention_masks": mask} for mask in _split_batch_offset_block_masks()],
        )
        mask0 = kwargs_mbs[0]["attention_masks"]
        mask1 = kwargs_mbs[1]["attention_masks"]

        self.assertEqual(_trace_mask_mod_replay(mask0, mask1), (False, True))

    def test_existing_stage_graphs_normalize_split_block_masks_in_place(self) -> None:
        arg_mbs = [(), ()]
        kwarg_mbs = [
            {"attention_masks": mask} for mask in _split_batch_offset_block_masks()
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
            compile_config=GraphTrainerCompileConfig(),
            model_config=None,
            parallelism=None,
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
        mask0 = ctx.kwarg_mbs[0]["attention_masks"]
        mask1 = ctx.kwarg_mbs[1]["attention_masks"]

        self.assertEqual(_trace_mask_mod_replay(mask0, mask1), (False, True))

    def test_step_does_not_wrap_upstream_split_inputs(self) -> None:
        original_split_inputs = object()
        stage = types.SimpleNamespace(
            state=GraphPPStageRuntimeState(),
            graphs=None,
        )

        class FakeSchedule:
            def __init__(self) -> None:
                self._stages = [stage]
                self._split_inputs = original_split_inputs
                self.step_called = False

            def step(self, *args, **kwargs) -> None:
                self.step_called = True
                if self._split_inputs is not original_split_inputs:
                    raise AssertionError("GraphPP replaced upstream split inputs")

        schedule = FakeSchedule()
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = schedule
        runner.overlap_graphs = {}
        runner.stage_graphs = {}
        runner.loss_kwargs = {}
        runner._graph_pp_ready = False

        runner.step(torch.ones(2))

        self.assertTrue(schedule.step_called)
        self.assertIs(schedule._split_inputs, original_split_inputs)

    def test_eval_forwards_to_schedule_and_clears_runtime_state(self) -> None:
        stage = types.SimpleNamespace(
            state=GraphPPStageRuntimeState(
                flat_param_values=[object()],
                flat_buffer_values=[object()],
                unsharded_param_values=[object()],
                unsharded_param_grads=[torch.ones(1)],
                sharded_param_grads=[torch.ones(1)],
                trainable_params=[torch.nn.Parameter(torch.ones(1))],
            )
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

        result = runner.eval(
            torch.ones(2),
            target=torch.ones(2),
            loss_kwargs={"global_valid_tokens": torch.tensor(2.0)},
        )

        self.assertEqual(result, "eval-result")
        self.assertEqual(len(schedule.args), 1)
        self.assertIn("loss_kwargs", schedule.kwargs)
        self.assertEqual(stage.state, GraphPPStageRuntimeState())
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
            ) -> dict[tuple[int, int], object]:
                self.ctx = provider_ctx
                stage.graphs = types.SimpleNamespace(zero_grad_=lambda: [])
                return {}

        provider = Provider()
        runner = GraphRuntime.__new__(GraphRuntime)
        runner.schedule = types.SimpleNamespace(_stages=[stage])
        runner.graph_provider = provider
        runner.loss_kwargs = {}
        runner.overlap_graphs = {}
        runner.stage_graphs = {}
        runner._graph_pp_ready = False

        runner.ensure_ready(ctx)

        self.assertIs(provider.ctx, ctx)
        self.assertTrue(runner._graph_pp_ready)
        self.assertEqual(len(stage.state.flat_param_values), 2)
        self.assertEqual(stage.state.unsharded_param_grads, [])

    def test_last_stage_forward_leaves_losses_to_upstream_update(self) -> None:
        loss = torch.tensor(1.0)
        stage = types.SimpleNamespace(
            is_last=True,
            stage_index=0,
            output_chunks=[],
            fwd_cache={},
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

    def test_graph_pp_warns_when_cuda_graph_pass_is_enabled(self) -> None:
        provider = GraphTrainerStageGraphProvider(
            loss_fn=lambda pred, target: (pred.sum(), {}),
            compile_config=GraphTrainerCompileConfig(enable_passes=True),
            model_config=None,
            parallelism=None,
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

    def test_extracted_fsdp_actions_reject_dense_region_overlap(self) -> None:
        compile_config = GraphTrainerCompileConfig(
            enable_fsdp_dense_region_overlap=True
        )
        stage = _make_test_stage(
            nn.Linear(2, 2),
            is_last=True,
            loss_fn=lambda prediction, target: (prediction.sum(), {}),
            compile_config=compile_config,
        )
        for extract_unshard, extract_reduce_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                extract_unshard=extract_unshard,
                extract_reduce_grad=extract_reduce_grad,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "all-gathers and gradient reductions",
                ):
                    _build_test_stage_graphs(
                        stage,
                        (torch.randn(2, 2),),
                        {},
                        None,
                        {},
                        compile_graphs=False,
                        extract_fsdp_param_unshard=extract_unshard,
                        extract_fsdp_grad_reduction=extract_reduce_grad,
                    )

    def test_precompile_rejects_scheduled_joint_graphs(self) -> None:
        parallel_dims = types.SimpleNamespace(pp_enabled=False, fsdp_enabled=False)
        with self.assertRaisesRegex(ValueError, "in-graph gradient accumulation"):
            make_graph_runtime(
                [mock.Mock()],
                num_microbatches=2,
                parallel_dims=parallel_dims,
                parallelism=ParallelismConfig(),
                compile_config=GraphTrainerCompileConfig(
                    precompile_artifact_dir="artifacts"
                ),
                model_config=None,
                loss_fn=mock.Mock(),
                trainer_config=mock.Mock(),
            )

    def test_joint_stage_graphs_bind_runtime_meshes(self) -> None:
        traced = mock.Mock()
        module = nn.Linear(2, 2)
        runtime_meshes = [mock.Mock()]
        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_builder.run_traced"
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
            "torchtitan.experiments.graph_trainer.graph_builder."
            "partition_joint_graph",
            side_effect=AssertionError("SPMD must not partition its joint graph"),
        ):
            _build_joint_stage_graph(
                stage,
                (x,),
                {},
                target,
                {"global_valid_tokens": torch.tensor(8)},
                loss_fn=loss_fn,
                compile_config=compile_config,
                trainer_config=trainer_config,
                parallel_dims=types.SimpleNamespace(),
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
            "torchtitan.experiments.graph_trainer.graph_builder."
            "partition_joint_graph",
            side_effect=AssertionError("SPMD must not partition its joint graph"),
        ):
            _build_joint_stage_graph(
                stage,
                (x,),
                {},
                target,
                {"global_valid_tokens": torch.tensor(8)},
                loss_fn=loss_fn,
                compile_config=compile_config,
                trainer_config=trainer_config,
                parallel_dims=types.SimpleNamespace(),
                accumulate_gradients_in_graph=True,
            )

        self.assertIsInstance(stage.graphs, GraphTrainerScheduledJointStageGraphs)
        graphs = stage.graphs
        flat_params = list(model.parameters())
        flat_buffers = list(model.buffers())
        unsharded_params = graphs.unshard_params(flat_params)
        accumulators = graphs.zero_grad_()
        for _ in range(2):
            graphs.forward_backward(
                (x,),
                {},
                target,
                {"global_valid_tokens": torch.tensor(8)},
                unsharded_param_values=unsharded_params,
                flat_buffer_values=flat_buffers,
                runtime_validate=True,
            )

        expected_loss = ((model(x) - target) ** 2).sum()
        expected_grads = torch.autograd.grad(expected_loss, tuple(model.parameters()))
        for actual, expected in zip(accumulators, expected_grads, strict=True):
            torch.testing.assert_close(actual, 2 * expected)

    def test_spmd_auto_fsdp_collective_placement(self) -> None:
        single_microbatch = resolve_graph_runtime_fsdp_policy(
            GraphTrainerCompileConfig(),
            num_microbatches=1,
            pp_enabled=False,
            fsdp_enabled=True,
        )
        gradient_accumulation = resolve_graph_runtime_fsdp_policy(
            GraphTrainerCompileConfig(),
            num_microbatches=3,
            pp_enabled=False,
            fsdp_enabled=True,
        )
        self.assertFalse(single_microbatch.extract_fsdp_param_unshard)
        self.assertFalse(single_microbatch.extract_fsdp_grad_reduction)
        self.assertTrue(gradient_accumulation.extract_fsdp_param_unshard)
        self.assertTrue(gradient_accumulation.extract_fsdp_grad_reduction)

        schedule = _make_runtime_schedule_mock()
        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline."
            "_new_spmd_runtime_schedule",
            return_value=schedule,
        ):
            _make_spmd_runtime_schedule(
                mock.Mock(),
                num_microbatches=3,
                parallelism=ParallelismConfig(),
                loss_fn=mock.Mock(),
                fsdp_enabled=True,
                extract_fsdp_param_unshard=(
                    gradient_accumulation.extract_fsdp_param_unshard
                ),
                extract_fsdp_grad_reduction=(
                    gradient_accumulation.extract_fsdp_grad_reduction
                ),
                accumulate_gradients_in_graph=False,
            )

        actions = schedule.pipeline_order_with_comms[0]
        self.assertEqual(
            [action.computation_type for action in actions],
            [UNSHARD, FULL_FORWARD_BACKWARD, RESHARD] * 3 + [REDUCE_GRAD],
        )

    def test_gradient_accumulation_policy_auto(self) -> None:
        config = GraphTrainerCompileConfig()

        single_microbatch = resolve_graph_runtime_gradient_accumulation_policy(
            config,
            num_microbatches=1,
            pp_enabled=False,
            fsdp_enabled=False,
            extract_fsdp_grad_reduction=False,
        )
        multiple_microbatches = resolve_graph_runtime_gradient_accumulation_policy(
            config,
            num_microbatches=2,
            pp_enabled=False,
            fsdp_enabled=False,
            extract_fsdp_grad_reduction=False,
        )

        self.assertFalse(single_microbatch.accumulate_in_graph)
        self.assertFalse(single_microbatch.fuse_wgrad_accumulation)
        self.assertTrue(multiple_microbatches.accumulate_in_graph)
        self.assertFalse(multiple_microbatches.fuse_wgrad_accumulation)

        optimized = resolve_graph_runtime_gradient_accumulation_policy(
            GraphTrainerCompileConfig(numerics_changing_optim=True),
            num_microbatches=2,
            pp_enabled=False,
            fsdp_enabled=False,
            extract_fsdp_grad_reduction=False,
        )
        self.assertTrue(optimized.accumulate_in_graph)
        self.assertTrue(optimized.fuse_wgrad_accumulation)

    def test_in_graph_accumulation_does_not_require_wgrad_fusion(self) -> None:
        policy = resolve_graph_runtime_gradient_accumulation_policy(
            GraphTrainerCompileConfig(
                gradient_accumulation_mode="in_graph",
                gradient_accum_in_wgrad_fusion="disabled",
            ),
            num_microbatches=1,
            pp_enabled=False,
            fsdp_enabled=False,
            extract_fsdp_grad_reduction=False,
        )

        self.assertTrue(policy.accumulate_in_graph)
        self.assertFalse(policy.fuse_wgrad_accumulation)

    def test_wgrad_fusion_enables_in_graph_accumulation(self) -> None:
        policy = resolve_graph_runtime_gradient_accumulation_policy(
            GraphTrainerCompileConfig(
                gradient_accum_in_wgrad_fusion="enabled",
            ),
            num_microbatches=1,
            pp_enabled=False,
            fsdp_enabled=False,
            extract_fsdp_grad_reduction=False,
        )

        self.assertTrue(policy.accumulate_in_graph)
        self.assertTrue(policy.fuse_wgrad_accumulation)

    def test_runtime_accumulation_rejects_wgrad_fusion(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires in-graph"):
            resolve_graph_runtime_gradient_accumulation_policy(
                GraphTrainerCompileConfig(
                    gradient_accumulation_mode="runtime",
                    gradient_accum_in_wgrad_fusion="enabled",
                ),
                num_microbatches=2,
                pp_enabled=False,
                fsdp_enabled=False,
                extract_fsdp_grad_reduction=False,
            )

    def test_pipeline_parallel_uses_runtime_gradient_accumulation(self) -> None:
        policy = resolve_graph_runtime_gradient_accumulation_policy(
            GraphTrainerCompileConfig(numerics_changing_optim=True),
            num_microbatches=2,
            pp_enabled=True,
            fsdp_enabled=True,
            extract_fsdp_grad_reduction=True,
        )

        self.assertFalse(policy.accumulate_in_graph)
        self.assertFalse(policy.fuse_wgrad_accumulation)

    def test_pipeline_parallel_rejects_in_graph_gradient_accumulation(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete optimizer step"):
            resolve_graph_runtime_gradient_accumulation_policy(
                GraphTrainerCompileConfig(gradient_accumulation_mode="in_graph"),
                num_microbatches=2,
                pp_enabled=True,
                fsdp_enabled=True,
                extract_fsdp_grad_reduction=True,
            )

    def test_spmd_schedule_has_one_joint_action_per_microbatch(self) -> None:
        schedule = _make_runtime_schedule_mock()
        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.pipeline."
            "_new_spmd_runtime_schedule",
            return_value=schedule,
        ):
            _make_spmd_runtime_schedule(
                mock.Mock(),
                num_microbatches=1,
                parallelism=ParallelismConfig(),
                loss_fn=mock.Mock(),
                fsdp_enabled=False,
                extract_fsdp_param_unshard=False,
                extract_fsdp_grad_reduction=False,
                accumulate_gradients_in_graph=False,
            )

        actions = schedule.pipeline_order_with_comms[0]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].computation_type, FULL_FORWARD_BACKWARD)
        self.assertEqual(actions[0].stage_index, 0)
        self.assertEqual(actions[0].microbatch_index, 0)
        self.assertIsNone(actions[0].sub_actions)

    def test_deferred_spmd_schedule_reduces_once_after_all_microbatches(
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
                num_microbatches=3,
                parallelism=ParallelismConfig(),
                loss_fn=mock.Mock(),
                fsdp_enabled=True,
                extract_fsdp_param_unshard=False,
                extract_fsdp_grad_reduction=True,
                accumulate_gradients_in_graph=True,
            )

        actions = schedule.pipeline_order_with_comms[0]
        action_types = [action.computation_type for action in actions]
        self.assertEqual(
            action_types,
            [ZERO_GRAD_ACCUMS] + [FULL_FORWARD_BACKWARD] * 3 + [REDUCE_GRAD],
        )
        self.assertEqual(
            [action.microbatch_index for action in actions[1:-1]],
            [0, 1, 2],
        )
        self.assertTrue(all(action.sub_actions is None for action in actions[1:-1]))
        self.assertEqual(actions[-1].stage_index, 0)

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

        in_graph_schedule = make_schedule()
        _set_graph_backward_actions(
            in_graph_schedule,
            extract_fsdp_grad_reduction=False,
        )
        in_graph_actions = in_graph_schedule.pipeline_order_with_comms[0]
        self.assertEqual(
            in_graph_actions[0].computation_type,
            BACKWARD_WITH_REDUCE_GRAD,
        )
        self.assertEqual(
            in_graph_actions[1].computation_type,
            BACKWARD_WEIGHT_WITH_REDUCE_GRAD,
        )
        self.assertEqual(
            in_graph_actions[2].sub_actions[1].computation_type,
            BACKWARD_WITH_REDUCE_GRAD,
        )

        self.assertFalse(_grad_reduction_runs_in_backward(deferred_actions[0]))
        self.assertTrue(_grad_reduction_runs_in_backward(in_graph_actions[0]))
        self.assertFalse(_grad_reduction_runs_in_backward(deferred_actions[1]))
        self.assertTrue(_grad_reduction_runs_in_backward(in_graph_actions[1]))

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
                "torchtitan.experiments.graph_trainer.graph_builder."
                "final_inductor_compile_passes",
                return_value=[],
            ) as final_inductor_passes,
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_builder."
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
            "torchtitan.experiments.graph_trainer.graph_builder."
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
            flat_buffer_values=[],
        )
        self.assertTrue(torch.allclose(output, model(x)))

        input_grads, param_grads = stage.graphs.full_backward(
            (output,),
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
            (output,),
            saved,
            (output_grad,),
        )
        dw_grads = stage.graphs.backward_weight(dw_inputs)
        for actual, expected in zip(dw_grads + di_grads, expected_grads, strict=True):
            self.assertTrue(torch.allclose(actual, expected))

    def test_stage_backward_accumulates_into_persistent_buffers(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3, dtype=torch.float64)
        reference = nn.Linear(4, 3, dtype=torch.float64)
        reference.load_state_dict(model.state_dict())
        output_grad = torch.randn(2, 3, dtype=torch.float64)
        stage = _make_test_stage(
            model,
            is_last=False,
            output_grads=output_grad,
        )
        inputs = [
            torch.randn(2, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)
        ]
        _build_test_stage_graphs(
            stage,
            (inputs[0],),
            {},
            None,
            {},
            accumulate_gradients_in_graph=True,
        )

        accumulators = stage.graphs.zero_grad_()
        accumulator_ids = tuple(id(grad) for grad in accumulators)
        for x in inputs:
            output, saved = stage.graphs.forward(
                (x,),
                {},
                None,
                {},
                unsharded_param_values=list(model.parameters()),
                flat_buffer_values=[],
            )
            stage.graphs.full_backward((output,), saved, (output_grad,))
        expected = [torch.zeros_like(param) for param in reference.parameters()]
        for x in inputs:
            grads = torch.autograd.grad(
                reference(x),
                tuple(reference.parameters()),
                grad_outputs=output_grad,
            )
            for accumulated, grad in zip(expected, grads, strict=True):
                accumulated.add_(grad)
        for actual, expected_grad in zip(accumulators, expected, strict=True):
            torch.testing.assert_close(actual, expected_grad)

        reset = stage.graphs.zero_grad_()
        self.assertEqual(tuple(id(grad) for grad in reset), accumulator_ids)
        for grad in reset:
            self.assertEqual(torch.count_nonzero(grad), 0)

    def test_stage_wgrad_annotation_drives_fusion(self) -> None:
        model = nn.Linear(4, 3, dtype=torch.bfloat16)
        output_grad = torch.randn(2, 3, dtype=torch.bfloat16)
        stage = _make_test_stage(
            model,
            is_last=False,
            output_grads=output_grad,
        )
        x = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
        _build_test_stage_graphs(
            stage,
            (x,),
            {},
            None,
            {},
            compile_graphs=False,
            accumulate_gradients_in_graph=True,
            fuse_wgrad_accumulation=True,
        )

        targets = {node.target for node in stage.graphs.modules.full_bw.graph.nodes}
        self.assertIn(torch.ops.aten.addmm_.default, targets)

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
                num_flat_param_values=2,
                num_fw_param_inputs=1,
                fwd_input_names=("unsharded_weight", "x"),
                fwd_flat_input_indices=(2,),
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
            flat_buffer_values=[],
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
            flat_buffer_values=[model.tokens],
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
        x = torch.randn(2, 4, requires_grad=True)
        stage0_mod = nn.Linear(4, 3)
        stage1_mod = nn.Linear(4, 3)
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
            output_grads=torch.empty_like(stage1_mod(x)),
        )
        _build_test_stage_graphs(stage0, (x,), {}, None, {}, compile_graphs=False)
        _build_test_stage_graphs(stage1, (x,), {}, None, {}, compile_graphs=False)
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

        overlap_graphs = _build_graph_pp_overlap_graphs(
            schedule,
            compile_config=compile_config,
        )

        self.assertIn((0, 1), overlap_graphs)

    def test_last_stage_graphs_return_loss_and_input_grad(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3)

        def loss_fn(pred, target, global_valid_tokens):
            return ((pred - target) ** 2).sum() / global_valid_tokens

        stage = _make_test_stage(
            model,
            is_last=True,
            loss_fn=loss_fn,
            stage_index=1,
        )
        x = torch.randn(2, 4, requires_grad=True)
        target = torch.randn(2, 3)
        global_valid_tokens = torch.tensor(2.0)

        _build_test_stage_graphs(
            stage,
            (x,),
            {},
            target,
            {"global_valid_tokens": global_valid_tokens},
        )
        self.assertEqual(stage.graphs.meta.partition.backward_grad_input_names, ())
        self.assertEqual(stage.graphs.meta.partition.backward_grad_input_indices, ())

        state = list(model.parameters())
        loss, saved = stage.graphs.forward(
            (x,),
            {},
            target,
            {"global_valid_tokens": global_valid_tokens},
            unsharded_param_values=state,
            flat_buffer_values=[],
        )
        expected_loss = loss_fn(model(x), target, global_valid_tokens)
        self.assertTrue(torch.allclose(loss, expected_loss))

        input_grads, param_grads = stage.graphs.full_backward(
            (loss,),
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
        global_valid_tokens = torch.tensor(float(labels.numel()))

        _build_test_stage_graphs(
            stage,
            (x,),
            {},
            labels,
            {"global_valid_tokens": global_valid_tokens},
        )

        state = list(model.parameters())
        loss, saved = stage.graphs.forward(
            (x,),
            {},
            labels,
            {"global_valid_tokens": global_valid_tokens},
            unsharded_param_values=state,
            flat_buffer_values=[],
        )
        input_grads, param_grads = stage.graphs.full_backward(
            (loss,),
            saved,
            (),
        )

        expected_loss, _ = loss_fn(model(x), labels, global_valid_tokens)
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

    def test_graph_pp_node_metadata_is_annotated(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3)
        x = torch.randn(2, 4, requires_grad=True)
        stage = _make_test_stage(
            model,
            is_last=False,
            loss_fn=None,
            stage_index=7,
            output_grads=torch.empty_like(model(x)),
        )
        _build_test_stage_graphs(stage, (x,), {}, None, {})

        for gm, callable_name, action_name in (
            (stage.graphs.modules.fw, "fw", "FORWARD"),
            (stage.graphs.modules.full_bw, "full_bw", "FULL_BACKWARD"),
        ):
            for node in gm.graph.nodes:
                self.assertEqual(node.meta["graph_pp_stage_index"], 7)
                self.assertEqual(node.meta["graph_pp_callable"], callable_name)
                self.assertEqual(node.meta["graph_pp_action"], action_name)


if __name__ == "__main__":
    unittest.main()
