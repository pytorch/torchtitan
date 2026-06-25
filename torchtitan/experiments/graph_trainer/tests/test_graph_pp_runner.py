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
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.pipelining.schedules import _PipelineContext

from torchtitan.config import ParallelismConfig
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    compute_annotated_loss,
    ensure_boxed_graph_module,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_pp.graph_builder import (
    _build_stage_graphs,
    _compile_graph_pp_module,
    _execute_graph_module,
    GraphTrainerStageGraphProvider,
)
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
    _validate_graph_pp_config,
)

from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    _post_fwd_common,
    _prepare_fwd_user_args,
    GraphPipelineRuntime,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import GraphPPStageRuntimeState
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    normalize_graph_pp_microbatch_inputs,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)


def _build_test_stage_graphs(
    stage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
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


class GraphPipelineRuntimeTraceTest(unittest.TestCase):
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
            graphs=types.SimpleNamespace(num_unsharded_param_grad_values=2),
            state=GraphPPStageRuntimeState(),
        )
        runner = GraphPipelineRuntime.__new__(GraphPipelineRuntime)
        runner.stage_graphs = {0: stage.graphs}
        runner._populate_stage_states(stage)

        self.assertEqual(len(stage.state.flat_param_values), 4)
        self.assertEqual(len(stage.state.trainable_params), 2)
        self.assertEqual(len(stage.state.unsharded_param_grads), 2)

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
            pipeline_order_with_comms={},
        )
        ctx = _PipelineContext(schedule, arg_mbs, kwarg_mbs, None, [])
        provider = GraphTrainerStageGraphProvider(
            loss_fn=lambda pred, target: pred.sum(),
            compile_config=GraphTrainerCompileConfig(enable=False),
            model_config=None,
            parallelism=None,
        )

        with (
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_pp.graph_builder."
                "_build_graph_pp_overlap_graphs",
                return_value={},
            ),
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_pp.graph_builder."
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
        runner = GraphPipelineRuntime.__new__(GraphPipelineRuntime)
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
        runner = GraphPipelineRuntime.__new__(GraphPipelineRuntime)
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
                stage.graphs = types.SimpleNamespace(num_unsharded_param_grad_values=2)
                return {}

        provider = Provider()
        runner = GraphPipelineRuntime.__new__(GraphPipelineRuntime)
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
        self.assertEqual(len(stage.state.unsharded_param_grads), 2)

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

    def test_graph_pp_warns_when_cudagraph_pass_is_enabled(self) -> None:
        provider = GraphTrainerStageGraphProvider(
            loss_fn=lambda pred, target: (pred.sum(), {}),
            compile_config=GraphTrainerCompileConfig(enable=True, enable_passes=True),
            model_config=None,
            parallelism=None,
        )

        with self.assertWarnsRegex(UserWarning, "use_cudagraph=False"):
            provider._warn_if_cudagraph_pass_requested()

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
        compile_config = GraphTrainerCompileConfig(enable=True)

        def boxed_apply_graph_passes(gm, example_inputs, passes, compile_config):
            return ensure_boxed_graph_module(gm)

        with (
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_pp.graph_builder."
                "final_inductor_compile_passes",
                return_value=[],
            ) as final_inductor_passes,
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_pp.graph_builder."
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
            use_cudagraph=False,
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
        x = torch.randn(2, 8, 16, requires_grad=True)
        labels = torch.randint(0, 33, (2, 8))
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
