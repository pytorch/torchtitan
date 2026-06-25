# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest
from unittest import mock

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.pipelining.schedules import (
    _Action,
    BACKWARD_INPUT,
    FORWARD,
    FULL_BACKWARD,
    OVERLAP_F_B,
)

from torchtitan.config import ParallelismConfig
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedCELossWithParamGrads,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_pp import multiplex_fw_bw_graph
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
    _validate_graph_pp_config,
)

from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    _build_graph_pp_multiplexed_graph_bundles,
    _build_stage_graph_bundle,
    _compile_graph_pp_module,
    _require_graph_bundle,
    _run_di_bw_module,
    _run_dw_bw_module,
    _run_full_bw_module,
    _run_fw_module,
    GraphPPRunner,
    GraphPPStageRuntimeState,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import extract_module_state


def _boxed_run(gm: fx.GraphModule, args: list[object]):
    return fx.Interpreter(gm).boxed_run(args)


def _manual_backward_args(stage, saved, *runtime_inputs):
    return [
        *saved,
        *[
            runtime_inputs[index]
            for index in stage.graph_meta.partition.bwd_runtime_input_indices
        ],
    ]


def _make_test_stage(
    submod: nn.Module,
    *,
    is_last: bool,
    loss_fn=None,
    stage_index: int = 0,
    output_grads=None,
    compile_config: GraphTrainerCompileConfig | None = None,
):
    stage = types.SimpleNamespace(
        submod=submod,
        is_last=is_last,
        loss_fn=loss_fn,
        stage_index=stage_index,
        compile_config=compile_config or GraphTrainerCompileConfig(enable_passes=False),
        model_config=None,
        parallelism=None,
    )
    if not is_last:
        if output_grads is None:
            raise ValueError("non-last test stages must provide output_grads metadata")
        stage._stage_meta = types.SimpleNamespace(
            output_grads=pytree.tree_leaves(output_grads)
        )
        stage._to_tensor = lambda meta: meta
    return stage


class GraphPPRunnerTraceTest(unittest.TestCase):
    def test_graph_pp_accumulates_grads_only_for_trainable_params(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for param in model[0].parameters():
            param.requires_grad_(False)

        stage = types.SimpleNamespace(
            submod=model,
            graph_meta=None,
            state=GraphPPStageRuntimeState(),
        )
        runner = GraphPPRunner.__new__(GraphPPRunner)
        runner._populate_stage_states(stage)

        self.assertEqual(len(stage.state.sharded_params), 4)
        self.assertEqual(len(stage.state.trainable_params), 2)
        self.assertEqual(len(stage.state.unsharded_grads), 2)

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

    def test_graph_pp_compile_uses_inductor_compilation_with_default_backend(
        self,
    ) -> None:
        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        for node in gm.graph.find_nodes(op="placeholder"):
            node.meta["val"] = torch.randn(2)

        with (
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_pp.runner."
                "annotate_flex_attention_for_regional_inductor_pass",
                side_effect=lambda gm, example_inputs, flex_compile_config: gm,
            ) as annotate_flex,
            mock.patch(
                "torchtitan.experiments.graph_trainer.graph_pp.runner."
                "regional_inductor_pass",
                side_effect=lambda gm, example_inputs: gm,
            ) as regional_inductor,
        ):
            compiled = _compile_graph_pp_module(
                gm,
                compile_config=GraphTrainerCompileConfig(enable=True),
                graph_name="test_graph",
            )

        self.assertIs(compiled, gm)
        annotate_flex.assert_called_once()
        regional_inductor.assert_called_once()

    def test_full_inductor_overlap_builds_multiplexed_graph(self) -> None:
        full_compile = GraphTrainerCompileConfig(
            enable=True,
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
        _build_stage_graph_bundle(stage0, (x,), {}, None, {})
        _build_stage_graph_bundle(stage1, (x,), {}, None, {})
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
                            _Action(1, FULL_BACKWARD, 0, None),
                        ),
                    )
                ]
            },
        )

        with mock.patch(
            "torchtitan.experiments.graph_trainer.graph_pp.runner."
            "_compile_graph_pp_module",
            side_effect=lambda gm, *, compile_config, graph_name: gm,
        ) as compile_graph:
            _build_graph_pp_multiplexed_graph_bundles(schedule)

        self.assertIn((0, 1), schedule._graph_pp_multiplexed_graphs)
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
            _build_graph_pp_multiplexed_graph_bundles(schedule)

    def test_multiplexed_graph_copies_backward_meta_to_forward_fake_mode(
        self,
    ) -> None:
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

        _build_stage_graph_bundle(stage, (x,), {}, None, {})

        state = [*model.parameters()]
        output, saved = _run_fw_module(
            stage.graph_callables.fw, stage.graph_meta, [*state, x]
        )
        self.assertTrue(torch.allclose(output, model(x)))

        input_grads, param_grads = _run_full_bw_module(
            stage.graph_callables.full_bw,
            stage.graph_meta,
            [*saved, output_grad],
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

        di_grads, dw_inputs = _run_di_bw_module(
            stage.graph_callables.bw_di,
            stage.graph_meta,
            [*saved, output_grad],
        )
        dw_grads = _run_dw_bw_module(
            stage.graph_callables.bw_dw,
            stage.graph_meta,
            dw_inputs,
        )
        for actual, expected in zip(dw_grads + di_grads, expected_grads, strict=True):
            self.assertTrue(torch.allclose(actual, expected))

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

        _build_stage_graph_bundle(stage, (x,), {}, None, {})

        self.assertTrue(torch.equal(model.tokens, initial_tokens))
        self.assertGreater(stage.graph_meta.partition.num_fwd_side_effect_outputs, 0)

        state = list(extract_module_state(model).values())
        _run_fw_module(
            stage.graph_callables.fw,
            stage.graph_meta,
            [*state, x],
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
        _build_stage_graph_bundle(stage, (x,), {}, None, {})

        state = [*model.parameters()]
        fw_outputs = _boxed_run(
            stage.graph_callables.fw,
            [*state, x],
        )
        bw_outputs = _boxed_run(
            stage.graph_callables.full_bw,
            [*fw_outputs[1:], output_grad],
        )
        multiplexed = multiplex_fw_bw_graph(
            stage.graph_callables.fw,
            stage.graph_callables.full_bw,
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
        _build_stage_graph_bundle(stage0, (x,), {}, None, {})
        _build_stage_graph_bundle(stage1, (x,), {}, None, {})
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
                            _Action(1, FULL_BACKWARD, 0, None),
                        ),
                    )
                ]
            },
        )

        _build_graph_pp_multiplexed_graph_bundles(schedule)

        self.assertIn((0, 1), schedule._graph_pp_multiplexed_graphs)

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

        _build_stage_graph_bundle(
            stage,
            (x,),
            {},
            target,
            {"global_valid_tokens": global_valid_tokens},
        )

        state = [*model.parameters()]
        loss, saved = _run_fw_module(
            stage.graph_callables.fw,
            stage.graph_meta,
            [*state, x, target, global_valid_tokens],
        )
        expected_loss = loss_fn(model(x), target, global_valid_tokens)
        self.assertTrue(torch.allclose(loss, expected_loss))

        input_grads, param_grads = _run_full_bw_module(
            stage.graph_callables.full_bw,
            stage.graph_meta,
            _manual_backward_args(stage, saved, torch.ones_like(loss)),
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
        loss_fn = ChunkedCELossWithParamGrads(
            ChunkedCELossWithParamGrads.Config(num_chunks=4)
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

        _build_stage_graph_bundle(
            stage,
            (x,),
            {},
            labels,
            {"global_valid_tokens": global_valid_tokens},
        )

        state = [*model.parameters()]
        loss, saved = _run_fw_module(
            stage.graph_callables.fw,
            stage.graph_meta,
            [*state, x, labels, global_valid_tokens],
        )
        input_grads, param_grads = _run_full_bw_module(
            stage.graph_callables.full_bw,
            stage.graph_meta,
            _manual_backward_args(stage, saved, torch.ones_like(loss)),
        )

        expected_loss = loss_fn(model(x), labels, global_valid_tokens)
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

    def test_stage_forward_requires_prebuilt_graph_bundle(self) -> None:
        stage = types.SimpleNamespace(
            stage_index=0,
            graph_callables=None,
            graph_meta=None,
        )

        with self.assertRaisesRegex(ValueError, "must be built before runtime"):
            _require_graph_bundle(stage, "FORWARD")

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
        _build_stage_graph_bundle(stage, (x,), {}, None, {})

        for gm, callable_name, action_name in (
            (stage.graph_callables.fw, "fw", "FORWARD"),
            (stage.graph_callables.full_bw, "full_bw", "FULL_BACKWARD"),
        ):
            for node in gm.graph.nodes:
                self.assertEqual(node.meta["graph_pp_stage_index"], 7)
                self.assertEqual(node.meta["graph_pp_callable"], callable_name)
                self.assertEqual(node.meta["graph_pp_action"], action_name)


if __name__ == "__main__":
    unittest.main()
