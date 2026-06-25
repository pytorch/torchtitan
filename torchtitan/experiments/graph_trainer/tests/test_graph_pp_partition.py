# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.fx as fx
import torch.nn as nn

from torchtitan.experiments.graph_trainer.graph_pp import partition_joint_graph
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
)
from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step


class _TinyTrainStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _loss_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    global_valid_tokens: torch.Tensor,
) -> torch.Tensor:
    return (pred - target).pow(2).sum() / global_valid_tokens


def _make_stage_step(model: nn.Module):
    def stage_step(x: torch.Tensor, output_grad: torch.Tensor):
        out = model(x)
        params = [
            p
            for _, p in model.named_parameters(remove_duplicate=False)
            if p.requires_grad
        ]
        grads = torch.autograd.grad(out, [*params, x], grad_outputs=output_grad)
        return [out, *grads]

    return stage_step


def _trace_train_step(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    global_valid_tokens: torch.Tensor,
):
    return minimal_fx_tracer(make_fwd_bwd_step(model, _loss_fn), module=model)(
        x,
        target,
        global_valid_tokens,
        {},
    )


def _flat_runtime_inputs(model: nn.Module, *user_inputs: torch.Tensor) -> list[object]:
    model_state = extract_module_state(model)
    return [*model_state.values(), *user_inputs]


def _boxed_run(gm: fx.GraphModule, args: list[object]):
    return fx.Interpreter(gm).boxed_run(args)


def _flat_args_from_sources(sources, flat_inputs: list[object]) -> list[object]:
    return [flat_inputs[source.index] for source in sources]


class GraphPPPartitionTest(unittest.TestCase):
    def test_partition_matches_joint_graph(self) -> None:
        torch.manual_seed(0)
        model = _TinyTrainStep()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        global_valid_tokens = torch.tensor(2.0)
        traced = _trace_train_step(model, x, target, global_valid_tokens)

        fw_module, bw_module, meta = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
        )

        flat_inputs = _flat_runtime_inputs(model, x, target, global_valid_tokens)
        joint_outputs = traced.gm(*flat_inputs)

        fw_args = _flat_args_from_sources(
            meta.fwd_input_sources,
            flat_inputs,
        )
        fw_outputs = _boxed_run(fw_module, fw_args)
        self.assertEqual(fw_args, [])
        self.assertTrue(torch.equal(fw_outputs[0], joint_outputs[0]))

        bw_args = list(fw_outputs[1:])
        bw_outputs = _boxed_run(bw_module, bw_args)
        self.assertEqual(bw_args, [])
        self.assertEqual(len(bw_outputs), len(joint_outputs) - 1)
        for actual, expected in zip(bw_outputs, joint_outputs[1:], strict=True):
            self.assertTrue(torch.equal(actual, expected))

    def test_metadata_describes_calling_convention(self) -> None:
        model = _TinyTrainStep()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        global_valid_tokens = torch.tensor(2.0)
        traced = _trace_train_step(model, x, target, global_valid_tokens)

        _, _, meta = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
        )

        self.assertEqual(meta.num_fwd_user_outputs, 1)
        self.assertEqual(meta.num_fwd_inputs, len(traced.example_inputs))
        self.assertEqual(meta.num_saved_for_backward, meta.num_bwd_inputs)
        self.assertEqual(meta.saved_for_backward_names, meta.bwd_input_names)
        self.assertEqual(
            [source.kind for source in meta.fwd_input_sources],
            ["flat_input"] * len(meta.fwd_input_sources),
        )
        self.assertEqual(
            [source.index for source in meta.fwd_input_sources],
            list(range(len(traced.example_inputs))),
        )

    def test_partition_preserves_node_metadata(self) -> None:
        model = _TinyTrainStep()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        global_valid_tokens = torch.tensor(2.0)
        traced = _trace_train_step(model, x, target, global_valid_tokens)

        marker_key = "graph_pp_test_marker"
        for node in traced.gm.graph.nodes:
            if node.op == "call_function":
                node.meta[marker_key] = "kept"
                break
        else:
            self.fail("Expected at least one call_function node")

        fw_module, bw_module, _ = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
        )

        self.assertTrue(
            any(
                node.meta.get(marker_key) == "kept"
                for node in [
                    *fw_module.graph.nodes,
                    *bw_module.graph.nodes,
                ]
            )
        )

    def test_partition_keeps_backward_only_tangent_out_of_forward(self) -> None:
        model = _TinyTrainStep()
        x = torch.randn(2, 4, requires_grad=True)
        output_grad = torch.randn(2, 3)
        traced = minimal_fx_tracer(_make_stage_step(model), module=model)(
            x,
            output_grad,
        )

        fw_module, bw_module, meta = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced.example_inputs) - 1,),
        )

        self.assertNotIn(
            meta.bwd_runtime_input_names[0],
            [source.name for source in meta.fwd_input_sources],
        )
        self.assertEqual(meta.num_bwd_runtime_inputs, 1)
        self.assertEqual(
            meta.num_bwd_inputs,
            meta.num_saved_for_backward + meta.num_bwd_runtime_inputs,
        )

        flat_inputs = _flat_runtime_inputs(model, x, output_grad)
        fw_args = _flat_args_from_sources(
            meta.fwd_input_sources,
            flat_inputs,
        )
        fw_outputs = _boxed_run(fw_module, fw_args)
        self.assertEqual(fw_args, [])
        self.assertEqual(len(fw_outputs), meta.num_fwd_user_outputs + 2)

        bw_args = [*fw_outputs[1:], output_grad]
        bw_outputs = _boxed_run(bw_module, bw_args)
        self.assertEqual(bw_args, [])
        joint_outputs = traced.gm(*_flat_runtime_inputs(model, x, output_grad))
        for actual, expected in zip(bw_outputs, joint_outputs[1:], strict=True):
            self.assertTrue(torch.equal(actual, expected))

    def test_partition_saves_backward_passthrough_placeholders(self) -> None:
        def stage_step(
            x: torch.Tensor,
            output_metadata: torch.Tensor,
            output_grad: torch.Tensor,
        ):
            out = x.sin()
            (grad_x,) = torch.autograd.grad(
                out,
                x,
                grad_outputs=output_grad,
            )
            return [out, grad_x, output_metadata]

        x = torch.randn(2, 4, requires_grad=True)
        output_metadata = torch.arange(2)
        output_grad = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(
            x,
            output_metadata,
            output_grad,
        )

        fw_module, bw_module, meta = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced.example_inputs) - 1,),
        )

        flat_inputs = [x, output_metadata, output_grad]
        fw_args = _flat_args_from_sources(meta.fwd_input_sources, flat_inputs)
        fw_outputs = _boxed_run(fw_module, fw_args)
        self.assertIn(
            "arg1_1",
            meta.saved_for_backward_names,
            "Backward passthrough metadata should be carried through forward.",
        )
        self.assertNotIn("arg2_1", [source.name for source in meta.fwd_input_sources])

        runtime_inputs = [output_grad]
        runtime_by_name = dict(
            zip(
                meta.bwd_runtime_input_names,
                [runtime_inputs[index] for index in meta.bwd_runtime_input_indices],
                strict=True,
            )
        )
        saved_by_name = dict(
            zip(meta.saved_for_backward_names, fw_outputs[1:], strict=True)
        )
        bw_args = [
            saved_by_name[name] if name in saved_by_name else runtime_by_name[name]
            for name in meta.bwd_input_names
        ]
        bw_outputs = _boxed_run(bw_module, bw_args)
        joint_outputs = traced.gm(*flat_inputs)
        for actual, expected in zip(bw_outputs, joint_outputs[1:], strict=True):
            self.assertTrue(torch.equal(actual, expected))

    def test_invalid_forward_output_count_raises(self) -> None:
        model = _TinyTrainStep()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        global_valid_tokens = torch.tensor(2.0)
        traced = _trace_train_step(model, x, target, global_valid_tokens)

        with self.assertRaisesRegex(ValueError, "num_fwd_outputs"):
            partition_joint_graph(
                traced,
                num_fwd_outputs=0,
            )


if __name__ == "__main__":
    unittest.main()
