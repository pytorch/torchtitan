# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from types import SimpleNamespace

import torch
import torch.fx as fx
import torch.nn as nn

from torchtitan.experiments.graph_trainer.graph_pp import (
    partition_joint_graph,
    split_di_dw_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_di_dw import (
    _collect_saved_values_for_dw,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
)
from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step


def _mesh_op() -> object:
    """Stand-in target for a single-output non-tensor op (e.g. mesh PG getter)."""


def _mesh_consumer(_pg: object) -> object:
    """Stand-in target for a direct (non-getitem) consumer of a mesh op."""


def _multi_output_op() -> tuple:
    """Stand-in target for a tuple-returning non-tensor op."""


class _TinyStage(nn.Module):
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


def _make_last_stage_step_with_input_grad(model: nn.Module):
    def train_step(
        x: torch.Tensor,
        target: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ):
        out = model(x)
        loss = _loss_fn(out, target, global_valid_tokens)
        params = [
            p
            for _, p in model.named_parameters(remove_duplicate=False)
            if p.requires_grad
        ]
        grads = torch.autograd.grad(loss, [*params, x])
        return [loss, *grads]

    return train_step


def _flat_runtime_inputs(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    global_valid_tokens: torch.Tensor,
):
    return [*extract_module_state(model).values(), x, target, global_valid_tokens]


def _boxed_run(gm: fx.GraphModule, args: list[object]):
    return fx.Interpreter(gm).boxed_run(args)


class GraphPPSplitDiDwTest(unittest.TestCase):
    def test_split_di_dw_reconstructs_full_backward(self) -> None:
        torch.manual_seed(0)
        model = _TinyStage()
        x = torch.randn(2, 4, requires_grad=True)
        target = torch.randn(2, 3)
        global_valid_tokens = torch.tensor(2.0)
        traced = minimal_fx_tracer(
            _make_last_stage_step_with_input_grad(model),
            module=model,
        )(x, target, global_valid_tokens)
        fw_module, bw_module, _ = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
        )

        split = split_di_dw_graph(
            bw_module,
            num_param_grads=2,
        )
        self.assertIsNotNone(split)
        assert split is not None

        fw_outputs = _boxed_run(
            fw_module,
            _flat_runtime_inputs(model, x, target, global_valid_tokens),
        )
        full_bw_outputs = _boxed_run(
            bw_module,
            list(fw_outputs[1:]),
        )

        di_outputs = _boxed_run(split.bw_di_module, list(fw_outputs[1:]))
        input_grads = di_outputs[: split.num_input_grads]
        dw_live_ins = di_outputs[split.num_input_grads :]
        dw_outputs = _boxed_run(split.bw_dw_module, list(dw_live_ins))

        self.assertEqual(split.num_input_grads, 1)
        self.assertEqual(len(dw_outputs), 2)
        for actual, expected in zip(dw_outputs, full_bw_outputs[:2], strict=True):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(input_grads[0], full_bw_outputs[2]))

    def test_first_stage_no_input_grads_returns_none(self) -> None:
        model = _TinyStage()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        global_valid_tokens = torch.tensor(2.0)
        traced = minimal_fx_tracer(
            make_fwd_bwd_step(model, _loss_fn),
            module=model,
        )(x, target, global_valid_tokens, {})
        _, bw_module, _ = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
        )

        split = split_di_dw_graph(
            bw_module,
            num_param_grads=2,
        )

        self.assertIsNone(split)


class GraphPPCollectSavedValuesForDwTest(unittest.TestCase):
    """Directly exercise the non-tensor branch of _collect_saved_values_for_dw.

    Under compile-on-one-rank, the backward graph carries non-tensor mesh/coor
    nodes (no ``tensor_meta``, no fake-tensor ``val``) that cross the dI/dW
    boundary. The function recomputes single-output ones in the dW graph and
    saves the getitem results of tuple-returning ones. These cases need CooR to
    appear end-to-end, so this test builds them synthetically on CPU.
    """

    @staticmethod
    def _build_graphs():
        # _collect_saved_values_for_dw only reads bw_gm.graph, so a SimpleNamespace
        # with a .graph attribute avoids GraphModule codegen of the stub targets.
        bw = fx.Graph()
        mesh = bw.create_node("call_function", _mesh_op, name="single_mesh_op")
        bw.create_node("call_function", _mesh_consumer, args=(mesh,), name="mesh_user")
        multi = bw.create_node("call_function", _multi_output_op, name="multi_op")
        gi0 = bw.create_node("call_function", operator.getitem, args=(multi, 0))
        gi1 = bw.create_node("call_function", operator.getitem, args=(multi, 1))
        bw.output((mesh, gi0, gi1))

        # The dI graph references both producers, so both cross the boundary.
        di = fx.Graph()
        di.create_node("placeholder", "single_mesh_op", name="single_mesh_op")
        di.create_node("placeholder", "multi_op", name="multi_op")
        di.output(())
        return SimpleNamespace(graph=bw), di, gi0.name, gi1.name

    def test_single_output_node_is_recomputed_multi_output_is_saved(self) -> None:
        bw, di, gi0_name, gi1_name = self._build_graphs()
        saved_values, saved_sym_nodes = _collect_saved_values_for_dw(bw, di)
        saved_names = {node.name for node in saved_values}

        # The single-output mesh op is recomputed in the dW graph, not saved.
        self.assertNotIn("single_mesh_op", saved_names)
        # The tuple-returning op's getitem results are saved across the boundary.
        self.assertEqual(saved_names, {gi0_name, gi1_name})
        self.assertEqual(saved_sym_nodes, [])


if __name__ == "__main__":
    unittest.main()
