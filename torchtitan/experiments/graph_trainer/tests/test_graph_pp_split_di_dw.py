# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.fx as fx
import torch.nn as nn

from torchtitan.experiments.graph_trainer.graph_pp import (
    partition_joint_graph,
    split_di_dw_graph,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
)
from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step


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


if __name__ == "__main__":
    unittest.main()
