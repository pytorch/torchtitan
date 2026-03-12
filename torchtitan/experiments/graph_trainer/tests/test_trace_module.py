# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.nonstrict_tracer import (
    rewrap_outputs,
    run_traced_module,
    trace_module,
)


def get_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
    )


class TrainStepModule(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, *args):
        *fwd_args, labels = args
        logits = self.model(*fwd_args)
        loss = self.loss_fn(logits, labels)
        params = [p for _, p in self.model.named_parameters(remove_duplicate=False)]
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)


def create_model(config_cls, model_config, device="cuda", dtype=torch.float32):
    model = config_cls(model_config)
    model.to(device=device, dtype=dtype)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device(device))
    return model


class SimpleMLP(nn.Module):
    def __init__(self, dim=64, hidden=128, vocab_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self.embed(x))))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceModule(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128
    NUM_STEPS = 5
    LR = 1e-3

    def setUp(self):
        # Required by torch.use_deterministic_algorithms(True) — cuBLAS needs
        # a larger workspace to select deterministic kernels.
        self._prev_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)
        if self._prev_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self._prev_cublas

    def _make_mlp(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        labels = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        return model, tokens, labels

    def test_mlp_forward(self):
        model, tokens, labels = self._make_mlp()
        traced = trace_module(model, (tokens,))
        out_eager = model(tokens)
        out_traced = run_traced_module(traced, model, (tokens,))
        out_traced = rewrap_outputs(out_traced, traced._output_subclass_metas)
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    def test_mlp_train_step(self):
        model_ref, tokens, labels = self._make_mlp()
        model_copy = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())

        train_step = TrainStepModule(model_ref, get_loss)
        traced = trace_module(train_step, (tokens, labels))

        logits_ref = model_ref(tokens)
        loss_ref = get_loss(logits_ref, labels)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        outputs = run_traced_module(
            traced, TrainStepModule(model_copy, get_loss), (tokens, labels)
        )
        wrapped = rewrap_outputs(outputs, traced._output_subclass_metas)
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref, loss_tr))
        for gr, gt in zip(grads_ref, grads_tr):
            self.assertTrue(torch.equal(gr, gt))

    def test_mlp_multistep_bitwise(self):
        model_ref, tokens, labels = self._make_mlp()
        model_copy = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())

        train_step_ref = TrainStepModule(model_ref, get_loss)
        train_step_copy = TrainStepModule(model_copy, get_loss)
        traced = trace_module(train_step_ref, (tokens, labels))

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=self.LR)
        opt_copy = torch.optim.Adam(model_copy.parameters(), lr=self.LR)

        for step in range(1, self.NUM_STEPS + 1):
            logits_ref = model_ref(tokens)
            loss_ref = get_loss(logits_ref, labels)
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            outputs = run_traced_module(traced, train_step_copy, (tokens, labels))
            wrapped = rewrap_outputs(outputs, traced._output_subclass_metas)
            loss_tr = wrapped[0]
            grads_tr = wrapped[1:]
            for p, g in zip(model_copy.parameters(), grads_tr):
                p.grad = g
            opt_copy.step()
            opt_copy.zero_grad()

            self.assertTrue(
                torch.equal(loss_ref, loss_tr),
                f"Step {step}: loss mismatch",
            )


if __name__ == "__main__":
    unittest.main()
