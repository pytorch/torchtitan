# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bitwise deterministic tests for graph_trainer's aot_fx_trace path.

Tests that running the same model and inputs twice produces bitwise identical
losses and gradients, and that aot_fx_trace matches eager numerics exactly.

Requires a CUDA GPU. Run with:
    pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x
"""

import copy
import unittest

import torch
import torch.nn as nn
from expecttest import assert_expected_inline
from torchtitan.components.loss import cross_entropy_loss
from torchtitan.experiments.graph_trainer.llama3 import model_registry
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    run_traced_module,
    trace_module,
)
from torchtitan.experiments.graph_trainer.trainer import FwdBwdStepModule
from torchtitan.tools.utils import hash_gradient, hash_model

SEED = 42
NUM_STEPS = 5
BATCH_SIZE = 4
SEQ_LEN = 128
VOCAB_SIZE = 2048  # must match llama3 debugmodel config


def _set_deterministic(seed: int = SEED) -> None:
    """Set all random seeds and enable deterministic mode."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_model() -> nn.Module:
    """Build a Llama3 debug model for testing."""
    model_spec = model_registry("debugmodel")
    with torch.device("meta"):
        model = model_spec.model.build()
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_states(buffer_device=None)
    model.train()
    return model


def _build_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic token inputs and labels."""
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device="cuda")
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device="cuda")
    return tokens, labels


def _run_eager_steps(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int = NUM_STEPS,
) -> tuple[torch.Tensor, str, str]:
    """Run eager forward-backward-optimizer steps.

    Returns:
        Tuple of (final_loss, final_model_hash, final_gradient_hash).
    """
    global_valid_tokens = torch.tensor(
        BATCH_SIZE * SEQ_LEN, dtype=torch.float, device="cuda"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(num_steps):
        optimizer.zero_grad()
        pred = model(inputs)
        loss = cross_entropy_loss(pred, labels) / global_valid_tokens
        loss.backward()
        optimizer.step()

    return loss.detach().clone(), hash_model(model), hash_gradient(model)


def _run_aot_fx_trace_steps(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int = NUM_STEPS,
) -> tuple[torch.Tensor, str, str]:
    """Run aot_fx_trace forward-backward-optimizer steps using graph_trainer's make_fx tracer.

    Returns:
        Tuple of (final_loss, final_model_hash, final_gradient_hash).
    """
    global_valid_tokens = torch.tensor(
        BATCH_SIZE * SEQ_LEN, dtype=torch.float, device="cuda"
    )
    fwd_bwd = FwdBwdStepModule(model, cross_entropy_loss)

    # Trace the full fwd+loss+bwd graph
    traced = trace_module(
        fwd_bwd,
        (inputs, labels, global_valid_tokens, {}, {}),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    params = [p for p in model.parameters() if p.requires_grad]

    for _ in range(num_steps):
        optimizer.zero_grad()
        params_and_buffers = {
            **dict(fwd_bwd.named_parameters(remove_duplicate=False)),
            **dict(fwd_bwd.named_buffers(remove_duplicate=False)),
        }
        outputs = run_traced_module(
            traced,
            params_and_buffers,
            (inputs, labels, global_valid_tokens, {}, {}),
        )
        loss = outputs[0]
        grads = outputs[1:]

        for param, grad in zip(params, grads):
            param.grad = grad
        optimizer.step()

    return loss.detach().clone(), hash_model(model), hash_gradient(model)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestBitwiseDeterministic(unittest.TestCase):
    """Test bitwise determinism for graph_trainer's aot_fx_trace path.

    Each test builds a Llama3 debug model, runs forward-backward-optimizer
    steps under deterministic settings, and asserts bitwise equality of
    losses, model hashes, and gradient hashes.
    """

    def setUp(self):
        _set_deterministic()
        self.model1 = _build_model()
        self.model2 = copy.deepcopy(self.model1)
        self.inputs1, self.labels1 = _build_inputs()
        self.inputs2 = self.inputs1.clone()
        self.labels2 = self.labels1.clone()

    def _assert_runs_match(
        self,
        run_a: tuple[torch.Tensor, str, str],
        run_b: tuple[torch.Tensor, str, str],
        msg_prefix: str = "",
    ) -> None:
        loss_a, model_hash_a, grad_hash_a = run_a
        loss_b, model_hash_b, grad_hash_b = run_b

        self.assertTrue(
            torch.equal(loss_a, loss_b),
            f"{msg_prefix}loss mismatch: {loss_a.item()} vs {loss_b.item()}",
        )
        self.assertEqual(model_hash_a, model_hash_b, f"{msg_prefix}model hash mismatch")
        self.assertEqual(
            grad_hash_a, grad_hash_b, f"{msg_prefix}gradient hash mismatch"
        )

    def test_eager_self_deterministic(self):
        """Eager mode: results match hardcoded expected values.

        Run `EXPECTTEST_ACCEPT=1 pytest <this_file> ` to update the inline expected values.
        """
        loss, model_hash, grad_hash = _run_eager_steps(
            self.model1, self.inputs1, self.labels1
        )
        assert_expected_inline(str(loss.item()), """7.961757659912109""")
        assert_expected_inline(
            model_hash,
            """15134607def7232e128240d553c8ee7021a7edbc2ed44d86e927ba61e490b865""",
        )
        assert_expected_inline(
            grad_hash,
            """66bbbbc98b4c1635e42a133ac1fbd499a2b8633ca879f4121cf206708c21dbdf""",
        )

    def test_aot_fx_trace_vs_eager(self):
        """aot_fx_trace and eager produce bitwise identical losses and grads."""
        run_eager = _run_eager_steps(self.model1, self.inputs1, self.labels1)
        run_traced = _run_aot_fx_trace_steps(self.model2, self.inputs2, self.labels2)

        self._assert_runs_match(run_eager, run_traced, "eager vs aot_fx_trace: ")


if __name__ == "__main__":
    unittest.main()
