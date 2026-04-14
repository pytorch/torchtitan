# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.experiments.graph_trainer.llama3 import (
    model_registry as llama3_registry,
)
from torchtitan.experiments.graph_trainer.tests._trainer_test_utils import (
    build_minimal_trainer,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.trainer import Trainer

DTYPE = torch.bfloat16
BATCH_SIZE = 2
SEQ_LEN = 2048
MAX_PEAK_MEMORY_RATIO = 1.10
DEBUGMODEL = "debugmodel"


def _set_deterministic() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)


def _build_model(model_flavor: str) -> nn.Module:
    model_spec = llama3_registry(model_flavor)
    with torch.device("meta"):
        model = model_spec.model.build()
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_states(buffer_device=None)
    model.to(dtype=DTYPE)
    model.train()
    return model


@dataclass(frozen=True)
class StepResult:
    loss: torch.Tensor
    grads: list[torch.Tensor]
    reserved_gib: float
    active_gib: float


def _measure_step(
    trainer: Trainer, tokens: torch.Tensor, labels: torch.Tensor
) -> StepResult:
    model = trainer.model_parts[0]
    model.zero_grad(set_to_none=True)
    global_valid_tokens = torch.tensor(labels.numel(), dtype=torch.float, device="cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    loss = trainer.forward_backward_step(
        input_dict={"input": tokens},
        labels=labels,
        global_valid_tokens=global_valid_tokens,
    )
    torch.cuda.synchronize()

    stats = torch.cuda.memory_stats()
    grads = [param.grad.detach().clone() for param in model.parameters()]
    return StepResult(
        loss=loss.detach().clone(),
        grads=grads,
        reserved_gib=torch.cuda.max_memory_reserved() / 1e9,
        active_gib=stats["active_bytes.all.peak"] / 1e9,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGraphSACPeakMemory(unittest.TestCase):
    def setUp(self):
        _set_deterministic()
        model = _build_model(DEBUGMODEL)
        self.state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        del model
        torch.cuda.empty_cache()
        self.tokens = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")
        self.labels = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def test_llama3_debugmodel_peak_memory_matches_eager_selective_ac(self):
        eager_model = _build_model(DEBUGMODEL)
        eager_model.load_state_dict(copy.deepcopy(self.state_dict))
        apply_ac(eager_model, ActivationCheckpointConfig(mode="selective"))
        eager_trainer = build_minimal_trainer(
            eager_model,
            llama3_registry(DEBUGMODEL).model,
            Trainer,
        )

        traced_model = _build_model(DEBUGMODEL)
        traced_model.load_state_dict(copy.deepcopy(self.state_dict))
        traced_trainer = build_minimal_trainer(
            traced_model,
            llama3_registry(DEBUGMODEL).model,
            GraphTrainer,
            activation_checkpoint_mode="selective",
        )

        # Warm up both paths so allocator and one-time tracing setup do not skew
        # the measured peak memory.
        _measure_step(eager_trainer, self.tokens, self.labels)
        _measure_step(traced_trainer, self.tokens, self.labels)
        torch.cuda.empty_cache()

        eager = _measure_step(eager_trainer, self.tokens, self.labels)
        traced = _measure_step(traced_trainer, self.tokens, self.labels)

        self.assertTrue(
            torch.equal(eager.loss, traced.loss),
            f"loss mismatch: eager={eager.loss.item()} traced={traced.loss.item()}",
        )
        for idx, (eager_grad, traced_grad) in enumerate(
            zip(eager.grads, traced.grads, strict=True)
        ):
            self.assertTrue(
                torch.equal(eager_grad, traced_grad), f"grad[{idx}] mismatch"
            )

        reserved_ratio = traced.reserved_gib / eager.reserved_gib
        active_ratio = traced.active_gib / eager.active_gib
        self.assertLessEqual(
            reserved_ratio,
            MAX_PEAK_MEMORY_RATIO,
            "graph SAC reserved peak memory too high: "
            f"traced={traced.reserved_gib:.3f} GiB, "
            f"eager={eager.reserved_gib:.3f} GiB, "
            f"ratio={reserved_ratio:.3f}",
        )
        self.assertLessEqual(
            active_ratio,
            MAX_PEAK_MEMORY_RATIO,
            "graph SAC active peak memory too high: "
            f"traced={traced.active_gib:.3f} GiB, "
            f"eager={eager.active_gib:.3f} GiB, "
            f"ratio={active_ratio:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
