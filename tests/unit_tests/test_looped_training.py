# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchtitan.components.loss import IGNORE_INDEX, LoopedEntropyLoss
from torchtitan.config import LoopingConfig, ParallelismConfig
from torchtitan.trainer import Trainer


def _manual_exit_probs(gates: torch.Tensor) -> torch.Tensor:
    steps = gates.shape[0]
    if steps == 1:
        return torch.ones_like(gates)
    lambdas = torch.sigmoid(gates)
    survival = torch.ones_like(gates[0])
    probs = []
    for idx in range(steps - 1):
        probs.append(lambdas[idx] * survival)
        survival = survival * (1.0 - lambdas[idx])
    probs.append(survival)
    return torch.stack(probs)


def test_looped_entropy_loss_matches_reference():
    torch.manual_seed(0)
    steps, batch, seq, vocab = 3, 2, 4, 7
    logits = torch.randn(steps, batch, seq, vocab, requires_grad=True)
    gates = torch.randn(steps, batch, seq, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[0, 1] = IGNORE_INDEX

    beta = 0.1
    loss_fn = LoopedEntropyLoss(LoopedEntropyLoss.Config(beta=beta))
    actual = loss_fn(logits, gates, labels)

    expanded_labels = labels.unsqueeze(0).expand(steps, -1, -1)
    ce = F.cross_entropy(
        logits.reshape(-1, vocab).float(),
        expanded_labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(steps, batch, seq)
    probs = _manual_exit_probs(gates)
    valid = labels != IGNORE_INDEX
    expected_task = (probs * ce).sum(dim=0)[valid].sum()
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=0)[valid].sum()
    expected = expected_task - beta * entropy

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert gates.grad is not None


def test_looped_entropy_loss_global_token_normalization():
    torch.manual_seed(1)
    steps, batch, seq, vocab = 2, 1, 3, 5
    logits = torch.randn(steps, batch, seq, vocab)
    gates = torch.zeros(steps, batch, seq)
    labels = torch.tensor([[1, IGNORE_INDEX, 3]])

    loss_fn = LoopedEntropyLoss(LoopedEntropyLoss.Config(beta=0.0))
    unnormalized = loss_fn(logits, gates, labels)
    normalized = loss_fn(logits, gates, labels, global_valid_tokens=2.0)

    torch.testing.assert_close(normalized, unnormalized / 2.0)


def test_looped_entropy_loss_steps_one_degenerates_to_ce():
    torch.manual_seed(2)
    logits = torch.randn(1, 2, 3, 11, requires_grad=True)
    gates = torch.randn(1, 2, 3, requires_grad=True)
    labels = torch.randint(0, 11, (2, 3))

    loss_fn = LoopedEntropyLoss(LoopedEntropyLoss.Config(beta=1.0))
    actual = loss_fn(logits, gates, labels)
    expected = F.cross_entropy(
        logits.squeeze(0).reshape(-1, 11).float(),
        labels.reshape(-1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert gates.grad is None


@pytest.mark.parametrize(
    ("parallelism", "message"),
    [
        (ParallelismConfig(enable_sequence_parallel=True), "enable_sequence_parallel"),
        (
            ParallelismConfig(
                enable_sequence_parallel=False,
                tensor_parallel_degree=2,
            ),
            "tensor_parallel_degree",
        ),
        (
            ParallelismConfig(
                enable_sequence_parallel=False,
                context_parallel_degree=2,
            ),
            "context_parallel_degree",
        ),
        (
            ParallelismConfig(
                enable_sequence_parallel=False,
                expert_parallel_degree=2,
            ),
            "expert_parallel_degree",
        ),
        (
            ParallelismConfig(
                enable_sequence_parallel=False,
                pipeline_parallel_degree=2,
            ),
            "pipeline_parallel_degree",
        ),
    ],
)
def test_looped_training_rejects_unsupported_parallelism(parallelism, message):
    with pytest.raises(ValueError, match=message):
        Trainer.Config(
            looping=LoopingConfig(enable=True),
            parallelism=parallelism,
        )


def test_looped_training_allows_data_parallel_only_config():
    cfg = Trainer.Config(
        looping=LoopingConfig(enable=True),
        parallelism=ParallelismConfig(enable_sequence_parallel=False),
    )
    assert cfg.looping.enable
