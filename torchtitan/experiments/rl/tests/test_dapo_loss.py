# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.losses import DAPOLoss, GRPOLoss
from torchtitan.experiments.rl.types import (
    RolloutTurnID,
    TrainingSample,
    TrainingSampleGroup,
)


def test_batcher_counts_only_finite_response_tokens_as_valid() -> None:
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=1, seq_len=2)).build(
        num_prompts_per_train_step=1, dp_degree=1, pad_id=0
    )
    sample = TrainingSample(
        min_policy_version=0,
        max_policy_version=0,
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        token_ids=[1, 2, 3],
        loss_mask=[False, True, True],
        logprobs=[0.0, -0.2, float("nan")],
        advantage=[0.0, 1.0, 1.0],
    )

    batch = batcher.add_training_samples(
        training_sample_group=TrainingSampleGroup(
            group_id=0, training_samples=[sample], metrics=[]
        )
    )

    assert batch is not None
    assert batch.num_global_valid_tokens == 1
    metrics = {metric.key: metric.value.value for metric in batch.metrics}
    assert metrics["loss/generator_logprob_nan_frac"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "loss_fn", [DAPOLoss(DAPOLoss.Config()), GRPOLoss(GRPOLoss.Config())]
)
def test_nonfinite_generator_logprob_is_excluded_from_loss_and_metrics(
    loss_fn: DAPOLoss,
) -> None:
    base_logits = torch.tensor([[[1.0, -1.0], [0.5, -0.5]]])
    labels = torch.tensor([[0, 1]])
    advantages = torch.ones_like(labels, dtype=torch.float32)
    generator_logprobs = torch.tensor([[-0.2, torch.nan]])

    actual_logits = base_logits.clone().requires_grad_(True)
    actual_loss, metrics = loss_fn(
        actual_logits,
        labels,
        1,
        generator_logprobs=generator_logprobs,
        advantages=advantages,
        loss_mask=torch.ones_like(labels, dtype=torch.bool),
    )
    actual_loss.backward()
    assert actual_logits.grad is not None

    reference_logits = base_logits.clone().requires_grad_(True)
    reference_loss, reference_metrics = loss_fn(
        reference_logits,
        labels,
        1,
        generator_logprobs=generator_logprobs,
        advantages=advantages,
        loss_mask=torch.tensor([[True, False]]),
    )
    reference_loss.backward()
    assert reference_logits.grad is not None

    torch.testing.assert_close(actual_loss, reference_loss)
    torch.testing.assert_close(actual_logits.grad, reference_logits.grad)
    assert metrics.keys() == reference_metrics.keys()
    for key in metrics:
        torch.testing.assert_close(metrics[key], reference_metrics[key])


@pytest.mark.parametrize(
    "loss_fn", [DAPOLoss(DAPOLoss.Config()), GRPOLoss(GRPOLoss.Config())]
)
def test_all_nonfinite_generator_logprobs_produce_zero_finite_loss(
    loss_fn: DAPOLoss,
) -> None:
    logits = torch.tensor([[[1.0, -1.0], [0.5, -0.5]]], requires_grad=True)
    labels = torch.tensor([[0, 1]])
    loss, metrics = loss_fn(
        logits,
        labels,
        0,
        generator_logprobs=torch.full_like(labels, torch.nan, dtype=torch.float32),
        advantages=torch.ones_like(labels, dtype=torch.float32),
        loss_mask=torch.ones_like(labels, dtype=torch.bool),
    )

    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))
    for metric in metrics.values():
        assert torch.isfinite(metric)
