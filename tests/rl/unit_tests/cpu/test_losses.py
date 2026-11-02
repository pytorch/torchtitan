# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.rl.trainer.losses.dapo import DAPOLoss


def _vllm_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: torch.Tensor
) -> torch.Tensor:
    """vLLM's sampler: logits.to(float32).div_(temperature), then log_softmax(dtype=float32)."""
    return (
        logits.to(torch.float32)
        .div_(temperature[:, None])
        .log_softmax(dim=-1, dtype=torch.float32)
        .gather(-1, labels[:, None])
        .squeeze(-1)
    )


def test_dapo_loss_and_mean_metrics_are_token_sums() -> None:
    torch.manual_seed(0)
    logits = torch.randn(6, 16)
    labels = torch.randint(0, 16, (6,))
    temperature = torch.ones(6)
    advantages = torch.arange(6, dtype=torch.float32)
    loss_mask = torch.tensor([True, True, False, True, True, False])

    loss, metrics = DAPOLoss.Config().build()(
        pred=logits,
        labels=labels,
        generator_logprobs=_vllm_logprobs(logits, labels, temperature),
        temperature=temperature,
        advantages=advantages,
        loss_mask=loss_mask,
    )

    # On-policy tokens have ratio 1, so each loss token contributes -advantage.
    assert metrics["bit_wise/logprob_diff/max"] == 0
    torch.testing.assert_close(loss, -(advantages * loss_mask).sum())
    torch.testing.assert_close(metrics["loss/mean"], loss)
    torch.testing.assert_close(metrics["loss/ratio_mean"], torch.tensor(4.0))


def test_dapo_loss_matches_vllm_tempered_logprobs_bitwise() -> None:
    torch.manual_seed(0)
    logits = (torch.randn(64, 128) * 3).to(torch.bfloat16)
    labels = torch.randint(0, 128, (64,))
    temperature = torch.full((64,), 0.7)
    generator_logprobs = _vllm_logprobs(logits, labels, temperature)

    _, metrics = DAPOLoss.Config().build()(
        pred=logits,
        labels=labels,
        generator_logprobs=generator_logprobs,
        temperature=temperature,
        advantages=torch.ones(64),
        loss_mask=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0
