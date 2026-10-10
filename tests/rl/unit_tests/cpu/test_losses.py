# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.rl.losses.dapo import _normalize, DAPOLoss


def test_loss_normalization_uses_mutable_tensor_denominator() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)
    global_loss_token_counts = torch.tensor([7, 3], dtype=torch.int64)

    normalized = _normalize(value, global_loss_token_counts)

    assert torch.equal(
        normalized,
        value * global_loss_token_counts[0].clamp_min(1).reciprocal(),
    )


def test_dapo_loss_matches_vllm_tempered_logprobs_bitwise() -> None:
    torch.manual_seed(0)
    logits = (torch.randn(64, 128) * 3).to(torch.bfloat16)
    labels = torch.randint(0, 128, (64,))
    temperature = torch.full((64,), 0.7)
    # vLLM's sampler: logits.to(float32).div_(temperature), then log_softmax(dtype=float32).
    generator_logprobs = (
        logits.to(torch.float32)
        .div_(temperature[:, None])
        .log_softmax(dim=-1, dtype=torch.float32)
        .gather(-1, labels[:, None])
        .squeeze(-1)
    )

    _, metrics = DAPOLoss.Config().build()(
        logits,
        labels,
        torch.tensor(64),
        generator_logprobs=generator_logprobs,
        temperature=temperature,
        advantages=torch.ones(64),
        loss_mask=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0


def test_dapo_logs_logprob_gap_metrics() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 8)
    labels = torch.tensor([1, 2, 3, 4])
    trainer_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None])[
        :, 0
    ]
    # log p_trainer - log q_generator per token; the last token is not trained.
    logprob_diffs = torch.tensor([0.1, -0.2, 1.0, 5.0])
    loss_fn = DAPOLoss.Config().build()

    _, metrics = loss_fn(
        logits,
        labels,
        torch.tensor(3),
        generator_logprobs=trainer_logprobs - logprob_diffs,
        temperature=torch.ones(4),
        advantages=torch.zeros(4),
        loss_mask=torch.tensor([True, True, True, False]),
    )

    trained = logprob_diffs[:3]
    torch.testing.assert_close(
        metrics["bit_wise/logprob_diff_abs/mean"], trained.abs().mean()
    )
    torch.testing.assert_close(
        metrics["bit_wise/kl_k3/mean"], (trained.exp() - 1 - trained).mean()
    )
    # Only the 1.0 diff has p/q above 2.
    torch.testing.assert_close(
        metrics["bit_wise/ratio_beyond_2x/mean"], torch.tensor(1 / 3)
    )
