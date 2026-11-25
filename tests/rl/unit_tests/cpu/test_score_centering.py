# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch

from torchtitan.components.loss import compute_topk_logprobs
from torchtitan.rl.trainer.losses import (
    CISPOLoss,
    DAPOLoss,
    GRPOLoss,
    IcePopLoss,
    ScoreCentering,
)
from torchtitan.rl.trainer.losses.policy_gradient import MAX_LOG_RATIO

_LOSS_CONFIGS = {
    "dapo": DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
    "grpo": GRPOLoss.Config(clip_eps=0.1),
    "cispo": CISPOLoss.Config(ratio_clip_high=0.2),
    "icepop": IcePopLoss.Config(ratio_mask_low=0.8, ratio_mask_high=1.25),
}


def _with_score_centering(config, num_topk_logprobs):
    return replace(
        config,
        score_centering=ScoreCentering.Config(num_topk_logprobs=num_topk_logprobs),
    )


@pytest.mark.parametrize("name", list(_LOSS_CONFIGS))
def test_log_token_weight_matches_token_loss_gradient(name) -> None:
    """The weight score centering centers is the one the surrogate's gradient puts on the
    token's score: d token_loss / d log p = -A * w, including the clip bounds and the
    clamp of |log p / q| to MAX_LOG_RATIO."""
    loss = _LOSS_CONFIGS[name].build()
    torch.manual_seed(0)
    log_ratio = torch.cat(
        [torch.linspace(-0.6, 0.6, 997), torch.tensor([-12.0, -10.5, 10.5, 12.0])]
    ).requires_grad_(True)
    advantages = torch.randn(log_ratio.shape[0])
    ratio = torch.exp(log_ratio.clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO))

    loss.token_loss(ratio, advantages).sum().backward()

    expected = -advantages * torch.exp(loss.log_token_weight(log_ratio, advantages))
    torch.testing.assert_close(log_ratio.grad, expected)
    # Both sides of both clip bounds, and both clamp sides, are exercised.
    assert (expected == 0).any() and (expected != 0).any()


def _expected_logit_gradient(loss, logits_row, generator_logprobs, advantage, topk_ids):
    """``sum_y q_y * d loss(y) / d logits`` at one position: the expected update."""
    vocab_size = logits_row.shape[0]
    logits = logits_row.expand(vocab_size, vocab_size).clone().requires_grad_(True)
    labels = torch.arange(vocab_size)
    num_topk = topk_ids.shape[0]
    total, _ = loss(
        logits,
        labels,
        None,
        generator_logprobs=generator_logprobs,
        temperature=torch.full((vocab_size,), 0.7),
        advantages=torch.full((vocab_size,), advantage),
        loss_mask=torch.ones(vocab_size, dtype=torch.bool),
        generator_topk_token_ids=topk_ids.expand(vocab_size, num_topk),
        generator_topk_logprobs=generator_logprobs[topk_ids].expand(
            vocab_size, num_topk
        ),
    )
    total.backward()
    return (generator_logprobs.exp()[:, None] * logits.grad).sum(0)


@pytest.mark.parametrize("name", list(_LOSS_CONFIGS))
@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_score_centering_cancels_drift_with_the_full_vocab(name, advantage) -> None:
    """At a constant advantage the reward carries no signal, so the expected update must be
    zero; the generator's mismatch makes it nonzero without score centering."""
    torch.manual_seed(0)
    vocab_size = 16
    logits_row = torch.randn(vocab_size)
    # A large mismatch, so both clip bounds bind somewhere.
    generator_logprobs = torch.log_softmax(
        (logits_row + torch.randn(vocab_size)) / 0.7, dim=-1
    )
    topk_ids = torch.arange(vocab_size)
    base = _LOSS_CONFIGS[name]

    drift = _expected_logit_gradient(
        base.build(), logits_row, generator_logprobs, advantage, topk_ids
    )
    centered = _expected_logit_gradient(
        _with_score_centering(base, vocab_size).build(),
        logits_row,
        generator_logprobs,
        advantage,
        topk_ids,
    )

    assert drift.abs().max() > 1e-2
    torch.testing.assert_close(centered, torch.zeros_like(centered), atol=1e-6, rtol=0)


@pytest.mark.parametrize("name", list(_LOSS_CONFIGS))
@pytest.mark.parametrize("tail_scale", [0.5, 0.95, 1.6])
def test_score_centering_tail_model_is_exact_for_a_proportional_tail(
    name, tail_scale
) -> None:
    """With only the top-k, cancellation is exact when the generator's tail is the trainer's
    scaled by rho, whether the tail's ratio 1 / rho is inside the clip band (0.95) or not."""
    torch.manual_seed(1)
    vocab_size, num_topk = 16, 4
    logits_row = torch.randn(vocab_size)
    trainer_probs = torch.softmax(logits_row / 0.7, dim=-1)
    head = torch.argsort(trainer_probs, descending=True)[:num_topk]
    tail = torch.ones(vocab_size, dtype=torch.bool)
    tail[head] = False
    generator_probs = torch.empty(vocab_size)
    generator_probs[tail] = tail_scale * trainer_probs[tail]
    head_mass = 1 - generator_probs[tail].sum()
    head_weights = torch.rand(num_topk) + 0.5
    generator_probs[head] = head_mass * head_weights / head_weights.sum()
    base = _LOSS_CONFIGS[name]

    centered = _expected_logit_gradient(
        _with_score_centering(base, num_topk).build(),
        logits_row,
        generator_probs.log(),
        1.0,
        head,
    )

    torch.testing.assert_close(centered, torch.zeros_like(centered), atol=1e-6, rtol=0)


@pytest.mark.parametrize("name", list(_LOSS_CONFIGS))
def test_score_centering_is_a_bitwise_noop_when_the_generator_matches(name) -> None:
    """A generator whose top-k logprobs equal the trainer's bitwise gives coefficients of 0,
    so the loss and gradient are bitwise those without score centering (batch-invariant
    mode relies on this)."""
    torch.manual_seed(2)
    num_tokens, vocab_size, num_topk = 32, 64, 8
    logits = (torch.randn(num_tokens, vocab_size) * 2).to(torch.bfloat16)
    labels = torch.randint(0, vocab_size, (num_tokens,))
    temperature = torch.full((num_tokens,), 0.7)
    trainer_logprobs = torch.log_softmax(logits.float() / 0.7, dim=-1)
    topk_ids = trainer_logprobs.topk(num_topk, dim=-1).indices.to(torch.int32)
    kwargs = dict(
        generator_logprobs=trainer_logprobs.gather(-1, labels[:, None])[:, 0],
        temperature=temperature,
        advantages=torch.randn(num_tokens),
        loss_mask=torch.rand(num_tokens) > 0.2,
    )
    topk_kwargs = dict(
        generator_topk_token_ids=topk_ids,
        generator_topk_logprobs=compute_topk_logprobs(
            logits,
            topk_ids,
            vocab_parallel_group=None,
            global_vocab_size=None,
            temperature=temperature,
        ),
    )
    base = _LOSS_CONFIGS[name]

    results = []
    for loss, extra in (
        (base.build(), {}),
        (_with_score_centering(base, num_topk).build(), topk_kwargs),
    ):
        x = logits.clone().requires_grad_(True)
        value, metrics = loss(x, labels, torch.tensor(num_tokens), **kwargs, **extra)
        value.backward()
        results.append((value.detach(), x.grad, metrics))

    (loss_off, grad_off, _), (loss_on, grad_on, metrics_on) = results
    assert torch.equal(loss_off, loss_on)
    assert torch.equal(grad_off, grad_on)
    assert metrics_on["score_centering/coefficient_l1/mean"] == 0
