# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.rl.trainer.losses.score_centering import ScoreCenteringLoss


def _reference_score_centering_loss(
    train_logp,
    samp_logp,
    topk_ids,
    sampled_token,
    samp_token_logp,
    advantage,
    weight_fn=torch.ones_like,
    eps=1e-6,
):
    """The PyTorch reference from github.com/martin-marek/score-centering (README.md)."""
    head_logp = train_logp.gather(-1, topk_ids)
    token_logp = train_logp.gather(-1, sampled_token[..., None])[..., 0]
    p_head, q_head = head_logp.exp(), samp_logp.exp()
    p_tail = (1 - p_head.sum(-1)).clamp_min(eps)
    q_tail = (1 - q_head.sum(-1)).clamp_min(eps)
    rho = q_tail / p_tail
    alpha = rho * weight_fn(1 / rho)
    head_weight = weight_fn((head_logp - samp_logp).exp())
    residual = q_head * head_weight - alpha[..., None] * p_head
    correction = (residual.detach() * head_logp).sum(-1)
    token_weight = weight_fn((token_logp - samp_token_logp).exp())
    return -advantage.detach() * (token_weight.detach() * token_logp - correction)


def _loss_inputs(logits, labels, generator_log_softmax, num_topk, temperature=1.0):
    """Loss kwargs for a generator whose full-vocab logprobs are `generator_log_softmax`."""
    num_tokens = logits.shape[0]
    topk_logprobs, topk_token_ids = generator_log_softmax.topk(num_topk, dim=-1)
    return {
        "generator_logprobs": generator_log_softmax.gather(-1, labels[:, None])[:, 0],
        "temperature": torch.full((num_tokens,), temperature),
        "advantages": torch.randn(num_tokens),
        "loss_mask": torch.ones(num_tokens, dtype=torch.bool),
        "generator_topk_token_ids": topk_token_ids.to(torch.int32),
        "generator_topk_logprobs": topk_logprobs,
    }


# 1.2 truncates sampled-token, head, and tail ratios in this setup.
@pytest.mark.parametrize("max_ratio", [None, 1.2], ids=["no_ratio", "truncated"])
def test_score_centering_matches_reference_implementation(max_ratio) -> None:
    torch.manual_seed(0)
    num_tokens, vocab_size, num_topk, temperature = 12, 32, 4, 0.7
    logits = torch.randn(num_tokens, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (num_tokens,))
    generator_logits = logits.detach() + 0.5 * torch.randn(num_tokens, vocab_size)
    generator_log_softmax = torch.log_softmax(generator_logits / temperature, dim=-1)
    loss_inputs = _loss_inputs(
        logits, labels, generator_log_softmax, num_topk, temperature
    )

    loss, _ = ScoreCenteringLoss.Config(max_ratio=max_ratio).build()(
        logits, labels, torch.tensor(num_tokens), **loss_inputs
    )
    (grad,) = torch.autograd.grad(loss, logits)

    weight_fn = (
        torch.ones_like if max_ratio is None else lambda r: r.clamp_max(max_ratio)
    )
    reference_loss = (
        _reference_score_centering_loss(
            torch.log_softmax(logits / temperature, dim=-1),
            loss_inputs["generator_topk_logprobs"],
            loss_inputs["generator_topk_token_ids"].long(),
            labels,
            loss_inputs["generator_logprobs"],
            loss_inputs["advantages"],
            weight_fn=weight_fn,
        ).sum()
        / num_tokens
    )
    (reference_grad,) = torch.autograd.grad(reference_loss, logits)

    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(grad, reference_grad)


@pytest.mark.parametrize("max_ratio", [None, 2.0], ids=["no_ratio", "truncated"])
def test_score_centering_cancels_drift_under_a_constant_advantage(max_ratio) -> None:
    # Score every vocab token as the sampled one at one position, with `advantages=q`: the
    # summed loss is then E_q[token_loss] at A = 1, which must have zero gradient although
    # q != p. Exact because q's tail outside the top-k is proportional to p's (the modeled tail).
    torch.manual_seed(0)
    vocab_size, num_topk = 8, 3
    trainer_logits = torch.randn(1, vocab_size, requires_grad=True)
    p = torch.softmax(trainer_logits.detach(), dim=-1)[0]
    head = torch.tensor([5, 1, 6])
    q = 0.4 * p
    q[head] = torch.tensor([0.30, 0.15, 0.20])
    q = q / q.sum()
    labels = torch.arange(vocab_size)

    loss, _ = ScoreCenteringLoss.Config(max_ratio=max_ratio).build()(
        trainer_logits.expand(vocab_size, -1),
        labels,
        torch.tensor(vocab_size),
        generator_logprobs=q.log(),
        temperature=torch.ones(vocab_size),
        advantages=q,
        loss_mask=torch.ones(vocab_size, dtype=torch.bool),
        generator_topk_token_ids=head.expand(vocab_size, num_topk),
        generator_topk_logprobs=q[head].log().expand(vocab_size, num_topk),
    )
    (grad,) = torch.autograd.grad(loss, trainer_logits)
    # Without the correction, the same expectation drifts (with TIS, only partly cancelled).
    token_weight = 1.0 if max_ratio is None else torch.clamp(p / q, max=max_ratio)
    (uncorrected_grad,) = torch.autograd.grad(
        -(q * token_weight * torch.log_softmax(trainer_logits, dim=-1)[0]).sum(),
        trainer_logits,
    )

    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-6, rtol=0)
    assert uncorrected_grad.abs().max() > 0.01


def test_score_centering_is_reinforce_when_generator_matches_trainer() -> None:
    torch.manual_seed(0)
    num_tokens, vocab_size, num_topk = 8, 16, 4
    logits = torch.randn(num_tokens, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (num_tokens,))
    loss_inputs = _loss_inputs(
        logits, labels, torch.log_softmax(logits.detach(), dim=-1), num_topk
    )

    loss, metrics = ScoreCenteringLoss.Config().build()(
        logits, labels, torch.tensor(num_tokens), **loss_inputs
    )
    (grad,) = torch.autograd.grad(loss, logits)
    reinforce_loss = (
        -(
            loss_inputs["advantages"]
            * torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None])[:, 0]
        ).sum()
        / num_tokens
    )
    (reinforce_grad,) = torch.autograd.grad(reinforce_loss, logits)

    torch.testing.assert_close(grad, reinforce_grad)
    assert metrics["score_centering/coefficient_l1/mean"] < 1e-6


def test_score_centering_skips_masked_and_non_finite_tokens() -> None:
    # Rows 0-1 are prompt tokens with the zero top-k rows from `to_loss_kwargs`. Row 2 has
    # NaN generator logprobs. None of them gets gradient, and nothing turns NaN.
    torch.manual_seed(0)
    num_tokens, vocab_size, num_topk = 6, 16, 4
    logits = torch.randn(num_tokens, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (num_tokens,))
    generator_log_softmax = torch.log_softmax(
        logits.detach() + torch.randn(num_tokens, vocab_size), dim=-1
    )
    loss_inputs = _loss_inputs(logits, labels, generator_log_softmax, num_topk)
    loss_inputs["loss_mask"][:2] = False
    loss_inputs["generator_topk_token_ids"][:2] = 0
    loss_inputs["generator_topk_logprobs"][:2] = 0.0
    loss_inputs["generator_logprobs"][2] = float("nan")
    loss_inputs["generator_topk_logprobs"][2] = float("nan")

    loss, metrics = ScoreCenteringLoss.Config().build()(
        logits, labels, torch.tensor(3), **loss_inputs
    )
    (grad,) = torch.autograd.grad(loss, logits)

    has_grad = grad.abs().sum(-1) > 0
    assert has_grad.tolist() == [False, False, False, True, True, True]
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())


def test_score_centering_through_chunked_loss_wrapper_matches_unchunked() -> None:
    torch.manual_seed(0)
    num_tokens, dim, vocab_size, num_topk = 16, 8, 32, 4
    lm_head = nn.Linear(dim, vocab_size, bias=False)
    hidden = torch.randn(num_tokens, dim)
    labels = torch.randint(0, vocab_size, (num_tokens,))
    with torch.no_grad():
        generator_log_softmax = torch.log_softmax(
            lm_head(hidden) + torch.randn(num_tokens, vocab_size), dim=-1
        )
    loss_inputs = _loss_inputs(lm_head(hidden), labels, generator_log_softmax, num_topk)
    loss_inputs["loss_mask"][:2] = False
    loss_inputs["generator_logprobs"][3] = float("nan")
    num_loss_tokens = torch.tensor(13)

    ref_hidden = hidden.clone().requires_grad_()
    ref_loss, ref_metrics = ScoreCenteringLoss.Config().build()(
        lm_head(ref_hidden), labels, num_loss_tokens, **loss_inputs
    )
    ref_loss.backward()
    ref_weight_grad = lm_head.weight.grad.clone()
    lm_head.weight.grad = None

    chunked_loss = ChunkedLossWrapper.Config(
        num_chunks=2, loss_fn=ScoreCenteringLoss.Config()
    ).build()
    chunked_loss.set_lm_head(lm_head)
    chunked_hidden = hidden.clone().requires_grad_()
    loss, metrics = chunked_loss(chunked_hidden, labels, num_loss_tokens, **loss_inputs)
    loss.backward()

    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(metrics, ref_metrics)
    torch.testing.assert_close(chunked_hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(lm_head.weight.grad, ref_weight_grad)
