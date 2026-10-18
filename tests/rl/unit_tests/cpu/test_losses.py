# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn

from torchtitan.components.loss import ChunkedLossWrapper, compute_logprobs
from torchtitan.rl.trainer.losses import DAPOLoss, DPPOLoss
from torchtitan.rl.trainer.losses.common import normalize


def test_loss_normalization_uses_mutable_tensor_denominator() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)
    global_loss_token_counts = torch.tensor([7, 3], dtype=torch.int64)

    normalized = normalize(value, global_loss_token_counts)

    assert torch.equal(
        normalized,
        value * global_loss_token_counts[0].clamp_min(1).reciprocal(),
    )


@pytest.mark.parametrize(
    "loss_config", [DAPOLoss.Config(), DPPOLoss.Config()], ids=["dapo", "dppo"]
)
def test_policy_loss_matches_vllm_tempered_logprobs_bitwise(loss_config) -> None:
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

    _, metrics = loss_config.build()(
        logits,
        labels,
        torch.tensor(64),
        generator_logprobs=generator_logprobs,
        temperature=temperature,
        advantages=torch.ones(64),
        loss_mask=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0


@pytest.mark.parametrize(
    "loss_config", [DAPOLoss.Config(), DPPOLoss.Config()], ids=["dapo", "dppo"]
)
def test_policy_loss_through_chunked_loss_wrapper_matches_unchunked(
    loss_config,
) -> None:
    torch.manual_seed(0)
    num_tokens, dim, vocab_size = 16, 8, 32
    lm_head = nn.Linear(dim, vocab_size, bias=False)
    hidden = 3 * torch.randn(num_tokens, dim)
    with torch.no_grad():
        logits = lm_head(hidden)
        labels = logits.argmax(-1)
        trainer_logprobs = compute_logprobs(logits, labels, vocab_parallel_group=None)
    # A large mismatch, so DAPO clips and DPPO drops and truncates some tokens.
    generator_logprobs = (trainer_logprobs + 2 * torch.randn(num_tokens)).clamp(max=0)
    generator_logprobs[3] = float("nan")
    generator_logprobs[4] = float("-inf")
    loss_mask = torch.ones(num_tokens, dtype=torch.bool)
    loss_mask[:2] = False
    loss_inputs = {
        "generator_logprobs": generator_logprobs,
        "temperature": torch.ones(num_tokens),
        "advantages": torch.randn(num_tokens),
        "loss_mask": loss_mask,
    }
    num_loss_tokens = (loss_mask & torch.isfinite(generator_logprobs)).sum()

    ref_hidden = hidden.clone().requires_grad_()
    ref_loss, ref_metrics = loss_config.build()(
        lm_head(ref_hidden), labels, num_loss_tokens, **loss_inputs
    )
    ref_loss.backward()
    ref_weight_grad = lm_head.weight.grad.clone()
    lm_head.weight.grad = None

    chunked_loss = ChunkedLossWrapper.Config(num_chunks=2, loss_fn=loss_config).build()
    chunked_loss.set_lm_head(lm_head)
    chunked_hidden = hidden.clone().requires_grad_()
    loss, metrics = chunked_loss(chunked_hidden, labels, num_loss_tokens, **loss_inputs)
    loss.backward()

    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(metrics, ref_metrics)
    torch.testing.assert_close(chunked_hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(lm_head.weight.grad, ref_weight_grad)


def _dppo_logits_grad_and_metrics(
    loss_config: DPPOLoss.Config,
    *,
    trainer_probs: list[float],
    generator_probs: list[float],
    advantages: list[float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run `DPPOLoss` on [T, 2] logits whose label-0 probability is `trainer_probs`."""
    probs = torch.tensor(trainer_probs)
    logits = torch.stack([probs, 1 - probs], dim=-1).log().requires_grad_()
    num_tokens = len(trainer_probs)
    loss, metrics = loss_config.build()(
        logits,
        torch.zeros(num_tokens, dtype=torch.long),
        torch.tensor(num_tokens),
        generator_logprobs=torch.tensor(generator_probs).log(),
        temperature=torch.ones(num_tokens),
        advantages=torch.tensor(advantages),
        loss_mask=torch.ones(num_tokens, dtype=torch.bool),
    )
    loss.backward()
    return logits.grad, metrics


# Token 3 moved +0.29 against its advantage: only the symmetric trust region drops it.
@pytest.mark.parametrize(
    "symmetric, expected_has_grad, expected_dropped_frac",
    [
        (False, [False, True, False, True], 0.5),
        (True, [False, True, False, False], 0.75),
    ],
    ids=["paper", "symmetric"],
)
def test_dppo_drops_tokens_that_moved_past_threshold(
    symmetric, expected_has_grad, expected_dropped_frac
) -> None:
    # The DPPOLoss docstring example.
    grad, metrics = _dppo_logits_grad_and_metrics(
        DPPOLoss.Config(divergence_threshold=0.2, symmetric=symmetric),
        trainer_probs=[0.50, 0.05, 0.10, 0.30],
        generator_probs=[0.20, 0.01, 0.40, 0.01],
        advantages=[1.0, 1.0, -1.0, -1.0],
    )

    has_grad = grad.abs().sum(-1) > 0
    assert has_grad.tolist() == expected_has_grad
    assert metrics["loss/trust_region_dropped_frac"] == expected_dropped_frac


def test_dppo_keeps_zero_advantage_and_a_move_of_exactly_the_threshold() -> None:
    # A == 0 never drops, even after a fall of 0.5. 0.25 -> 0.5 is exactly 0.25 in fp32.
    _, metrics = _dppo_logits_grad_and_metrics(
        DPPOLoss.Config(divergence_threshold=0.25),
        trainer_probs=[0.10, 0.50],
        generator_probs=[0.60, 0.25],
        advantages=[0.0, 1.0],
    )

    assert metrics["loss/trust_region_dropped_frac"] == 0


def test_dppo_truncates_ratio_but_keeps_gradient() -> None:
    # Ratio 0.06 / 0.01 = 6 is past max_ratio 5; the move of +0.05 is inside the region.
    grad, metrics = _dppo_logits_grad_and_metrics(
        DPPOLoss.Config(max_ratio=5.0),
        trainer_probs=[0.06],
        generator_probs=[0.01],
        advantages=[1.0],
    )

    # d log p / d logits = [1 - p, -(1 - p)] for label 0 of two tokens.
    expected_grad = -5.0 * torch.tensor([[0.94, -0.94]])
    torch.testing.assert_close(grad, expected_grad)
    assert metrics["loss/ratio_truncated_frac"] == 1


def test_dppo_skips_non_finite_generator_logprobs() -> None:
    # Generator logprobs nan, -inf, finite. Without the skip, the -inf token is dropped.
    grad, metrics = _dppo_logits_grad_and_metrics(
        DPPOLoss.Config(),
        trainer_probs=[0.5, 0.5, 0.5],
        generator_probs=[float("nan"), 0.0, 0.4],
        advantages=[1.0, 1.0, 1.0],
    )

    has_grad = grad.abs().sum(-1) > 0
    assert has_grad.tolist() == [False, False, True]
    assert metrics["loss/trust_region_dropped_frac"] == 0
    assert all(torch.isfinite(value) for value in metrics.values())


def test_dppo_gradient_matches_unclipped_policy_gradient_inside_trust_region() -> None:
    torch.manual_seed(0)
    num_tokens, vocab_size = 16, 32
    logits = torch.randn(num_tokens, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (num_tokens,))
    advantages = torch.randn(num_tokens)
    with torch.no_grad():
        trainer_logprobs = compute_logprobs(logits, labels, vocab_parallel_group=None)
    # Small mismatch: every ratio is within ~5% of 1, every probability change < 0.2.
    generator_logprobs = trainer_logprobs + 0.05 * (2 * torch.rand(num_tokens) - 1)

    loss, metrics = DPPOLoss.Config().build()(
        logits,
        labels,
        torch.tensor(num_tokens),
        generator_logprobs=generator_logprobs,
        temperature=torch.ones(num_tokens),
        advantages=advantages,
        loss_mask=torch.ones(num_tokens, dtype=torch.bool),
    )
    (dppo_grad,) = torch.autograd.grad(loss, logits)

    ratio = torch.exp(
        compute_logprobs(logits, labels, vocab_parallel_group=None) - generator_logprobs
    )
    ppo_unclipped_loss = -(ratio * advantages).sum() / num_tokens
    (ppo_grad,) = torch.autograd.grad(ppo_unclipped_loss, logits)

    assert metrics["loss/trust_region_dropped_frac"] == 0
    torch.testing.assert_close(dppo_grad, ppo_grad)


def _reference_symmetric_trust_region_loss_sum(
    trainer_logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    threshold: float,
    max_ratio: float,
) -> torch.Tensor:
    """A separate implementation on already-masked tokens: drop |p - q| > threshold, cap the ratio's value."""
    log_ratio = trainer_logprobs - generator_logprobs
    larger = torch.maximum(trainer_logprobs, generator_logprobs)
    smaller = torch.minimum(trainer_logprobs, generator_logprobs)
    abs_prob_diff = torch.exp(larger) * -torch.expm1(smaller - larger)
    keep = ~(abs_prob_diff > threshold)
    kept = log_ratio[keep]
    capped = kept.detach().clamp(max=kept.new_tensor(max_ratio).log())
    importance_ratio = torch.exp(capped + (kept - kept.detach()))
    return (-advantages[keep] * importance_ratio).sum()


def test_dppo_symmetric_gradient_matches_reference() -> None:
    torch.manual_seed(0)
    num_tokens, vocab_size = 256, 64
    logits = (torch.randn(num_tokens, vocab_size) * 4).requires_grad_(True)
    labels = torch.randint(0, vocab_size, (num_tokens,))
    # The first 64 labels are each row's likeliest token, so a far-off generator crosses 0.3.
    labels[:64] = logits[:64].argmax(-1)
    with torch.no_grad():
        exact = torch.log_softmax(logits, -1).gather(-1, labels[:, None])[:, 0]
    generator_logprobs = exact + torch.randn(num_tokens) * 0.5
    # Rare under the generator: dropped where p is large, a capped huge ratio elsewhere.
    generator_logprobs[:8] = -30.0
    generator_logprobs[64:72] = -30.0
    advantages = torch.randn(num_tokens)
    loss_mask = torch.rand(num_tokens) > 0.2
    num_loss_tokens = int(loss_mask.sum())

    loss_config = DPPOLoss.Config(
        divergence_threshold=0.3, max_ratio=1e4, symmetric=True
    )
    loss, metrics = loss_config.build()(
        logits,
        labels,
        torch.tensor(num_loss_tokens),
        generator_logprobs=generator_logprobs,
        temperature=torch.ones(num_tokens),
        advantages=advantages,
        loss_mask=loss_mask,
    )
    (grad,) = torch.autograd.grad(loss, logits)

    reference_logits = logits.detach().clone().requires_grad_(True)
    trainer_logprobs = torch.log_softmax(reference_logits, -1).gather(
        -1, labels[:, None]
    )[:, 0]
    reference_loss = (
        _reference_symmetric_trust_region_loss_sum(
            trainer_logprobs[loss_mask],
            generator_logprobs[loss_mask],
            advantages[loss_mask],
            threshold=0.3,
            max_ratio=1e4,
        )
        / num_loss_tokens
    )
    (reference_grad,) = torch.autograd.grad(reference_loss, reference_logits)

    # Only the gradients match: this loss reports -A * w * log p, the reference -A * w.
    torch.testing.assert_close(grad, reference_grad)
    assert 0 < metrics["loss/trust_region_dropped_frac"] < 1
