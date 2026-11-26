# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.rl.trainer.losses import (
    CISPOLoss,
    DAPOLoss,
    DPPOLoss,
    GRPOLoss,
    IcePopLoss,
)
from torchtitan.rl.trainer.losses.policy_gradient import (
    _normalize,
    MAX_LOG_RATIO,
    ScoredTokens,
)


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
        logits=logits,
        labels=labels,
        global_loss_token_counts=torch.tensor(64),
        generator_logprobs=generator_logprobs,
        temperature=temperature,
        advantages=torch.ones(64),
        loss_mask=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0


def _weight_on_score(loss, *, trainer_logprobs, generator_logprobs, advantages):
    """The weight ``w`` in ``d token_loss / d log p = -A * w``, per token."""
    trainer_logprobs = trainer_logprobs.clone().requires_grad_(True)
    log_ratio = (trainer_logprobs - generator_logprobs).clamp(
        -MAX_LOG_RATIO, MAX_LOG_RATIO
    )
    tokens = ScoredTokens(
        trainer_logprobs=trainer_logprobs,
        generator_logprobs=generator_logprobs,
        ratio=torch.exp(log_ratio),
        advantages=advantages,
    )
    loss.token_loss(tokens).sum().backward()
    return -trainer_logprobs.grad / advantages


@pytest.mark.parametrize(
    "name, loss_config, expected_weight",
    [
        (
            "dapo",
            DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
            lambda r, a, p, q: r * torch.where(a > 0, r <= 1.28, r >= 0.8),
        ),
        (
            "grpo",
            GRPOLoss.Config(clip_eps=0.1),
            lambda r, a, p, q: r * torch.where(a > 0, r <= 1.1, r >= 0.9),
        ),
        ("cispo", CISPOLoss.Config(), lambda r, a, p, q: r.clamp(max=2.0)),
        (
            "icepop",
            IcePopLoss.Config(),
            lambda r, a, p, q: r * ((r >= 0.5) & (r <= 5.0)),
        ),
        (
            "dppo",
            DPPOLoss.Config(),
            lambda r, a, p, q: r.clamp(max=5.0) * (a.sign() * (p - q) <= 0.2),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_policy_loss_weights_each_token_score(name, loss_config, expected_weight):
    """Each loss's gradient is -A * w * grad(log p) with the weight its docstring states."""
    torch.manual_seed(0)
    num_tokens = 4096
    generator_logprobs = torch.log(torch.rand(num_tokens).clamp_min(1e-3))
    # Ratios spanning [e^-2, e^2], so every clip, cap, band and trust region binds.
    trainer_logprobs = (generator_logprobs + 4 * torch.rand(num_tokens) - 2).clamp(
        max=0
    )
    advantages = torch.randn(num_tokens)

    weight = _weight_on_score(
        loss_config.build(),
        trainer_logprobs=trainer_logprobs,
        generator_logprobs=generator_logprobs,
        advantages=advantages,
    )

    ratio = torch.exp(trainer_logprobs - generator_logprobs)
    expected = expected_weight(
        ratio, advantages, trainer_logprobs.exp(), generator_logprobs.exp()
    ).float()
    torch.testing.assert_close(weight, expected)
    # The clip, cap, band or trust region binds on some tokens and not on others.
    assert (expected != ratio).any() and (expected == ratio).any()


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


def test_dppo_drops_tokens_that_moved_past_threshold() -> None:
    # The DPPOLoss docstring example.
    grad, metrics = _dppo_logits_grad_and_metrics(
        DPPOLoss.Config(divergence_threshold=0.2),
        trainer_probs=[0.50, 0.05, 0.10, 0.30],
        generator_probs=[0.20, 0.01, 0.40, 0.01],
        advantages=[1.0, 1.0, -1.0, -1.0],
    )

    has_grad = grad.abs().sum(-1) > 0
    assert has_grad.tolist() == [False, True, False, True]
    assert metrics["loss/trust_region_dropped_frac"] == 0.5


def test_dppo_keeps_zero_advantage_and_a_move_of_exactly_the_threshold() -> None:
    # A == 0 never drops, even after a fall of 0.5. 0.25 -> 0.5 is exactly 0.25 in fp32.
    _, metrics = _dppo_logits_grad_and_metrics(
        DPPOLoss.Config(divergence_threshold=0.25),
        trainer_probs=[0.10, 0.50],
        generator_probs=[0.60, 0.25],
        advantages=[0.0, 1.0],
    )

    assert metrics["loss/trust_region_dropped_frac"] == 0


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
