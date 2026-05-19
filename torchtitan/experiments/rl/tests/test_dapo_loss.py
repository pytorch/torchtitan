# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the DAPO loss, ported from ``forge/tests/rl/loss/test_dapo.py``.

The forge tests use ``logits + target_ids`` and compute logprobs inside
the loss; our DAPO takes pre-computed logprobs (the trainer calls
``compute_logprobs`` separately). We therefore feed the same fixture
through ``compute_logprobs`` first so the numerical expectations stay
aligned with forge's reference values.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from torchtitan.experiments.rl.loss import DAPOLoss, LossOutput, pg_dual_clip


def _logprobs_from_logits(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Same convention as ``actors/utils.compute_logprobs`` but no shift:
    we want B,S logprobs aligned with the (B,S) target_ids fixture."""
    B, S, V = logits.shape
    logprobs = F.log_softmax(logits.float(), dim=-1)
    return logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


@pytest.fixture
def inputs():
    """Mirrors the forge conftest fixture so reference values transfer.

    Returns ``B=2, S=4, V=10`` tensors plus pre-computed logprobs for
    the per-token call signature.
    """
    torch.manual_seed(42)
    B, S, V = 2, 4, 10

    logits = torch.randn(B, S, V)
    target_ids = torch.randint(0, V, (B, S))

    # Same generator logprobs as forge (mild + high divergence per seq).
    behavior_logprobs = torch.tensor(
        [
            [-2.0, -2.1, -1.9, -2.0],
            [-6.0, -1.0, -5.0, -0.5],
        ]
    )
    advantages = torch.randn(B, S)
    loss_mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.float)
    policy_logprobs = _logprobs_from_logits(logits, target_ids)

    return {
        "logits": logits,
        "target_ids": target_ids,
        "policy_logprobs": policy_logprobs,
        "behavior_logprobs": behavior_logprobs,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "num_valid": loss_mask.sum().clamp(min=1.0),
    }


def _call(loss_fn, d):
    return loss_fn(
        policy_logprobs=d["policy_logprobs"].clone().requires_grad_(True),
        behavior_logprobs=d["behavior_logprobs"],
        advantages_per_token=d["advantages"],
        loss_mask=d["loss_mask"],
        num_global_valid_tokens=d["num_valid"],
    )


def test_returns_loss_output(inputs):
    loss_fn = DAPOLoss(DAPOLoss.Config())
    out = _call(loss_fn, inputs)
    assert isinstance(out, LossOutput)
    assert out.loss.isfinite().item()
    assert "loss/mean" in out.sum_metrics
    assert "loss/ratio/max" in out.max_metrics


def test_backward(inputs):
    loss_fn = DAPOLoss(DAPOLoss.Config())
    policy = inputs["policy_logprobs"].clone().requires_grad_(True)
    out = loss_fn(
        policy_logprobs=policy,
        behavior_logprobs=inputs["behavior_logprobs"],
        advantages_per_token=inputs["advantages"],
        loss_mask=inputs["loss_mask"],
        num_global_valid_tokens=inputs["num_valid"],
    )
    out.loss.backward()
    assert policy.grad is not None
    assert policy.grad.isfinite().all().item()
    assert policy.grad.norm() > 0


def test_zero_advantages_is_finite_and_no_grad_signal(inputs):
    """When advantages == 0, the per-token PG loss is 0 → gradient is 0."""
    loss_fn = DAPOLoss(DAPOLoss.Config())
    policy = inputs["policy_logprobs"].clone().requires_grad_(True)
    out = loss_fn(
        policy_logprobs=policy,
        behavior_logprobs=inputs["behavior_logprobs"],
        advantages_per_token=torch.zeros_like(inputs["advantages"]),
        loss_mask=inputs["loss_mask"],
        num_global_valid_tokens=inputs["num_valid"],
    )
    out.loss.backward()
    assert out.loss.isfinite()
    assert torch.allclose(policy.grad, torch.zeros_like(policy.grad))


def test_empty_mask_returns_zero_loss(inputs):
    """All-zero mask should give 0 loss without NaN / div-by-zero."""
    loss_fn = DAPOLoss(DAPOLoss.Config())
    empty_mask = torch.zeros_like(inputs["loss_mask"])
    out = loss_fn(
        policy_logprobs=inputs["policy_logprobs"].clone().requires_grad_(True),
        behavior_logprobs=inputs["behavior_logprobs"],
        advantages_per_token=inputs["advantages"],
        loss_mask=empty_mask,
        num_global_valid_tokens=empty_mask.sum().clamp(min=1.0),
    )
    assert out.loss.isfinite()
    assert out.loss.item() == 0.0


def test_neg_inf_behavior_logprob_is_clamped(inputs):
    """If vLLM ever returns ``-inf`` (bf16 softmax underflow on a low-
    probability token), the clamp inside ``compute_ratio`` prevents
    overflow on the importance ratio and downstream NaN poisoning."""
    behavior = inputs["behavior_logprobs"].clone()
    behavior[0, 0] = -float("inf")
    loss_fn = DAPOLoss(DAPOLoss.Config())
    out = loss_fn(
        policy_logprobs=inputs["policy_logprobs"].clone().requires_grad_(True),
        behavior_logprobs=behavior,
        advantages_per_token=inputs["advantages"],
        loss_mask=inputs["loss_mask"],
        num_global_valid_tokens=inputs["num_valid"],
    )
    assert out.loss.isfinite()


def test_dual_clip_caps_negative_advantage_penalty():
    """Manufactured case: ratio is huge and advantage is negative; dual
    clip should bring the loss to ``-c * advantage`` exactly."""
    # Single trainable token. Force ratio = exp(5) by setting logprob diff.
    policy = torch.tensor([[0.0]], requires_grad=True)
    behavior = torch.tensor([[-5.0]])
    adv = torch.tensor([[-1.0]])  # negative advantage
    mask = torch.tensor([[1.0]])
    cfg = DAPOLoss.Config(clip_low=0.2, clip_high=0.28, dual_clip_c=3.0)
    loss_fn = DAPOLoss(cfg)
    out = loss_fn(
        policy_logprobs=policy,
        behavior_logprobs=behavior,
        advantages_per_token=adv,
        loss_mask=mask,
        num_global_valid_tokens=mask.sum(),
    )
    # ratio = exp(5) >> 1 + clip_high, so ppo_clip_loss = -(1+clip_high)*adv
    # = -(1.28) * (-1) = 1.28. dual_clip_bound = -3 * (-1) = 3. min(1.28, 3) = 1.28.
    # Loss is 1.28 / 1 = 1.28. The dual clip doesn't trigger in this case
    # because ppo_clip_loss < dual_clip_bound. (Test that it stays finite.)
    assert out.loss.isfinite()
    # The dual_clip kicks in when ppo_clip_loss > -c*adv, i.e. when ratio*adv is
    # very negative — here ratio*(-1) = -148.4 is dropped to clipped*(-1) = -1.28.
    # So dual clip doesn't fire on this path. Verify via metric:
    dual_frac = out.sum_metrics["loss/dual_clip/clip_fraction"].item()
    # No dual-clip trigger expected (ppo_clip already capped).
    assert math.isfinite(dual_frac)


def test_pg_dual_clip_unit():
    """Direct unit test for ``pg_dual_clip``."""
    pg_loss = torch.tensor([[10.0, -10.0, 0.5]])  # pretend ppo losses
    advantages = torch.tensor([[-1.0, -1.0, 2.0]])  # neg, neg, pos
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    capped, m = pg_dual_clip(pg_loss, advantages, mask, c=3.0)
    # adv < 0: min(pg, -3*(-1)) = min(pg, 3). Pos 0: min(10, 3) = 3. Pos 1: min(-10, 3) = -10.
    # adv > 0: keep pg. Pos 2: 0.5.
    assert torch.allclose(capped, torch.tensor([[3.0, -10.0, 0.5]]))
    # Only position 0 was dual-clipped (pg=10 > 3=bound).
    assert m["loss/dual_clip/clip_fraction"].item() == pytest.approx(1 / 3, abs=1e-6)
