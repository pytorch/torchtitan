# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-token loss primitives shared across DAPO/GRPO/GSPO/etc.

Ported from ``forge/src/forge/rl/loss/ops.py``. The metric dicts use
the SUM/MAX split described in :mod:`loss.types`: each op returns a
``(tensor, sum_metrics, max_metrics)`` triple so the caller can stitch
them into the final :class:`LossOutput`.
"""

from __future__ import annotations

import torch

from torchtitan.experiments.rl.loss.types import AggType

__all__ = ["aggregate", "compute_ratio", "masked_mean", "pg_ppo_clip"]


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    loss_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sum(values * mask) / divisor`` with optional explicit divisor.

    In distributed settings pass ``loss_scale = global_loss_mask_sum``
    so the local masked-sum / global-N produces the correct global
    mean once SUM-reduced across DP ranks.
    """
    masked_sum = (values * mask).sum()
    if loss_scale is not None:
        divisor = loss_scale.clamp(min=1.0)
    else:
        divisor = mask.sum().clamp(min=1.0)
    return masked_sum / divisor


def compute_ratio(
    policy_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    mask: torch.Tensor,
    *,
    log_ratio_clamp: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Token-level importance ratio ``r_t = exp(policy - behavior)``.

    The log-ratio is computed only at ``mask=1`` positions (off-mask
    positions get ``log_ratio = 0`` so a stale ``behavior_logprob = 0``
    on a prompt position can't contribute a spurious ratio), and then
    clamped to ``[-log_ratio_clamp, +log_ratio_clamp]`` so a vLLM bf16
    softmax underflow (``-inf`` logprob on a low-probability sampled
    token) can't blow the sum to NaN downstream (``NaN * 0 == NaN``
    in IEEE 754).

    Returns:
        (ratio, log_ratio, sum_metrics)
    """
    mask_bool = mask > 0.5
    log_ratio = torch.where(
        mask_bool,
        policy_logprobs - behavior_logprobs,
        torch.zeros_like(policy_logprobs),
    )
    log_ratio = torch.clamp(log_ratio, min=-log_ratio_clamp, max=log_ratio_clamp)
    ratio = torch.exp(log_ratio)

    with torch.no_grad():
        denom = mask.sum().clamp(min=1.0)
        sum_metrics = {
            "loss/ratio/mean": (ratio * mask).sum() / denom,
            "loss/kl_policy/mean": (-log_ratio * mask).sum() / denom,
        }
    return ratio, log_ratio, sum_metrics


def aggregate(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    agg_type: AggType = "token_mean",
    loss_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce per-token loss to scalar with the chosen strategy.

    - ``token_mean``  — ``sum(loss * mask) / loss_scale`` (DAPO default).
      In distributed training pass ``loss_scale = global_loss_mask_sum``.
    - ``fixed_horizon`` — ``sum(loss * mask) / (B * S)`` (constant
      denominator, removes length bias).
    - ``sequence_mean`` — mean per sequence, then mean across batch
      (DR-GRPO; length-biased).
    """
    if agg_type == "token_mean":
        loss = masked_mean(per_token_loss, mask, loss_scale)
    elif agg_type == "fixed_horizon":
        loss = (per_token_loss * mask).sum() / max(mask.numel(), 1)
    elif agg_type == "sequence_mean":
        seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
        seq_means = (per_token_loss * mask).sum(dim=-1) / seq_lengths
        loss = seq_means.sum() / max(seq_means.numel(), 1)
    else:
        raise ValueError(f"Unknown agg_type: {agg_type}")
    return loss, {}


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO clipped surrogate (Schulman et al. 2017, arXiv:1707.06347).

    ``L = max(-r * A, -clip(r, 1 - clip_low, 1 + clip_high) * A)``.

    Asymmetric clip (``clip_high > clip_low``) is the DAPO modification
    — allows more upward exploration on low-probability tokens.

    Returns:
        (per_token_loss, sum_metrics)
    """
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    unclipped_loss = -ratio * advantages
    clipped_loss = -clipped_ratio * advantages
    pg_loss = torch.maximum(unclipped_loss, clipped_loss)

    with torch.no_grad():
        denom = mask.sum().clamp(min=1.0)
        mask_b = mask.bool()
        clipped_high = (ratio > 1 + clip_high) & mask_b
        clipped_low = (ratio < 1 - clip_low) & mask_b
        pos_adv = advantages > 0
        neg_adv = advantages < 0
        sum_metrics = {
            "loss/clip/clipped_ratio/mean": (clipped_ratio * mask).sum() / denom,
            "loss/clip/high_fraction": ((clipped_high & pos_adv).float() * mask).sum()
            / denom,
            "loss/clip/low_fraction": ((clipped_low & neg_adv).float() * mask).sum()
            / denom,
        }
    return pg_loss, sum_metrics
