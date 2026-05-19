# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO: Decoupled clip + Dynamic sAmpling Policy Optimization.

Ported from ``forge/src/forge/rl/loss/dapo.py``. The dual-clip helper
and the loss body are the same per-token math; the only difference
is the metric carrier (forge's ``list[Metric]`` is split into
SUM/MAX dicts to match the trainer's existing all-reduce pattern).

Reference: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning
System at Scale" (2025), https://arxiv.org/abs/2503.14476.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.config import Configurable
from torchtitan.experiments.rl.loss.ops import aggregate, compute_ratio, pg_ppo_clip
from torchtitan.experiments.rl.loss.types import AggType, LossOutput

__all__ = ["DAPOLoss", "pg_dual_clip"]


def pg_dual_clip(
    pg_loss: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    c: float = 3.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DAPO's dual-clip for negative advantages.

    Standard PPO clipping over-penalizes "wrong" tokens that may
    actually be productive exploration. Dual-clip adds a ceiling:
    penalties on negative-advantage tokens can't exceed ``c * |A|``.

    Formula: ``L = min(L_PPO, -c * A)`` when ``A < 0``, else ``L_PPO``.

    Returns:
        (per_token_loss, sum_metrics)
    """
    dual_clip_bound = -c * advantages
    loss = torch.where(
        advantages < 0,
        torch.minimum(pg_loss, dual_clip_bound),
        pg_loss,
    )

    with torch.no_grad():
        denom = mask.sum().clamp(min=1.0)
        neg_mask = (advantages < 0) & mask.bool()
        was_dual_clipped = (pg_loss > dual_clip_bound) & neg_mask
        sum_metrics = {
            "loss/dual_clip/clip_fraction": (was_dual_clipped.float() * mask).sum()
            / denom,
        }
    return loss, sum_metrics


class DAPOLoss(Configurable):
    """DAPO loss — asymmetric clip + dual clip, no reference KL.

    Per-token:
        ``L_clip = max(-r*A, -clip(r, 1-clip_low, 1+clip_high)*A)``
        ``L_t  = min(L_clip, -c*A)`` when ``A < 0``, else ``L_clip``
    Aggregated: ``sum(L_t * mask) / loss_scale``, where ``loss_scale``
    is the global loss-mask sum across DP ranks.

    Differences from vanilla GRPO:

    - **Clip-higher** (``clip_high > clip_low``): asymmetric trust
      region — more headroom upward for low-probability tokens.
    - **Dual-clip**: ``-c * A`` ceiling on negative-advantage tokens.
    - **Token-level aggregation**: shard-invariant gradient via the
      global ``loss_scale``.

    Not in this loss (preprocessing concerns the orchestrator owns):
    Dynamic sampling, overlong-reward shaping.

    The trainer already computes ``policy_logprobs`` (via
    :func:`compute_logprobs` in ``actors/utils.py``); this loss takes
    them directly rather than recomputing from logits.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_low: float = 0.2
        """Lower clip bound (DAPO paper default 0.2)."""

        clip_high: float = 0.28
        """Upper clip bound (DAPO paper default 0.28). Must be > clip_low."""

        dual_clip_c: float = 3.0
        """Dual-clip cap for negative advantages (DAPO paper default 3.0)."""

        agg_type: AggType = "token_mean"
        """Aggregation strategy; ``token_mean`` is DAPO's recommendation."""

    def __init__(self, config: Config) -> None:
        self.clip_low = config.clip_low
        self.clip_high = config.clip_high
        self.dual_clip_c = config.dual_clip_c
        self.agg_type = config.agg_type

    def __call__(
        self,
        *,
        policy_logprobs: torch.Tensor,
        behavior_logprobs: torch.Tensor,
        advantages_per_token: torch.Tensor,
        loss_mask: torch.Tensor,
        num_global_valid_tokens: torch.Tensor,
    ) -> LossOutput:
        ratio, _, ratio_sum = compute_ratio(
            policy_logprobs, behavior_logprobs, loss_mask
        )
        pg_loss, clip_sum = pg_ppo_clip(
            ratio,
            advantages_per_token,
            loss_mask,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
        )
        pg_loss, dual_sum = pg_dual_clip(
            pg_loss, advantages_per_token, loss_mask, c=self.dual_clip_c
        )
        loss, _ = aggregate(
            pg_loss, loss_mask, self.agg_type, loss_scale=num_global_valid_tokens
        )

        with torch.no_grad():
            ratio_masked = ratio * loss_mask
            sum_metrics: dict[str, torch.Tensor] = {
                "loss/mean": loss.detach(),
                **ratio_sum,
                **clip_sum,
                **dual_sum,
            }
            max_metrics: dict[str, torch.Tensor] = {
                "loss/ratio/max": ratio_masked.max(),
            }
        return LossOutput(loss=loss, sum_metrics=sum_metrics, max_metrics=max_metrics)
