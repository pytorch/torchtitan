# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO loss: per-token clipped surrogate with asymmetric "clip-higher" bounds."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.config import Configurable

# Clamp |log(π_θ/π_old)| before exp() so a large generator/trainer logprob mismatch —
# notably the NaNs vLLM can emit under cudagraph — can't overflow exp() to inf/NaN.
_MAX_LOG_RATIO = 10.0


class DAPOLoss(Configurable):
    """Per-token clipped surrogate loss with DAPO-style "clip-higher".

    The same PPO clip as GRPO, but the importance ratio's lower and upper bounds are
    set independently (https://arxiv.org/abs/2503.14476): a larger upper bound keeps
    more probability mass on up-weighted tokens, countering entropy collapse. A token
    whose generator logprob is non-finite (vLLM under cudagraph) is dropped from the
    loss rather than trained as if it were on-policy.

    The scalar loss is the sum of per-token losses over loss positions divided by
    ``num_global_valid_tokens``, so gradient accumulation matches a single large batch.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        ratio_clip_low: float = 0.2
        """Lower clip: the importance ratio is clamped to ``>= 1 - ratio_clip_low``."""

        ratio_clip_high: float = 0.2
        """Upper clip: the ratio is clamped to ``<= 1 + ratio_clip_high``. Set larger
        than ``ratio_clip_low`` for DAPO "clip-higher" (e.g. 0.28)."""

    def __init__(self, config: Config) -> None:
        self.ratio_clip_low = config.ratio_clip_low
        self.ratio_clip_high = config.ratio_clip_high

    def __call__(
        self,
        policy_logprobs: torch.Tensor,
        generator_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        num_global_valid_tokens: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the per-token clip-higher surrogate loss.

        Args:
            policy_logprobs: [B, L] log π_θ(a_t | s_t) from the current policy.
            generator_logprobs: [B, L] log π_old(a_t | s_t) from the sampling policy.
            loss_mask: [B, L] bool mask; True for response tokens.
            advantages: [B, L] per-token advantages (0.0 for prompt/padding).
            num_global_valid_tokens: total response tokens across all microbatches and
                DP ranks; the loss denominator.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        # A non-finite generator logprob (notably under cudagraph) has no valid
        # old-policy reference, so DROP that token from the loss + denominator (cleaner
        # than nan->0, which trains it as if it were on-policy). `response_mask` keeps
        # the original tokens for the nan-frac metric.
        response_mask = loss_mask
        raw_log_ratio = policy_logprobs - generator_logprobs
        loss_mask = loss_mask & torch.isfinite(raw_log_ratio)
        log_ratio = torch.clamp(
            torch.nan_to_num(raw_log_ratio), -_MAX_LOG_RATIO, _MAX_LOG_RATIO
        )
        ratio = torch.exp(log_ratio)

        clipped_ratio = torch.clamp(
            ratio, 1 - self.ratio_clip_low, 1 + self.ratio_clip_high
        )
        token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        masked_loss = token_loss * loss_mask
        loss_denominator = max(num_global_valid_tokens, 1)
        loss = masked_loss.sum() / loss_denominator

        with torch.no_grad():
            masked_ratio = ratio * loss_mask
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": masked_ratio.sum() / loss_denominator,
                "loss/ratio_clipped_frac": (
                    (torch.abs(ratio - clipped_ratio) > 1e-6).float() * loss_mask
                ).sum()
                / loss_denominator,
                # Fraction of response tokens whose generator logprob is nan (dropped
                # above; tracked vs the original response_mask).
                "loss/generator_logprob_nan_frac": (
                    (~torch.isfinite(generator_logprobs)).float() * response_mask
                ).sum()
                / loss_denominator,
            }

        return loss, metrics
