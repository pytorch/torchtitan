# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO loss: per-token clipped surrogate with asymmetric "clip-higher" bounds."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss, compute_logprobs
from torchtitan.config import CompileConfig

# Clamp |log(pi_theta/pi_old)| before exp() so a large generator/trainer
# logprob mismatch cannot overflow exp() to inf/NaN.
_MAX_LOG_RATIO = 10.0


class DAPOLoss(BaseLoss):
    """Per-token clipped surrogate loss with DAPO-style "clip-higher".

    The same PPO clip as GRPO, but the importance ratio's lower and upper bounds are
    set independently (https://arxiv.org/abs/2503.14476): a larger upper bound keeps
    more probability mass on up-weighted tokens, countering entropy collapse. A token
    whose generator logprob is non-finite (vLLM under cudagraph) is dropped from the
    loss rather than trained as if it were on-policy.

    The scalar loss is the sum of per-token losses over loss positions divided by
    ``global_valid_tokens``, so gradient accumulation matches a single large batch.
    ``logits`` is the current-policy output passed to ``compute_logprobs``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        ratio_clip_low: float = 0.2
        """Lower clip: the importance ratio is clamped to ``>= 1 - ratio_clip_low``."""

        ratio_clip_high: float = 0.2
        """Upper clip: the ratio is clamped to ``<= 1 + ratio_clip_high``. Set larger
        than ``ratio_clip_low`` for DAPO "clip-higher" (e.g. 0.28)."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        del compile_config
        self.ratio_clip_low = config.ratio_clip_low
        self.ratio_clip_high = config.ratio_clip_high

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | torch.Tensor | None = None,
        *,
        generator_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the per-token clip-higher surrogate loss.

        Args:
            logits: [B, L, V] current-policy output.
            labels: [B, L] pre-shifted target token ids.
            generator_logprobs: [B, L] logprobs from the sampling policy.
            loss_mask: [B, L] bool mask; True for response tokens.
            advantages: [B, L] per-token advantages (0.0 for prompt/padding).
            global_valid_tokens: total response tokens across all microbatches and
                DP ranks; the loss denominator.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        trainer_logprobs, token_entropy = compute_logprobs(
            logits, labels, return_entropy=True
        )
        # A non-finite generator logprob (notably under cudagraph) has no valid
        # old-policy reference, so DROP that token from the loss + denominator (cleaner
        # than nan->0, which trains it as if it were on-policy). `response_mask` keeps
        # the original tokens for the nan-frac metric.
        response_mask = loss_mask
        raw_log_ratio = trainer_logprobs - generator_logprobs
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
        if isinstance(global_valid_tokens, torch.Tensor):
            # aot_fx_trace path: a 0-d tensor keeps the denominator a graph
            # input rather than a baked constant. global_valid_tokens changes
            # every step, so baking the first step's value would silently
            # mis-scale the loss and grads on every later step.
            loss_denominator = torch.clamp(global_valid_tokens, min=1.0)
        elif global_valid_tokens is not None:
            loss_denominator = max(global_valid_tokens, 1)
        else:
            loss_denominator = 1
        loss = masked_loss.sum() / loss_denominator

        with torch.no_grad():
            diff = trainer_logprobs - generator_logprobs
            diff_for_metrics = torch.where(
                loss_mask,
                diff,
                torch.zeros_like(diff),
            )
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
                # Mean per-token log-ratio (log p_trainer - log q_generator) over
                # sampled tokens. This is the k1 Monte-Carlo estimate of -KL(q || p).
                "bit_wise/logprob_diff/mean": diff_for_metrics.float().sum()
                / loss_denominator,
                "bit_wise/ratio_tokens_different/mean": (
                    (diff_for_metrics.abs() > 1e-6).float() * loss_mask
                ).sum()
                / loss_denominator,
                "bit_wise/logprob_diff/max": diff_for_metrics.abs().max(),
                # Mean trainer-policy entropy H(p) over response tokens.
                "trainer/entropy/mean": (token_entropy * response_mask).sum()
                / loss_denominator,
            }

        return loss, metrics
