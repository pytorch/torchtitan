# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO loss: per-token clipped surrogate with asymmetric "clip-higher" bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import torch
import tyro

from torchtitan.components.loss import BaseLoss, compute_logprobs
from torchtitan.config import CompileConfig
from torchtitan.distributed.spmd_types import spmd_mesh_group

# Clamp |log(pi_theta/pi_old)| before exp() so a large generator/trainer
# logprob mismatch cannot overflow exp() to inf/NaN.
_MAX_LOG_RATIO = 10.0


def _normalize(
    value: torch.Tensor,
    global_valid_tokens: torch.Tensor | None,
) -> torch.Tensor:
    if global_valid_tokens is None:
        return value
    # A device tensor is required because the count is a mutable CUDA graph
    # input. Multiplication also preserves the established CUDA scalar-division
    # rounding for the float32 RL loss.
    return value * global_valid_tokens.clamp_min(1).reciprocal()


class DAPOLoss(BaseLoss):
    """Per-token clipped surrogate loss with DAPO-style "clip-higher".

    The same PPO clip as GRPO, but the importance ratio's lower and upper bounds are
    set independently (https://arxiv.org/abs/2503.14476): a larger upper bound keeps
    more probability mass on up-weighted tokens, countering entropy collapse. A token
    whose generator logprob is non-finite (vLLM under CUDA graph) is dropped from the
    loss rather than trained as if it were on-policy.

    The scalar loss is the sum of per-token losses over positions with a finite
    old-policy logprob divided by ``global_valid_tokens``, so gradient accumulation
    matches a single large batch. ``logits`` is the current-policy output passed to
    ``compute_logprobs``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        ratio_clip_low: float = 0.2
        """Lower clip: the importance ratio is clamped to ``>= 1 - ratio_clip_low``."""

        ratio_clip_high: float = 0.2
        """Upper clip: the ratio is clamped to ``<= 1 + ratio_clip_high``. Set larger
        than ``ratio_clip_low`` for DAPO "clip-higher" (e.g. 0.28)."""

        global_vocab_size: Annotated[int | None, tyro.conf.Suppress] = None
        """Full vocabulary size from the model spec, set when building RL configs.
        Leave unset for batch-invariant mode to retain the full-gather path."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        del compile_config
        self.ratio_clip_low = config.ratio_clip_low
        self.ratio_clip_high = config.ratio_clip_high
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        *,
        generator_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the per-token clip-higher surrogate loss.

        Args:
            logits: [T, V] current-policy output.
            labels: [T] pre-shifted target token ids.
            generator_logprobs: [T] logprobs from the sampling policy.
            loss_mask: [T] bool mask; True for response tokens.
            advantages: [T] per-token advantages (0.0 for prompt/padding).
            global_valid_tokens: total response tokens with finite generator logprobs
                across all microbatches and DP ranks; the loss denominator.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        trainer_logprobs, token_entropy = compute_logprobs(
            logits,
            labels,
            vocab_parallel_group=spmd_mesh_group("tp"),
            return_entropy=True,
            global_vocab_size=self.global_vocab_size,
        )
        # A non-finite generator logprob (notably under CUDA graph) has no valid
        # old-policy reference, so DROP that token from the loss + denominator (cleaner
        # than nan->0, which trains it as if it were on-policy).
        effective_loss_mask = loss_mask & torch.isfinite(generator_logprobs)
        raw_log_ratio = trainer_logprobs - generator_logprobs
        masked_log_ratio = torch.where(
            effective_loss_mask, raw_log_ratio, torch.zeros_like(raw_log_ratio)
        )
        log_ratio = torch.clamp(masked_log_ratio, -_MAX_LOG_RATIO, _MAX_LOG_RATIO)
        ratio = torch.exp(log_ratio)

        clipped_ratio = torch.clamp(
            ratio, 1 - self.ratio_clip_low, 1 + self.ratio_clip_high
        )
        token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        masked_loss = token_loss * effective_loss_mask
        loss = _normalize(masked_loss.sum(), global_valid_tokens)

        with torch.no_grad():
            diff_for_metrics = torch.where(
                effective_loss_mask,
                raw_log_ratio,
                torch.zeros_like(raw_log_ratio),
            )
            masked_ratio = ratio * effective_loss_mask
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": _normalize(masked_ratio.sum(), global_valid_tokens),
                "loss/ratio_clipped_frac": _normalize(
                    (
                        (torch.abs(ratio - clipped_ratio) > 1e-6).float()
                        * effective_loss_mask
                    ).sum(),
                    global_valid_tokens,
                ),
                # Mean per-token log-ratio (log p_trainer - log q_generator) over
                # sampled tokens. This is the k1 Monte-Carlo estimate of -KL(q || p).
                "bit_wise/logprob_diff/mean": _normalize(
                    diff_for_metrics.float().sum(), global_valid_tokens
                ),
                "bit_wise/ratio_tokens_different/mean": _normalize(
                    (
                        (diff_for_metrics.abs() > 1e-6).float() * effective_loss_mask
                    ).sum(),
                    global_valid_tokens,
                ),
                "bit_wise/logprob_diff/max": diff_for_metrics.abs().max(),
                # Mean trainer-policy entropy H(p) over tokens used by the loss.
                "trainer/entropy/mean": _normalize(
                    (token_entropy * effective_loss_mask).sum(), global_valid_tokens
                ),
            }

        return loss, metrics
