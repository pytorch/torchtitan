# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO loss: per-token clipped surrogate with asymmetric "clip-higher" bounds."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss
from torchtitan.rl.trainer.losses.common import normalize, policy_metrics, score_tokens


class DAPOLoss(BaseLoss):
    """Per-token clipped surrogate loss with DAPO-style "clip-higher".

    The same PPO clip as GRPO, but the importance ratio's lower and upper bounds are
    set independently (https://arxiv.org/abs/2503.14476): a larger upper bound keeps
    more probability mass on up-weighted tokens, countering entropy collapse. A token
    whose generator logprob is non-finite (vLLM under CUDA graph) is dropped from the
    loss rather than trained as if it were on-policy.

    The scalar loss is the sum of per-token losses over positions with a finite
    old-policy logprob divided by ``global_loss_token_counts``, so gradient accumulation
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

        global_vocab_size: int | None = None
        """Full vocabulary size from the model spec, set when building RL configs.
        Leave unset for batch-invariant mode to retain the full-gather path."""

    def __init__(self, config: Config) -> None:
        self.ratio_clip_low = config.ratio_clip_low
        self.ratio_clip_high = config.ratio_clip_high
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the per-token clip-higher surrogate loss.

        Args:
            logits: [T, V] current-policy output.
            labels: [T] pre-shifted target token ids.
            global_loss_token_counts: total response tokens with finite generator logprobs
                across all microbatches and DP ranks; the loss denominator.
            **loss_inputs: the [T] tensors `score_tokens` takes (generator logprobs,
                temperature, loss mask, advantages).

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        tokens = score_tokens(
            logits=logits,
            labels=labels,
            global_vocab_size=self.global_vocab_size,
            **loss_inputs,
        )
        clipped_ratio = torch.clamp(
            tokens.ratio, 1 - self.ratio_clip_low, 1 + self.ratio_clip_high
        )
        token_loss = -torch.min(
            tokens.ratio * tokens.advantages, clipped_ratio * tokens.advantages
        )
        loss = normalize((token_loss * tokens.mask).sum(), global_loss_token_counts)

        with torch.no_grad():
            was_clipped = (torch.abs(tokens.ratio - clipped_ratio) > 1e-6).float()
            loss_metric_sums = {
                "loss/ratio_clipped_frac": (was_clipped * tokens.mask).sum()
            }
        return loss, policy_metrics(
            tokens,
            loss_metric_sums,
            loss=loss,
            global_loss_token_counts=global_loss_token_counts,
        )
