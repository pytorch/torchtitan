# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base class of the per-token policy-gradient losses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from torchtitan.components.loss import compute_logprobs, Loss
from torchtitan.distributed.local_compile import local_compile
from torchtitan.distributed.spmd_types import spmd_mesh_group

# Clamp |log(pi_theta/pi_old)| before exp() so a large generator/trainer
# logprob mismatch cannot overflow exp() to inf/NaN.
_MAX_LOG_RATIO = 10.0


def _normalize(
    value: torch.Tensor,
    global_loss_token_counts: torch.Tensor | None,
) -> torch.Tensor:
    if global_loss_token_counts is None:
        return value
    global_loss_token_count = (
        global_loss_token_counts
        if global_loss_token_counts.ndim == 0
        else global_loss_token_counts[0]
    )
    # A device tensor is required because the count is a mutable CUDA graph
    # input. Multiplication also preserves the established CUDA scalar-division
    # rounding for the float32 RL loss.
    return value * global_loss_token_count.clamp_min(1).reciprocal()


class PolicyGradientLoss(Loss[torch.Tensor], ABC):
    """Per-token policy-gradient loss on tokens the generator sampled.

    Subclasses define ``token_loss``: each loss token's loss from its importance ratio
    ``p / q`` (trainer over generator prob) and its advantage. This class scores
    the tokens under the distribution the generator sampled from (``softmax(logits / T)``),
    drops tokens whose generator logprob is non-finite (vLLM under CUDA graph) rather than
    training them as if on-policy, and divides the summed loss by
    ``global_loss_token_counts``, so gradient accumulation matches a single large batch.

    Example (DAPOLoss with ratio_clip_high=0.2):

        ratio = p / q:   [1.0,  1.65, (q non-finite)]
        advantages:      [1.0,  1.0,  1.0]
        token_loss:      [-1.0, -1.2, (dropped)]   # ratio 1.65 clipped to 1.2
        loss = (-1.0 - 1.2) / global_loss_token_count
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        global_vocab_size: int | None = None
        """Full vocabulary size from the model spec, set when building RL configs.
        Leave unset for batch-invariant mode to retain the full-gather path."""

    def __init__(self, config: Config) -> None:
        self.global_vocab_size = config.global_vocab_size

    @abstractmethod
    def token_loss(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """[T] per-token loss.

        Args:
            ratio: [T] ``p / q``, with ``log(p / q)`` clamped to ``[-10, 10]``; 1 off the loss mask.
            advantages: [T] per-token advantages.
        """

    def token_metrics(self, ratio: torch.Tensor) -> dict[str, torch.Tensor]:
        """[T] per-token metric values, averaged over the loss tokens; none by default."""
        return {}

    @local_compile("loss", batch_invariant=False)
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        *,
        generator_logprobs: torch.Tensor,
        temperature: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the policy-gradient loss.

        Args:
            logits: [T, V] current-policy logits.
            labels: [T] pre-shifted target token ids.
            generator_logprobs: [T] logprobs from the sampling policy.
            temperature: [T] temperature each token was sampled at.
            loss_mask: [T] bool mask; True for response tokens.
            advantages: [T] per-token advantages (0.0 for prompt/padding).
            global_loss_token_counts: total response tokens with finite generator logprobs
                across all microbatches and DP ranks; the loss denominator.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        # Score tokens under the distribution vLLM sampled from.
        trainer_logprobs, token_entropy = compute_logprobs(
            logits,
            labels,
            vocab_parallel_group=spmd_mesh_group("tp"),
            return_entropy=True,
            global_vocab_size=self.global_vocab_size,
            temperature=temperature,
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

        token_loss = self.token_loss(ratio, advantages)

        masked_loss = token_loss * effective_loss_mask
        loss = _normalize(masked_loss.sum(), global_loss_token_counts)

        with torch.no_grad():
            diff_for_metrics = torch.where(
                effective_loss_mask,
                raw_log_ratio,
                torch.zeros_like(raw_log_ratio),
            )
            masked_ratio = ratio * effective_loss_mask
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": _normalize(
                    masked_ratio.sum(), global_loss_token_counts
                ),
                **{
                    key: _normalize(
                        (value * effective_loss_mask).sum(), global_loss_token_counts
                    )
                    for key, value in self.token_metrics(ratio).items()
                },
                # Mean per-token log-ratio (log p_trainer - log q_generator) over
                # sampled tokens. This is the k1 Monte-Carlo estimate of -KL(q || p).
                "bit_wise/logprob_diff/mean": _normalize(
                    diff_for_metrics.float().sum(), global_loss_token_counts
                ),
                "bit_wise/ratio_tokens_different/mean": _normalize(
                    (
                        (diff_for_metrics.abs() > 1e-6).float() * effective_loss_mask
                    ).sum(),
                    global_loss_token_counts,
                ),
                "bit_wise/logprob_diff/max": diff_for_metrics.abs().max(),
                # Mean entropy of softmax(logits / temperature) over tokens used by the loss.
                "trainer/entropy/mean": _normalize(
                    (token_entropy * effective_loss_mask).sum(),
                    global_loss_token_counts,
                ),
            }

        return loss, metrics
