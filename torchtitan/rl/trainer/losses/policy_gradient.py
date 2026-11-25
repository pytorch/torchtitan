# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base class of the per-token policy-gradient losses."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import torch

from torchtitan.components.loss import compute_logprobs, Loss
from torchtitan.distributed.local_compile import local_compile
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.rl.trainer.losses.score_centering import ScoreCentering

# Clamp |log(pi_theta/pi_old)| before exp() so a large generator/trainer
# logprob mismatch cannot overflow exp() to inf/NaN.
MAX_LOG_RATIO = 10.0


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


@dataclass(frozen=True, slots=True)
class ScoredTokens:
    """Per-token inputs of a policy loss's surrogate, all [T]."""

    trainer_logprobs: torch.Tensor
    """``log p`` under ``softmax(logits / T)``; differentiable."""
    generator_logprobs: torch.Tensor
    """``log q``; equal to the detached ``log p`` off the loss mask, so a surrogate never
    sees the non-finite logprobs of dropped tokens."""
    ratio: torch.Tensor
    """``p / q``, with ``log(p / q)`` clamped to ``[-MAX_LOG_RATIO, MAX_LOG_RATIO]``;
    differentiable; 1 off the loss mask."""
    advantages: torch.Tensor


class PolicyGradientLoss(Loss[torch.Tensor]):
    """Per-token policy-gradient loss on tokens the generator sampled.

    Subclasses define ``token_loss``: each loss token's loss from its ``ScoredTokens``
    (trainer and generator logprobs, their ratio ``p / q``, the advantage), and
    ``log_token_weight``: the weight ``w`` that loss puts on the token's score, so that the
    gradient of ``token_loss`` is ``-A * w(p / q) * grad(log p)``. ``score_centering`` uses
    ``w`` to cancel the drift a trainer/generator mismatch adds to the gradient. This class scores
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

        score_centering: ScoreCentering.Config | None = None
        """Score centering (arXiv 2609.20807); None disables it. It needs the generator's
        top-k logprobs, which the controller then requests on every training rollout."""

    def __init__(self, config: Config) -> None:
        self.global_vocab_size = config.global_vocab_size
        self.score_centering = (
            None if config.score_centering is None else config.score_centering.build()
        )

    @abstractmethod
    def token_loss(self, tokens: ScoredTokens) -> torch.Tensor:
        """[T] per-token loss."""

    @abstractmethod
    def log_token_weight(
        self, log_ratio: torch.Tensor, advantages: torch.Tensor
    ) -> torch.Tensor:
        """``log w``: the log of the weight ``token_loss`` puts on a token's score.

        Args:
            log_ratio: ``log(p / q)`` of any tokens, not clamped.
            advantages: per-token advantages, broadcastable to ``log_ratio``.

        Returns:
            ``log w``, -inf where ``token_loss`` has no gradient (``w = 0``).
        """

    def token_metrics(self, tokens: ScoredTokens) -> dict[str, torch.Tensor]:
        """[T] per-token metric values, averaged over the loss tokens; none by default.

        Keys end in ``_frac`` or ``_mean``: ``combine_microbatch_metrics`` averages only those
        across microbatches.
        """
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
        generator_topk_token_ids: torch.Tensor | None = None,
        generator_topk_logprobs: torch.Tensor | None = None,
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
            generator_topk_token_ids: [T, k] the generator's top-k token ids, with
                ``score_centering`` only.
            generator_topk_logprobs: [T, k] the generator's logprobs of those ids.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        vocab_parallel_group = spmd_mesh_group("tp")
        # Score tokens under the distribution vLLM sampled from.
        trainer_logprobs, token_entropy = compute_logprobs(
            logits,
            labels,
            vocab_parallel_group=vocab_parallel_group,
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
        log_ratio = torch.clamp(masked_log_ratio, -MAX_LOG_RATIO, MAX_LOG_RATIO)
        ratio = torch.exp(log_ratio)
        tokens = ScoredTokens(
            trainer_logprobs=trainer_logprobs,
            generator_logprobs=torch.where(
                effective_loss_mask, generator_logprobs, trainer_logprobs.detach()
            ),
            ratio=ratio,
            advantages=advantages,
        )

        token_loss = self.token_loss(tokens)
        score_centering_metrics = {}
        if self.score_centering is not None:
            assert (
                generator_topk_token_ids is not None
                and generator_topk_logprobs is not None
            ), "score centering needs the generator's top-k logprobs"
            correction, score_centering_metrics = self.score_centering(
                logits,
                temperature=temperature,
                advantages=advantages,
                loss_mask=effective_loss_mask,
                generator_topk_token_ids=generator_topk_token_ids,
                generator_topk_logprobs=generator_topk_logprobs,
                log_token_weight=self.log_token_weight,
                vocab_parallel_group=vocab_parallel_group,
                global_vocab_size=self.global_vocab_size,
            )
            token_loss = token_loss + advantages * correction

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
                    for key, value in (
                        self.token_metrics(tokens) | score_centering_metrics
                    ).items()
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
