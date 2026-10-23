# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Steps every RL policy loss shares. A loss composes them around its own per-token loss:

    tokens = score_tokens(logits=logits, labels=labels, global_vocab_size=..., **loss_inputs)
    token_loss = ...                                    # the part each loss defines
    loss = normalize((token_loss * tokens.mask).sum(), global_loss_token_counts)
    metrics = policy_metrics(tokens, {...its own metric sums...}, loss=loss, ...)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.components.loss import compute_logprobs
from torchtitan.distributed.spmd_types import spmd_mesh_group

# Clamp |log(pi_theta/pi_old)| before exp() so a large generator/trainer
# logprob mismatch cannot overflow exp() to inf/NaN.
MAX_LOG_RATIO = 10.0


@dataclass(frozen=True, kw_only=True, slots=True)
class ScoredTokens:
    """What an RL loss reads about each token, all `[T]`: the batcher's inputs and the trainer's scores."""

    generator_logprobs: torch.Tensor
    """Logprob of the sampled token under the generator; can be non-finite (see `mask`)."""

    advantages: torch.Tensor
    """Advantage of the token; 0.0 on prompt and padding tokens."""

    trainer_logprobs: torch.Tensor
    """Logprob of the sampled token under the trainer, at the sampling temperature. Carries the gradient."""

    entropy: torch.Tensor
    """Entropy of the trainer's distribution at the sampling temperature; a metric, no gradient."""

    mask: torch.Tensor
    """Response tokens with a finite generator logprob: the tokens every loss trains on."""

    log_ratio: torch.Tensor
    """`trainer_logprobs - generator_logprobs` on `mask`, 0 elsewhere."""

    ratio: torch.Tensor
    """`exp(log_ratio)`, its log clamped to +-`MAX_LOG_RATIO`; 1 off `mask`. Carries the gradient."""


def score_tokens(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    global_vocab_size: int | None,
    generator_logprobs: torch.Tensor,
    temperature: torch.Tensor,
    loss_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> ScoredTokens:
    """Score each label under the trainer, next to the generator's logprob of it.

    Every RL loss passes its `**loss_inputs` (the tensors `TrainingMicrobatch.loss_kwargs`
    sends) straight here. So an input added here, with a default, reaches every loss
    without changing any loss's signature.

    Args:
        logits: [T, V] current-policy output; this rank's vocab shard under tensor parallel.
        labels: [T] pre-shifted target token ids.
        global_vocab_size: full vocabulary size; required when tensor parallel shards the vocabulary.
        generator_logprobs: [T] logprobs from the sampling policy.
        temperature: [T] temperature each token was sampled at.
        loss_mask: [T] bool; True for response tokens.
        advantages: [T] per-token advantages (0.0 for prompt/padding).

    Example:

        trainer logprobs   = [-1.0, -1.0, -3.0, -1.0]
        generator_logprobs = [-1.0, -2.0, -1.0,  nan]
        loss_mask          = [True, True, False, True]
        # -> mask      = [True, True, False, False]
        #    log_ratio = [0.0,  1.0,  0.0,   0.0]
        #    ratio     = [1.0,  2.72, 1.0,   1.0]
    """
    # Score tokens under the distribution vLLM sampled from. Dividing by a per-token fp32
    # tensor matches vLLM's op, so batch-invariant runs stay bitwise equal to the generator.
    logits = logits / temperature.unsqueeze(-1)
    trainer_logprobs, entropy = compute_logprobs(
        logits,
        labels,
        vocab_parallel_group=spmd_mesh_group("tp"),
        return_entropy=True,
        global_vocab_size=global_vocab_size,
    )
    # A non-finite generator logprob (notably under CUDA graph) has no valid
    # old-policy reference, so DROP that token from the loss + denominator (cleaner
    # than nan->0, which trains it as if it were on-policy).
    mask = loss_mask & torch.isfinite(generator_logprobs)
    raw_log_ratio = trainer_logprobs - generator_logprobs
    log_ratio = torch.where(mask, raw_log_ratio, torch.zeros_like(raw_log_ratio))
    ratio = torch.exp(torch.clamp(log_ratio, -MAX_LOG_RATIO, MAX_LOG_RATIO))
    return ScoredTokens(
        generator_logprobs=generator_logprobs,
        advantages=advantages,
        trainer_logprobs=trainer_logprobs,
        entropy=entropy,
        mask=mask,
        log_ratio=log_ratio,
        ratio=ratio,
    )


def policy_metrics(
    tokens: ScoredTokens,
    loss_metric_sums: dict[str, torch.Tensor],
    *,
    loss: torch.Tensor,
    global_loss_token_counts: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return the loss's own metrics plus the ones every RL loss logs, ready to SUM over DP ranks.

    Each sum is divided by `global_loss_token_counts`, so the trainer's SUM over microbatches
    and ranks is a mean over the step's loss tokens. `/max` values are kept as they are; the
    trainer takes their max. Each key needs a suffix `ChunkedLossWrapper` can combine across
    chunks: `mean` or `frac` (summed), `/max` (max).

    Example (global_loss_token_counts = 10):

        policy_metrics(tokens, {"loss/ratio_clipped_frac": tensor(3.0)}, loss=loss, ...)
        # -> {"loss/ratio_clipped_frac": 0.3, "loss/ratio_mean": ..., ..., "loss/mean": loss}
    """
    with torch.no_grad():
        metric_sums = {
            **loss_metric_sums,
            "loss/ratio_mean": (tokens.ratio * tokens.mask).sum(),
            # Mean per-token log-ratio (log p_trainer - log q_generator) over
            # sampled tokens. This is the k1 Monte-Carlo estimate of -KL(q || p).
            "bit_wise/logprob_diff/mean": tokens.log_ratio.float().sum(),
            "bit_wise/ratio_tokens_different/mean": (
                (tokens.log_ratio.abs() > 1e-6).float() * tokens.mask
            ).sum(),
            "bit_wise/logprob_diff/max": tokens.log_ratio.abs().max(),
            # Mean entropy of softmax(logits / temperature) over tokens used by the loss.
            "trainer/entropy/mean": (tokens.entropy * tokens.mask).sum(),
        }
        metrics = {
            key: (
                value
                if key.endswith("/max")
                else normalize(value, global_loss_token_counts)
            )
            for key, value in metric_sums.items()
        }
        metrics["loss/mean"] = loss.detach()
    return metrics


def normalize(
    value: torch.Tensor,
    global_loss_token_counts: torch.Tensor | None,
) -> torch.Tensor:
    """Divide `value` by the step's loss-token count, over all microbatches and DP ranks."""
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
