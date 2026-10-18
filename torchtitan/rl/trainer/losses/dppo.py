# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DPPO (Divergence PPO) loss: policy gradient inside a trust region on each token's probability."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss
from torchtitan.rl.trainer.losses.common import normalize, policy_metrics, score_tokens


class DPPOLoss(BaseLoss):
    """Policy-gradient loss that drops a token once its probability moved too far from
    the generator's (https://arxiv.org/abs/2602.04879).

    PPO bounds the ratio `p / p_gen` of the sampled token, which is tight for rare
    tokens and loose for common ones. Raising a token from 1e-5 to 1e-3 is a ratio of
    100 and gets clipped. Lowering one from 0.99 to 0.80 is a ratio of 0.81 and is not,
    though it moves ~190x more probability. DPPO bounds the probability change instead.

    For each response token with advantage `A`:

        dropped    = sign(A) * (p - p_gen) > divergence_threshold
                     (symmetric=True: |p - p_gen| > divergence_threshold)
        token_loss = -A * detach(min(p / p_gen, max_ratio)) * log p,  0 if dropped
        loss       = sum(token_loss) / global_loss_token_counts

    This is the paper's Eq. 23 gradient: on kept tokens below `max_ratio`, it equals
    PPO's unclipped `-A * (p / p_gen) * grad log p`. Like `DAPOLoss`, it skips tokens
    with a non-finite generator logprob.

    Example (threshold 0.2):

        p_gen      = [0.20, 0.01,  0.40,  0.01]
        p          = [0.50, 0.05,  0.10,  0.30]
        advantages = [ 1.0,  1.0,  -1.0,  -1.0]
        dropped    = [True, False, True, False]
        # Token 1 moved +0.04: kept, though PPO clips its ratio of 5.
        # Token 3 moved against its advantage: kept, so its gradient pulls p back.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        divergence_threshold: float = 0.2
        """Drop a token whose probability moved more than this from the generator's, in
        the direction its advantage pushes it, or either way with `symmetric` (the
        paper's binary-TV delta)."""

        max_ratio: float = 5.0
        """Truncate the detached ratio `p / p_gen` at this value. Unlike a PPO clip, a
        truncated token keeps its gradient, scaled by `max_ratio`."""

        symmetric: bool = False
        """Also drop tokens that moved past the threshold against their advantage.
        With p_gen = 0.5 and threshold 0.2:

            p = 0.8, A = 1: moved +0.3, the way A pushes   dropped either way
            p = 0.2, A = 1: moved -0.3, against A          False: kept, True: dropped

        False is the paper's rule: a token that moved back keeps the gradient that pulls it
        back. True bounds |p - p_gen| on every token, whatever its advantage."""

        global_vocab_size: int | None = None
        """Full vocabulary size; required when TP shards the vocabulary."""

    def __init__(self, config: Config) -> None:
        self.divergence_threshold = config.divergence_threshold
        self.max_ratio = config.max_ratio
        self.symmetric = config.symmetric
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the trust-region-masked policy-gradient loss.

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

        # The ratio and the trust region are constants in the gradient.
        with torch.no_grad():
            # TODO: add the binary-KL variant (paper Eq. 14).
            # Left out to keep one variant with one default; the paper finds them on par.
            prob_change = torch.exp(tokens.trainer_logprobs) - torch.exp(
                tokens.generator_logprobs
            )
            if self.symmetric:
                prob_change = prob_change.abs()
            else:
                # How far p moved in the direction A pushes it; negative if it moved back.
                prob_change = tokens.advantages.sign() * prob_change
            dropped = (prob_change > self.divergence_threshold) & tokens.mask
            keep_mask = tokens.mask & ~dropped
            truncated_ratio = torch.clamp(tokens.ratio, max=self.max_ratio)

        token_loss = -tokens.advantages * truncated_ratio * tokens.trainer_logprobs
        loss = normalize((token_loss * keep_mask).sum(), global_loss_token_counts)

        with torch.no_grad():
            truncated = (tokens.ratio > self.max_ratio) & keep_mask
            loss_metric_sums = {
                "loss/trust_region_dropped_frac": dropped.float().sum(),
                "loss/ratio_truncated_frac": truncated.float().sum(),
            }
        return loss, policy_metrics(
            tokens,
            loss_metric_sums,
            loss=loss,
            global_loss_token_counts=global_loss_token_counts,
        )
