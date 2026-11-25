# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DPPO (Divergence PPO) loss: policy gradient inside a trust region on each token's probability."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchtitan.rl.trainer.losses.policy_gradient import (
    MAX_LOG_RATIO,
    PolicyGradientLoss,
    ScoredTokens,
)


class DPPOLoss(PolicyGradientLoss):
    """Policy-gradient loss that drops a token once its probability moved too far from
    the generator's (https://arxiv.org/abs/2602.04879).

    PPO bounds the ratio ``p / p_gen`` of the sampled token, which is tight for rare
    tokens and loose for common ones. Raising a token from 1e-5 to 1e-3 is a ratio of
    100 and gets clipped. Lowering one from 0.99 to 0.80 is a ratio of 0.81 and is not,
    though it moves ~190x more probability. DPPO bounds the probability change instead.

    For each loss token with advantage ``A``:

        dropped    = sign(A) * (p - p_gen) > divergence_threshold
        token_loss = -A * sg(min(p / p_gen, max_ratio)) * log p,  0 if dropped

    This is the paper's Eq. 23 gradient: on kept tokens below ``max_ratio``, it equals
    PPO's unclipped ``-A * (p / p_gen) * grad(log p)``.

    Example (threshold 0.2):

        p_gen      = [0.20, 0.01,  0.40,  0.01]
        p          = [0.50, 0.05,  0.10,  0.30]
        advantages = [ 1.0,  1.0,  -1.0,  -1.0]
        dropped    = [True, False, True, False]
        # Token 1 moved +0.04: kept, though PPO clips its ratio of 5.
        # Token 3 moved against its advantage: kept, so its gradient pulls p back.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(PolicyGradientLoss.Config):
        divergence_threshold: float = 0.2
        """Drop a token whose probability moved more than this from the generator's, in
        the direction its advantage pushes it (the paper's binary-TV delta)."""

        max_ratio: float = 5.0
        """Truncate the detached ratio ``p / p_gen`` at this value. Unlike a PPO clip, a
        truncated token keeps its gradient, scaled by ``max_ratio``."""

        def __post_init__(self) -> None:
            if self.divergence_threshold <= 0 or self.max_ratio < 1:
                raise ValueError(
                    "DPPOLoss needs divergence_threshold > 0 and max_ratio >= 1, got "
                    f"{self.divergence_threshold} and {self.max_ratio}"
                )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.divergence_threshold = config.divergence_threshold
        self.max_ratio = config.max_ratio

    def token_loss(self, tokens: ScoredTokens) -> torch.Tensor:
        # TODO: add the binary-KL variant (paper Eq. 14); the paper finds it on par.
        dropped = self._dropped(
            tokens.trainer_logprobs, tokens.generator_logprobs, tokens.advantages
        )
        weight = torch.clamp(tokens.ratio, max=self.max_ratio) * ~dropped
        return -tokens.advantages * weight.detach() * tokens.trainer_logprobs

    def log_token_weight(
        self,
        log_ratio: torch.Tensor,
        advantages: torch.Tensor,
        trainer_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        # The weight reads PolicyGradientLoss's ratio, whose |log r| is clamped.
        clamped = torch.clamp(log_ratio, -MAX_LOG_RATIO, MAX_LOG_RATIO)
        log_weight = torch.clamp(clamped, max=math.log(self.max_ratio))
        # log p_gen = log p - log r.
        dropped = self._dropped(
            trainer_logprobs, trainer_logprobs - log_ratio, advantages
        )
        return torch.where(dropped, -torch.inf, log_weight)

    def token_metrics(self, tokens: ScoredTokens) -> dict[str, torch.Tensor]:
        dropped = self._dropped(
            tokens.trainer_logprobs, tokens.generator_logprobs, tokens.advantages
        )
        return {
            "loss/trust_region_dropped_frac": dropped.float(),
            "loss/ratio_truncated_frac": (
                (tokens.ratio > self.max_ratio) & ~dropped
            ).float(),
        }

    def _dropped(
        self,
        trainer_logprobs: torch.Tensor,
        generator_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        # How far p moved in the direction A pushes it; negative if it moved back.
        prob_change = advantages.sign() * (
            torch.exp(trainer_logprobs) - torch.exp(generator_logprobs)
        )
        return prob_change.detach() > self.divergence_threshold
